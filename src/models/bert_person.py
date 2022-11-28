import argparse
import copy
import csv
import heapq
import json
import math
import os
import socket
from builtins import breakpoint
from calendar import c
from gettext import translation
from importlib.metadata import metadata
from typing import Any, List

import cv2
import dill
import gdown
import joblib
import neural_renderer as nr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from torch.utils.data import DataLoader
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import src
from src.ActivityNet.Evaluation.get_ava_performance import run_evaluation
from src.models.components.encoding_head import EncodingHead_3c
from src.models.components.joint_mapper import JointMapper, smpl_to_openpose
from src.models.components.smpl import create
from src.models.components.smpl_head_re import SMPLHeadRE
from src.models.components.vit import ViT
from src.utils import losses
from src.utils.losses import compute_loss
from src.utils.renderer import Renderer
from src.utils.smooth_pose import smooth_pose, smooth_tran
from src.utils.utils import (compute_uvsampler, draw_text, get_colors,
                             numpy_to_torch_image, perspective_projection)

log = src.utils.get_pylogger(__name__)

import csv
import os

import matplotlib.pyplot as plt
# import joblib
import numpy as np

class_sum = joblib.load('data/ava/class_sum.pkl')



class RAKEL(nn.Module):
    def __init__(self, dim_in, num_labels, num_ensembles=10, labelset_size=10, dropout=0.1):
        super(RAKEL, self).__init__()
        
        self.num_labels = num_labels
        self.fc_0 = nn.Linear(dim_in, num_labels)
        self.fc_i = nn.ModuleList([nn.Linear(dim_in, num_labels) for i in range(num_ensembles)])
        self.dropout = nn.Dropout(dropout)
        np.random.seed(0)
        self.label_set = [np.random.choice(num_labels, labelset_size, replace=False) for i in range(num_ensembles)]

    def forward(self, x):

        x_ = x
        x = self.dropout(x)
        x = self.fc_0(x)        
        for i in range(len(self.fc_i)):
            x2 = self.dropout(x_)
            x[:, self.label_set[i]] += self.fc_i[i](x2)[:, self.label_set[i]]
    
        return x    
    
def read_ava_pkl(pkl_file, refence_file=None, best=False):
    def get_actions(pkl_file):
            
        data          = joblib.load(pkl_file)
        mAP_values    = data[0]
        catagories    = data[1]
        catagories_   = {}
        map_per_class = {}
        for m in catagories: catagories_[m['name']] = m['id']
        for key in mAP_values.keys():
            if("PascalBoxes_PerformanceByCategory/AP@0.5IOU" in key):
                key_                         = key.split("PascalBoxes_PerformanceByCategory/AP@0.5IOU/")[1]
                # if("person" in key_ or "an object" in key_): continue
                map_per_class[key_]          = mAP_values[key]*100

        actions    = list(map_per_class.keys())
        action_map = list(map_per_class.values())
        counts     = [class_sum[catagories_[i]] for i in actions]
        idx_       = np.argsort(counts)[::-1]
        actions    = np.array(actions)[idx_]
        action_map = np.array(action_map)[idx_]

        return actions, action_map
        
        
    if(best):
        print("reading best file from ", pkl_file)
        files = os.listdir(pkl_file)
        score = []
        results = []
        for file_ in files:
            actions, action_map = get_actions(os.path.join(pkl_file, file_))
            results.append([actions, action_map])
            if(refence_file is None):
                score.append(np.mean(action_map))
            else:
                actions_, action_map_ = refence_file[0], refence_file[1]
                sum_ = 0
                for i in range(len(actions)):
                    if(action_map[i]>action_map_[i]):
                        sum_ += action_map[i]-action_map_[i]
                score.append(sum_)
                
        idx_ = np.argsort(score)[::-1]
        print(score)
        return results[idx_[0]]
    else:  
        if(".pkl" in pkl_file):
            # read the given pkl file
            print("reading ", pkl_file)
            pkl_file = pkl_file
        else:
            # read the last pkl file
            print("reading last file.")
            files = os.listdir(pkl_file)
            filesId = [int(file.split(".")[0]) for file in files]
            ids     = np.argsort(filesId)
            pkl_file = os.path.join(pkl_file, files[-1])
            print(filesId)
            
        return get_actions(pkl_file)
    
    
class BERT_PERSON_LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):

        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        
        self.train_acc_1  = Accuracy(top_k=1).to(self.device)
        self.train_acc_5  = Accuracy(top_k=5).to(self.device)
        self.val_acc_1    = Accuracy(top_k=1).to(self.device)
        self.val_acc_5    = Accuracy(top_k=5).to(self.device)
        
        # if(self.cfg.in_feat!=self.cfg.extra_feat.action.en_dim):
        #     self.cfg.in_feat = self.cfg.in_feat + self.cfg.extra_feat.action.en_dim
        # self.cfg.in_feat  = 229
        # self.cfg.in_feat  = 997
        # self.cfg.in_feat  = 1253
        self.encoder      = ViT(   
                                opt         = self.cfg, 
                                dim         = self.cfg.in_feat,
                                depth       = self.cfg.vit.depth,
                                heads       = self.cfg.vit.heads,
                                mlp_dim     = self.cfg.vit.mlp_dim,
                                dim_head    = self.cfg.vit.dim_head,
                                dropout     = self.cfg.vit.dropout,
                                emb_dropout = self.cfg.vit.emb_dropout,
                                droppath    = self.cfg.vit.droppath,
                                device      = self.device,
                                )

        
        
        ######################################
        ############ SMPL stuff ##############
        ######################################
        smplx_params  = {k.lower(): v for k,v in dict(self.cfg.smpl_cfg.SMPL).items()}
        joint_mapper  = JointMapper(smpl_to_openpose(model_type=self.cfg.smpl_cfg.SMPL.MODEL_TYPE))
        self.smpl     = create(**smplx_params,
                                  batch_size=1,
                                  joint_mapper = joint_mapper,
                                  create_betas=False,
                                  create_body_pose=False,
                                  create_global_orient=False,
                                  create_left_hand_pose=False,
                                  create_right_hand_pose=False,
                                  create_expression=False,
                                  create_leye_pose=False,
                                  create_reye_pose=False,
                                  create_jaw_pose=False,
                                  create_transl=False,)
        
        self.smpl_head              = nn.ModuleList([SMPLHeadRE(self.cfg.smpl_cfg, self.cfg) for _ in range(self.cfg.num_smpl_heads)])
        
        self.loca_head              = nn.ModuleList([nn.Sequential(
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat),
                                            nn.ReLU(),         
                                            nn.Linear(self.cfg.in_feat, 3)
                                        ) for _ in range(self.cfg.num_smpl_heads)])
        
        # for 1802a
        ava_action_classes          = self.cfg.ava.num_action_classes if not(self.cfg.ava.predict_valid) else self.cfg.ava.num_valid_action_classes
        if(self.cfg.use_rakel):
            self.action_head_ava    = nn.ModuleList([RAKEL(self.cfg.in_feat, ava_action_classes, num_ensembles=10, labelset_size=10, dropout=0.1) for _ in range(self.cfg.num_smpl_heads)])
        else:
            self.action_head_ava        = nn.ModuleList([nn.Sequential(    
                                                # nn.Dropout(self.cfg.ava.head_dropout),
                                                nn.Linear(self.cfg.in_feat, ava_action_classes),
                                            ) for _ in range(self.cfg.num_smpl_heads)])
        
        self.action_head_kinetics   = nn.Sequential(
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat),
                                            nn.ReLU(),         
                                            nn.Linear(self.cfg.in_feat, self.cfg.kinetics.num_action_classes),
                                        )
        
        self.encoding_head    = EncodingHead_3c(opt=self.cfg, img_H=256, img_W=256) 
        checkpoint_file = torch.load('/private/home/jathushan/3D/TENET/_DATA/hmar_v2_weights.pth')
        state_dict_filt = {}
        for k, v in checkpoint_file['model'].items():
            if ("encoding_head" in k): state_dict_filt.setdefault(k[5:].split("encoding_head.")[1], v)
        self.encoding_head.load_state_dict(state_dict_filt, strict=True)
        self.encoding_head.eval()
        for param in self.encoding_head.parameters():
            param.requires_grad = False
                   
            
        if(self.cfg.finetune):
            self.encoder.eval()
            self.smpl_head.eval()
            self.loca_head.eval()
            self.action_head_kinetics.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.smpl_head.parameters():
                param.requires_grad = False
            for param in self.loca_head.parameters():
                param.requires_grad = False
            for param in self.action_head_kinetics.parameters():
                param.requires_grad = False
                
        self.best_val_acc = 0
        try:
            actions_, action_map_ = read_ava_pkl(self.cfg.storage_folder + "/results/", best=True)
            self.best_val_acc = np.mean(actions_)
        except Exception as e:
            log.warning(e)
            pass
            
        texture_file          = np.load(self.cfg.smpl_cfg.SMPL.TEXTURE)
        self.faces_cpu        = texture_file['smpl_faces'].astype('uint32')
         
        vt                    = texture_file['vt']
        ft                    = texture_file['ft']
        uv_sampler            = compute_uvsampler(vt, ft, tex_size=6)
        uv_sampler            = torch.tensor(uv_sampler, dtype=torch.float)
        uv_sampler            = uv_sampler.unsqueeze(0)
        self.F                = uv_sampler.size(1)   
        self.T                = uv_sampler.size(2) 
        self.uv_sampler       = uv_sampler.view(-1, self.F, self.T*self.T, 2)
        self.uv_sampler       = self.uv_sampler.to(self.device)
        
        if(self.cfg.render.engine=="PYR"):
            self.py_render        = Renderer(focal_length=self.cfg.smpl_cfg.EXTRA.FOCAL_LENGTH, img_res=self.cfg.render.res*self.cfg.render.render_up_scale, faces=self.faces_cpu)
        elif(self.cfg.render.engine=="NMR"):
            self.nmr_renderer     = nr.Renderer(dist_coeffs=None, orig_size=self.cfg.render.res*self.cfg.render.render_up_scale,
                                                image_size=self.cfg.render.res*self.cfg.render.render_up_scale,
                                                light_intensity_ambient=0.8,
                                                light_intensity_directional=0.0,
                                                anti_aliasing=False, far=200)
        else:
            raise NotImplementedError("Renderer not implemented")
    
        self.mean_, self.std_ = np.load("data/ava/mean_std.npy")
        self.mean_            = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_             = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        self.mean_, self.std_ = torch.tensor(self.mean_), torch.tensor(self.std_)
        self.mean_, self.std_ = self.mean_.float(), self.std_.float()
        self.mean_, self.std_ = self.mean_.unsqueeze(0), self.std_.unsqueeze(0)
        
        self.font             = cv2.FONT_HERSHEY_SIMPLEX

        os.makedirs(self.cfg.storage_folder + "/slowfast/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)
        
        self.ava_valid_classes = joblib.load("data/ava/ava_class_mappping.pkl")
        self.ava_valid_classes_inv = {v: k for k, v in self.ava_valid_classes.items()}
        self.colors            = get_colors()
        
    def forward(self, tokens, mask_type):

        output, vq_loss       = self.encoder(tokens, mask_type)
        smpl_outputs          = self.decode(output, input=False)
    
        return output, smpl_outputs, vq_loss

    def decode(self, output, input=False, render=False):

        if(input):
            # return pesudo gt pose, betas and location
            smooth_embed = output.view(output.shape[0]*output.shape[1], -1)
            a_ = smooth_embed[:, :9]
            b_ = smooth_embed[:, 9:9+207]
            c_ = smooth_embed[:, 9+207:9+207+10]
            pred_cam   = smooth_embed[:, 226:229]
            pred_cam_t = pred_cam.view(1, pred_cam.shape[0], 3)
            pred_smpl_params = {
                'global_orient': a_.view(a_.shape[0], 1, 3, 3),
                'body_pose': b_.view(b_.shape[0], 23, 3, 3),
                'betas': c_.view(c_.shape[0], 10),
            }

            # if("gt_pose" in self.cfg.one_euro_filter): pred_smpl_params = smooth_pose(pred_smpl_params)

            smpl_output                       = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
            BS                                = smooth_embed.size(0)
            dtype                             = smooth_embed.dtype
            focal_length                      = self.cfg.smpl_cfg.EXTRA.FOCAL_LENGTH * torch.ones(BS, 2, dtype=dtype).to(output.device)

            pred_cam_t   = pred_cam_t.permute(1, 0, 2)
            smpl_outputs = {
                'pred_smpl_params' : [pred_smpl_params],
                'smpl_output'      : smpl_output,
                'cam'              : [pred_cam],
                'cam_t'            : pred_cam_t,
                'focal_length'     : focal_length,
            }
        else:
            
            # return predicted gt pose, betas and location
            class_token      = output[:, 0:self.cfg.max_people, :].contiguous()
            pose_tokens      = output[:, self.cfg.max_people:, :].contiguous()
            
            smooth_embed     = pose_tokens.view(pose_tokens.shape[0]*pose_tokens.shape[1], -1)
            pred_smpl_params = [self.smpl_head[i](smooth_embed[:, :])[0] for i in range(self.cfg.num_smpl_heads)]
            pred_cam         = [self.loca_head[i](smooth_embed[:, :]) for i in range(self.cfg.num_smpl_heads)]
            action_preds_ava = [self.action_head_ava[i](smooth_embed[:, :]) for i in range(self.cfg.num_smpl_heads)]
            # action_preds_ava2 = [self.action_head_ava[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)] # [bs, 125, 256]

            smpl_output      = self.smpl(**{k: v.float() for k,v in pred_smpl_params[0].items()}, pose2rot=False)
            BS               = smooth_embed.size(0)
            dtype            = smooth_embed.dtype
            focal_length     = self.cfg.smpl_cfg.EXTRA.FOCAL_LENGTH * torch.ones(BS, 2, dtype=dtype).to(output.device)

            pred_cam_t       = torch.stack(pred_cam, dim=0)
            pred_cam_t       = pred_cam_t.permute(1, 0, 2)
    
            action_preds_ava = torch.stack(action_preds_ava, dim=0)
            action_preds_ava = action_preds_ava.permute(1, 0, 2)
            
            # action_preds_ava2 = torch.stack(action_preds_ava2, dim=0) # [1, bs, 125, 256]
            # action_preds_ava2 = action_preds_ava2.permute(1, 2, 0, 3) # [bs, 125, 1, 256]
            
            action_preds_kinetics = self.action_head_kinetics(class_token)
            
            ##### reporject the keypoints #####
            pred_keypoints_2d_smpl_ = []
            for i in range(self.cfg.num_smpl_heads):
                if("kp_l1" in self.cfg.loss_type):
                    smpl_output_            = self.smpl(**{k: v.float() for k,v in pred_smpl_params[i].items()}, pose2rot=False)
                    pred_joints             = smpl_output_.joints
                    zeros_                  = torch.zeros((pred_joints.shape[0], 1, 3)).to(output.device)
                    pred_joints_            = torch.concat((pred_joints, zeros_), 1)
                    focal                   = self.cfg.smpl_cfg.EXTRA.FOCAL_LENGTH * torch.ones(pred_joints_.shape[0], 2, dtype=torch.float32)/self.cfg.render.res
                    rotation                = torch.eye(3,).unsqueeze(0).expand(pred_joints_.shape[0], -1, -1).float().to(output.device)
                    center                  = torch.zeros(pred_joints_.shape[0], 2).float().to(output.device)
                    pred_cam_t_x            = torch.cat((pred_cam_t[:, i, 0:1], pred_cam_t[:, i, 1:2], 200.0*pred_cam_t[:, i, 2:]), 1)
                    pred_cam_t_x            = pred_cam_t_x.detach()
                    pred_keypoints_2d_smpl_.append(perspective_projection(pred_joints_, rotation=rotation, translation=pred_cam_t_x[:, :], focal_length=focal, camera_center=center))
                    pred_keypoints_2d_smpl  = torch.stack(pred_keypoints_2d_smpl_, dim=0)
                    pred_keypoints_2d_smpl   = pred_keypoints_2d_smpl.permute(1, 0, 2, 3)
                    del smpl_output_, pred_joints, zeros_, pred_joints_, focal, rotation, center, pred_cam_t_x
                else:
                    pred_keypoints_2d_smpl  = None

                
            smpl_outputs = {
                'pred_smpl_params'      : pred_smpl_params,
                'smpl_output'           : smpl_output,
                'cam'                   : pred_cam,
                'cam_t'                 : pred_cam_t,
                'focal_length'          : focal_length,
                'pred_actions_ava'      : action_preds_ava,
                # 'pred_actions_ava2'      : action_preds_ava2,
                'pred_actions_kinetics' : action_preds_kinetics,
                'pred_keypoints'        : pred_keypoints_2d_smpl
            }
        
        return smpl_outputs

    
    def moving_window_smooth(self, input_data, mask_type="null"):

        fl              = self.cfg.frame_length*self.cfg.max_people
        s               = 0
        full_length     = input_data['pose_shape'].shape[1]

        if(full_length>fl):
            w_steps = range(s, s+full_length-fl+1)
            for w_ in w_steps:
                input_data_tmp = {}
                for k in input_data.keys(): 
                    input_data_tmp[k] = input_data[k][:, w_:w_+fl]

                out_, _, _ = self.forward(input_data_tmp, mask_type)
                if(w_==s):
                    output           = out_[:, :fl//2+1, :].clone()
                elif(w_==w_steps[-1]):
                    output           = torch.cat([output, out_[:, fl//2:, :]], 1)
                else:
                    output           = torch.cat([output, out_[:, fl//2:fl//2+1, :]], 1)
            smpl_stuff = self.decode(output, input=False)
        else:
            e = s+fl
            # input_data_tmp = {}
            # for k in input_data.keys(): 
            #     input_data_tmp[k] = input_data[k][:, s:e]
            with torch.no_grad():
                output, smpl_stuff, _ = self.forward(input_data, mask_type)

        return output, smpl_stuff, 0

    def store_results_batch(self, input_data, output_data, meta_data, smpl_output, video_name, save_path, output=None):
        frame_name_array, frame_size_array, frame_conf_array, frame_bbox_array = self.process_meta_data(meta_data)
        BS, T, P, _ = input_data['has_detection'].shape
        pred_action = smpl_output['pred_actions_ava'].view(BS, T, P, smpl_output['pred_actions_ava'].shape[-1])
        # pred_action2 = smpl_output['pred_actions_ava2'].view(BS, T, P, smpl_output['pred_actions_ava2'].shape[-1])

        for bid in range(len(video_name)):
            video_id     = video_name[bid].split("ava-val_")[1][:11]
            key_frame    = video_name[bid].split("ava-val_")[1].split(".jpg")[0][12:].split("_")[0]
            frame_id     = "%04d"%(int(key_frame)//30 + 900,)

            kfid         = None
            max_length   = 0
            for fid, frame_name_ in enumerate(frame_name_array[bid]):
                if(frame_name_=="-1"):
                    break
                else:
                    max_length+=1
                    if(key_frame in frame_name_): 
                        kfid = fid
            # import ipdb; ipdb.set_trace()
            if(kfid is not None):
                if(input_data['has_detection'][bid][kfid][0]):
                    size        = frame_size_array[bid][kfid] #meta_data['frame_size'][kfid]
                    conf        = frame_conf_array[bid][kfid] #meta_data['frame_conf'][kfid][0].item()
                    

                    if("gt" in self.cfg.test_type.split("|")[1]):
                        preds     = input_data['action_emb'][bid, kfid, 0, :].cpu().view(1, -1)
                        preds     = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)  
                        
                    elif("GT" in self.cfg.test_type.split("|")[1]):
                        if(self.cfg.ava.predict_valid):
                            preds     = output_data['action_label_ava'][bid, kfid, 0, 0, :self.cfg.ava.num_valid_action_classes].cpu().view(1, -1)
                        else:
                            preds     = output_data['action_label_ava'][bid, kfid, 0, 0, :].cpu().view(1, -1)
                        preds     = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)  
                        
                    elif("clip" in self.cfg.test_type.split("|")[1]):
                        preds     = input_data['clip_emb'][bid, kfid, 0, :].cpu().view(1, -1)
                        preds     = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)  

                    elif("avg" in self.cfg.test_type.split("|")[1]):
                        w_x       = int(self.cfg.test_type.split("avg.")[1].split(".")[0])
                        s_        = kfid - w_x if kfid-w_x>0 else 0
                        e_        = kfid + w_x if kfid+w_x<max_length else max_length
                        if(self.cfg.debug):
                            import ipdb; ipdb.set_trace()
                        pred      = torch.sigmoid(pred_action[bid, s_:e_, 0, :].cpu())
                        pred      = pred.mean(0, keepdim=True)
                        preds     = torch.cat([torch.zeros(pred.shape[0],1), pred], dim=1)

                    elif("max" in self.cfg.test_type.split("|")[1]):
                        w_x       = int(self.cfg.test_type.split("max.")[1].split(".")[0])
                        s_        = kfid - w_x if kfid-w_x>0 else 0
                        e_        = kfid + w_x if kfid+w_x<max_length else max_length
                        # import ipdb; ipdb.set_trace()
                        pred      = torch.sigmoid(pred_action[bid, s_:e_, 0, :].cpu())
                        pred      = pred.max(0, keepdim=True)[0]
                        preds     = torch.cat([torch.zeros(pred.shape[0],1), pred], dim=1)

                    else:
                        pred      = torch.sigmoid(pred_action[bid, kfid, 0, :].cpu().view(1, -1))
                        preds     = torch.cat([torch.zeros(pred.shape[0],1), pred], dim=1)

                    bbox      = frame_bbox_array[bid][kfid] #meta_data['frame_bbox'][kfid][0]
                    bbox_     = torch.cat((bbox[:2], bbox[:2] + bbox[2:]))
                    bbox_2    = np.array([i.item() for i in bbox_])
                    bbox_norm = [bbox_[0].item()/size[1].item(), bbox_[1].item()/size[0].item(), bbox_[2].item()/size[1].item(), bbox_[3].item()/size[0].item()]
                    bbox_norm = np.array(bbox_norm).reshape(1, 4)

                    result = [video_name[bid], np.array(preds), bbox_2.reshape(1, 4), np.array([size[0].item(), size[1].item()]).reshape(1, 2), conf.cpu()]
                    joblib.dump(result, save_path[bid])
                    
                    if(self.cfg.debug and self.cfg.store_svm_vectors):
                        out_ = output[bid, kfid]
                        svm_result = [video_name[bid], np.array(preds), out_.cpu().numpy(), bbox_2.reshape(1, 4), np.array([size[0].item(), size[1].item()]).reshape(1, 2), conf.cpu()]
                        try:    self.svm_pred.append(svm_result)
                        except: self.svm_pred = [svm_result]
                        
                    del result
        
    def write_ava_csv(self, slowfast_path, csv_path):
        AVA_VALID_FRAMES = range(902, 1799)
        log.info("Start reading predictions.")
        slowfast_files = [i for i in os.listdir(slowfast_path) if i.endswith(".pkl")]
        label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/ava/ava_action_list.pbtxt')
        f = open(csv_path, 'w')
        writer = csv.writer(f)
        counter = 0
        
        slowfast_pkl_files = joblib.Parallel(n_jobs=8, timeout=9999)(joblib.delayed(joblib.load)(slowfast_path + path) for path in slowfast_files)
        
        for sl_, slowfast_file in enumerate(slowfast_files):
            
            video_id  = slowfast_file.split("ava-val_")[1][:11]
            key_frame = slowfast_file.split("ava-val_")[1][12:18]
            frame_id  = "%04d"%(int(key_frame)//30 + 900,)
            # data      = joblib.load(slowfast_path + slowfast_file)
            data      = slowfast_pkl_files[sl_]
            if(int(key_frame)//30+900 not in AVA_VALID_FRAMES): continue
            
            h, w = data[-2][0][0], data[-2][0][1]
            det_conf_ = data[-1]
            if(det_conf_ < 0.80): continue
            for i in range(len(data[2])):
                x1, y1, x2, y2 = data[-3][i][0], data[-3][i][1], data[-3][i][2], data[-3][i][3]
                pred  = data[1][i]
                pred_ = np.argsort(pred)[::-1]
                conf  = pred[pred_]
                loc_  = conf>-1
                pred_ = pred_[loc_]
                conf  = conf[loc_]

                for j in range(len(pred_)):
                    if(len(pred_)==self.cfg.ava.num_valid_action_classes+1):
                        pred_class = self.ava_valid_classes[pred_[j]]
                    else:
                        pred_class = pred_[j]
                    if(pred_class!=0 and pred_class in allowed_class_ids):
                        result = [video_id, frame_id, x1/w, y1/h, x2/w, y2/h, pred_class, conf[j]]
                        writer.writerow(result)
                counter += 1
        log.info("number of bbox detected : " + str(counter))
        f.close()
        
        del slowfast_pkl_files


    def load_image(self, image_path, resolution=1080):

        cv_image                     = cv2.imread(image_path)
        img_height_, img_width_      = cv_image.shape[:2]
  
        ratio                        = resolution/img_width_
        cv_image                     = cv2.resize(cv_image, (resolution, int(img_height_*ratio)))
        img_height_, img_width_, _   = cv_image.shape
  
        new_image_size               = max(img_height_, img_width_)
        delta_w                      = new_image_size - img_width_
        delta_h                      = new_image_size - img_height_
        top_, bottom_, left_, right_ = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
        image_bordered               = cv2.copyMakeBorder(cv_image, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image_small                  = cv2.resize(image_bordered, (self.cfg.render.res*self.cfg.render.render_up_scale, self.cfg.render.res*self.cfg.render.render_up_scale))
        if("fullframe" in self.cfg.test_type):
            top, bottom, left, right = 0, 0, 0, 0
            img_height_              = new_image_size
            img_width_               = new_image_size
        else:
            top, bottom, left, right = top_, bottom_, left_, right_
        image_large                  = numpy_to_torch_image(np.array(image_bordered)/255.)[:, :, top:top+img_height_, left:left+img_width_]


        images_ = {
            'img_small': image_small, 
            'img_large': image_large,
            'img_path': image_path,
            'img_height': img_height_,
            'img_width': img_width_,
            'img_size_new': new_image_size,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
            'ratio' : ratio,
            'top_': top_,
            'bottom_': bottom_,
            'left_': left_,
            'right_': right_,
        }

        return images_

    def composite_image(self, rgb_from_pred, valid_mask, images_, background_rgb=True):
        
        rendered_image             = cv2.resize(rgb_from_pred, (images_['img_size_new'], images_['img_size_new']))
        rendered_image             = numpy_to_torch_image(np.array(rendered_image))[:, :, images_['top']:images_['top']+images_['img_height'], images_['left']:images_['left']+images_['img_width']]
        
        if(self.cfg.render.walker=="PL"):
            if(background_rgb):
                rendered_image         = copy.deepcopy(images_['img_large']) 
            else:
                rendered_image         = rendered_image*0.0
        else:
            if(background_rgb):
                valid_mask                 = np.repeat(valid_mask, 3, 2).astype(float)
                valid_mask                 = cv2.resize(valid_mask, (images_['img_size_new'], images_['img_size_new']))
                valid_mask                 = numpy_to_torch_image(np.array(valid_mask))[:, :, images_['top']:images_['top']+images_['img_height'], images_['left']:images_['left']+images_['img_width']]
                
                loc_b                      = valid_mask==1
                image_rgb_1                = copy.deepcopy(images_['img_large']) 
                image_rgb_1[loc_b]         = 0.0
                rendered_image[torch.logical_not(loc_b)] = 0.0
                rendered_image             = rendered_image + image_rgb_1

        return rendered_image

    def process_meta_data(self, meta_data):
        frame_name_array = np.array(meta_data['frame_name']).transpose(1, 0)
        frame_size_array = []
        frame_conf_array = []
        frame_bbox_array = []
        for i in range(len(meta_data['frame_size'])):
            frame_size_array.append(meta_data['frame_size'][i])
            frame_conf_array.append(meta_data['frame_conf'][i])
            frame_bbox_array.append(meta_data['frame_bbox'][i])
        frame_size_array  = torch.stack(frame_size_array).permute(1, 0, 2)
        frame_conf_array  = torch.stack(frame_conf_array).permute(1, 0)
        frame_bbox_array  = torch.stack(frame_bbox_array).permute(1, 0, 2)

        return frame_name_array, frame_size_array, frame_conf_array, frame_bbox_array
    
    def render_vert(self, pred_vertices_, pred_camera_bs_, color, image, texture=None, use_image=True, side_view=False):
        
        if(self.cfg.render.engine=="PYR"):
            img_rendered, valid_mask = self.py_render.visualize_all(pred_vertices_, pred_camera_bs_, color, image, use_image=use_image, side_view=side_view)
            
        elif(self.cfg.render.engine=="NMR"):
            # initialize camera params and mesh faces for NMR
            batch_size = pred_vertices_.shape[0]
            K = torch.eye(3, device=self.device)
            K[0, 0] = K[1, 1]  = self.cfg.smpl_cfg.EXTRA.FOCAL_LENGTH
            K[2, 2] = 1
            K[1, 2] = K[0, 2]  = self.cfg.render.res*self.cfg.render.render_up_scale/2.0
                                                             
            K = K.unsqueeze(0).repeat(batch_size, 1, 1).float()  # to BS
            R = torch.eye(3, device=self.device).unsqueeze(0).float()
            t = torch.zeros(3, device=self.device).unsqueeze(0).float()
            face_tensor = torch.tensor(self.faces_cpu.astype(np.int64), dtype=torch.long, device=self.device).unsqueeze_(0)
            face_tensor = face_tensor.repeat(batch_size, 1, 1)
            
            verts = torch.from_numpy(pred_vertices_ + pred_camera_bs_).to(self.device).float()

            if(texture is None):
                texture = torch.ones(batch_size, 3, 256, 256).to(self.device)*0.5
            texture_vert = torch.nn.functional.grid_sample(texture, self.uv_sampler.repeat(batch_size,1,1,1).to(self.device))
            texture_vert = texture_vert.view(texture_vert.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1).contiguous()
            texture      = texture_vert.unsqueeze(4).expand(-1, -1, -1, -1, 6, -1) 

            img_rendered, depth, valid_mask = self.nmr_renderer(verts, face_tensor, textures=texture, K=K, R=R, t=t, dist_coeffs=torch.tensor([[0., 0., 0., 0., 0.]], device=self.device))
            img_rendered = np.array(img_rendered[0].permute(1,2,0).cpu().detach().numpy())
            valid_mask   = np.array(valid_mask.permute(1,2,0).cpu().detach().numpy())
            
        else:
            raise ValueError("render engine not supported")
        
        return img_rendered, valid_mask


    @torch.no_grad()
    def render_with_smpl(self, input_data, output_data, meta_data, smpl_output, video_names, video_path):
        
        frame_name_array, frame_size_array, frame_conf_array, frame_bbox_array = self.process_meta_data(meta_data)

        
        # pred_smpl                    = copy.deepcopy(smpl_output['smpl_output'])
        pred_smpl                    = smpl_output['smpl_output']
        pred_vertices_all            = pred_smpl.vertices
        pred_joints_all              = pred_smpl.joints
        # pred_camera_all              = copy.deepcopy(smpl_output['cam_t'])[:, 0, :]
        pred_camera_all              = smpl_output['cam_t'][:, 0, :]
        pred_action_all              = smpl_output['pred_actions_ava'][:, 0, :]

        a1 = input_data['pose_shape'].shape[0]
        a2 = input_data['pose_shape'].shape[1]
        a3 = input_data['pose_shape'].shape[2]
        # print(a1, a2, a3, pred_vertices_all.shape)
        
        pred_vertices_all = pred_vertices_all.view(a1, a2, a3, pred_vertices_all.shape[-2], pred_vertices_all.shape[-1])
        pred_joints_all   = pred_joints_all.view(a1, a2, a3, pred_joints_all.shape[-2], pred_joints_all.shape[-1])
        pred_camera_all   = pred_camera_all.view(a1, a2, a3, pred_camera_all.shape[-1])
        pred_action_all   = pred_action_all.view(a1, a2, a3, pred_action_all.shape[-1])

        # pred_vertices_all = pred_vertices_all[:, :, :, :, :]
        # pred_joints_all = pred_joints_all[:, :, :, :, :]
        # pred_camera_all = pred_camera_all[:, :, :, :]
        # pred_action_all = pred_action_all[:, :, :, :]
        if(self.cfg.render.walker=="PL"):
            # pred_keypoints_all = copy.deepcopy(smpl_output['pred_keypoints'][:, 0, :, :])
            pred_keypoints_all = smpl_output['pred_keypoints'][:, 0, :, :]
            pred_keypoints_all = pred_keypoints_all.view(a1, a2, a3, pred_keypoints_all.shape[1], pred_keypoints_all.shape[2])
            pred_keypoints_all = pred_keypoints_all[:, :, 0, :, :]
            # vitpose_keypoints_all = copy.deepcopy(output_data['vitpose'][:, :, 0, 0, :, :]) 
            vitpose_keypoints_all = output_data['vitpose'][:, :, 0, 0, :, :]
            
        if("T" in self.cfg.render.walker):
            input_texture_emb_all = input_data['appe_emb']
        
        for vid, video_name in enumerate(video_names):
            if(self.rendered_videos>=self.cfg.render.num_videos): break     
            self.rendered_videos += 1
            log.info("Rendering video {} of {}".format(self.rendered_videos, self.cfg.render.num_videos))

            input_pose_shape_tokens  = input_data['pose_shape'][vid]

            if(self.cfg.use_mean_std): input_pose_shape_tokens = input_pose_shape_tokens*self.std_.to(input_pose_shape_tokens.device) + self.mean_.to(input_pose_shape_tokens.device)
            with torch.no_grad():      input_pose_shape        = self.decode(input_pose_shape_tokens, input=True)
            
            # input_smpl               = copy.deepcopy(input_pose_shape['smpl_output'])
            input_smpl               = input_pose_shape['smpl_output']
            input_vertices           = input_smpl.vertices.detach().view(a2, a3, input_smpl.vertices.shape[-2], input_smpl.vertices.shape[-1])
            # input_camera             = copy.deepcopy(input_pose_shape['cam_t'])[:, 0, :]
            # if(self.cfg.debug): import ipdb; ipdb.set_trace()
            input_camera             = input_pose_shape['cam_t'][:, 0, :].view(a2, a3, input_pose_shape['cam_t'].shape[-1])

            if(self.cfg.render.vis_pred_loca):
                # pred_camera          = copy.deepcopy(pred_camera_all[vid].detach())
                pred_camera          = pred_camera_all[vid].detach()
            else:
                # pred_camera          = copy.deepcopy(input_camera.detach())
                pred_camera          = input_camera.detach()
            # pred_vertices            = copy.deepcopy(pred_vertices_all[vid])
            pred_vertices            = pred_vertices_all[vid]
            # pred_action              = copy.deepcopy(pred_action_all[vid])
            pred_action              = pred_action_all[vid]
            if(self.cfg.render.walker=="PL"):
                # pred_keypoints       = copy.deepcopy(pred_keypoints_all[vid])
                pred_keypoints       = pred_keypoints_all[vid]
                # vit_pose_keypoints   = copy.deepcopy(vitpose_keypoints_all[vid])
                vit_pose_keypoints   = vitpose_keypoints_all[vid]

            if("pred_loca" in self.cfg.one_euro_filter): pred_camera      = smooth_tran(pred_camera.view(-1,3)).view(pred_camera.shape)
            if("gt_loca"   in self.cfg.one_euro_filter): input_camera     = smooth_tran(input_camera.view(-1,3)).view(input_camera.shape)

            pred_camera_bs               = pred_camera.unsqueeze(2).repeat(1, 1, pred_vertices.size(2), 1) #pred_camera.unsqueeze(1).repeat(1, pred_vertices.size(1), 1)
            pred_camera_bs[:, :, :, 2]   = pred_camera_bs[:, :, :, 2]/self.cfg.render.render_up_scale*200

            input_camera_bs              = input_camera.unsqueeze(2).repeat(1, 1, pred_vertices.size(2), 1)
            input_camera_bs[:, :, :, 2]  = input_camera_bs[:, :, :, 2]/self.cfg.render.render_up_scale*200
            
            for fid in range(min(len(frame_name_array[vid]), 1000)):
                
                if(frame_name_array[vid][fid]=="-1"): break
                image_ = self.load_image(frame_name_array[vid][fid], resolution=1080)
                # color  = np.array([[125, 125, 125]]).repeat(self.cfg.max_people, 0)/255.0
                color  = np.array([[125, 125, 125]])/255.0
                color  = color.repeat(self.cfg.max_people,0)
                color[0] = copy.deepcopy(self.colors[self.rendered_videos%20])/255.0
                
                if(fid==0):
                    # frame_size             = (int(image_['img_width']*2), int(image_['img_height']*2)) if("sideview" in self.cfg.test_type) else (int(image_['img_width']), int(image_['img_height']*2))
                    frame_size             = (int(image_['img_width']*2), int(image_['img_height']*2)) if("sideview" in self.cfg.test_type) else (int(image_['img_width']*2), int(image_['img_height']))
                    video_file             = cv2.VideoWriter(video_path[vid], cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=frame_size)
                # import pdb; pdb.set_trace()
                has_detection_  = input_data['has_detection'][vid][fid:fid+1, :, 0]==1
                mask_detection_ = input_data['mask_detection'][vid][fid:fid+1, :, 0]==1
                loc_1  = torch.logical_or(has_detection_, mask_detection_).cpu().numpy()[0]
                loc_0  = torch.logical_or(torch.logical_not(has_detection_), mask_detection_)
                # if(self.cfg.max_people>1):
                #     color[loc_.cpu()] /= 4.0
                # else:
                #     if(loc_.cpu()): color /= 4.0
                
                pred_vertices_      = pred_vertices[fid][:, :, :].cpu().numpy()                
                pred_camera_bs_     = pred_camera_bs[fid][:, :, :].cpu().numpy()
                input_vertices_     = input_vertices[fid][:, :, :].cpu().numpy()
                input_camera_bs_    = input_camera_bs[fid][:, :, :].cpu().numpy()
                if(self.cfg.render.walker=="PL"):
                    pred_keypoint_  = pred_keypoints[fid:fid+1][:, :, :].cpu().numpy()
                    vit_pose_keypoint_  = vit_pose_keypoints[fid:fid+1][:, :, :].cpu().numpy()

                if("T" in self.cfg.render.walker):
                    input_texture   = input_texture_emb_all[vid, fid, 0, :]
                    texture_maps    = self.encoding_head(input_texture.view(1, 16, 16, 16).float(), en=False)
                    texture_maps    = texture_maps[:, :3, :, :]*5.0
                    texture_maps    = texture_maps * torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1,3,1,1)
                    texture_maps    = texture_maps + torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1,3,1,1)
                    texture_maps    = texture_maps.clamp_(0, 1)
                    # ndarr           = make_grid(texture_maps[:1], nrow=1)
                    # ndarr           = ndarr.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    # cv_ndarr        = cv2.resize(ndarr, (256, 256))
                    # cv2.imwrite("0.png", cv_ndarr)
                else:
                    texture_maps    = None

                ### render predicted meshes
                # img_rendered, valid_mask  = self.render_vert(pred_vertices_[loc_1], pred_camera_bs_[loc_1], color[loc_1], 0*image_['img_small']/255.0, texture=texture_maps, use_image=True, side_view=False) 
                # img_rendered = self.composite_image(img_rendered, valid_mask, image_, background_rgb="B" in self.cfg.render.walker)
                img_rendered = copy.deepcopy(image_['img_large'])
                if("sideview" in self.cfg.test_type):
                    img_rendered_rot, valid_mask_rot  = self.render_vert(pred_vertices_, pred_camera_bs_, color, 0*image_['img_small']/255.0, use_image=True, side_view=True)
                    img_rendered_rot = self.composite_image(img_rendered_rot, valid_mask_rot, image_, background_rgb=False)
                
                
                ### render input meshes
                img_rendered_init, valid_mask_init  = self.render_vert(input_vertices_[loc_1], input_camera_bs_[loc_1], color[loc_1], 0*image_['img_small']/255.0, texture=texture_maps, use_image=True, side_view=False)
                img_rendered_init = self.composite_image(img_rendered_init, valid_mask_init, image_, background_rgb=True)
                if("sideview" in self.cfg.test_type):
                    img_rendered_init_rot, valid_mask_init_rot  = self.render_vert(input_vertices_, input_camera_bs_, color, 0*image_['img_small']/255.0, use_image=True, side_view=True)
                    img_rendered_init_rot = self.composite_image(img_rendered_init_rot, valid_mask_init_rot, image_, background_rgb=False)
                
                
                if("sideview" in self.cfg.test_type):
                    grid_img = make_grid(torch.cat((img_rendered, img_rendered_rot, img_rendered_init, img_rendered_init_rot), dim=0), nrow=2)
                else:
                    grid_img = make_grid(torch.cat([img_rendered, img_rendered_init], 0), nrow=2)

                grid_img = grid_img[[2,1,0], :, :]
                ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                cv_ndarr = cv2.resize(ndarr, frame_size)
                draw_text(cv_ndarr, str(fid), (40,40), color=(255, 255, 255), fontScale=1, thickness=1, fontFace=self.font, outline_color=(0, 0, 0), line_spacing=1.5)

                # bbox            = fbboxes[fid][0].int().numpy()
                # bbox_           = bbox*image_['ratio']
                # label           = faction[fid][0][0] + "\n" + faction[fid][1][0]  + "\n" + faction[fid][2][0]
                pred_label      = torch.sigmoid(pred_action[fid][0])
                _, order        = torch.topk(pred_label, k=10)

                label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/ava_action_list.pbtxt')
                top_labels = []
                top_probs  = []
                for i in order: 
                    class_map_id = i.item()+1 if not(self.cfg.ava.predict_valid) else self.ava_valid_classes[i.item()+1]
                    label_ = label_map.get(class_map_id, "n/a")
                    if(self.cfg.render.vis_action_label=="person"):
                        if(("an object" not in label_) and ("a person" not in label_)):
                            top_labels.append(label_)
                            top_probs.append(pred_label[i].item())
                    else:
                        top_labels.append(label_)
                        top_probs.append(pred_label[i].item())
                        
                pred_label_text = ""
                for i in range(min(5, len(top_labels))):
                    pred_label_text += top_labels[i] + " " + str(np.round(top_probs[i],2)) + "\n"
                
                
                ava_label = output_data['action_label_ava'][vid][fid][0, 0]
                _, order  = torch.topk(ava_label, k=10)
                
                top_labels = []
                top_probs  = []
                for i in order: 
                    class_map_id = i.item()+1 if not(self.cfg.ava.predict_valid) else self.ava_valid_classes[i.item()+1]
                    label_ = label_map.get(class_map_id, "n/a")
                    if(self.cfg.render.vis_action_label=="person"):
                        if(("an object" not in label_) and ("a person" not in label_)):
                            top_labels.append(label_)
                            top_probs.append(ava_label[i].item())
                    else:
                        top_labels.append(label_)
                        top_probs.append(ava_label[i].item())
                        
                gt_label_text = ""
                for i in range(min(5, len(top_labels))):
                    gt_label_text += top_labels[i] + " " + str(np.round(top_probs[i],2)) + "\n"
                

                draw_text(cv_ndarr, gt_label_text,   (100,10), color=(0, 255, 0), fontScale=0.3, thickness=1, fontFace=self.font, outline_color=(0, 50, 0), line_spacing=1.5)
                draw_text(cv_ndarr, pred_label_text, (500,10), color=(0, 0, 255), fontScale=0.3, thickness=1, fontFace=self.font, outline_color=(0, 50, 0), line_spacing=1.5)
                
                if(self.cfg.render.walker=="PL"):
                    # kp = (vit_pose_keypoint_[0, :, :2]+0.5)*image_['img_size_new']
                    kp_pred = (pred_keypoint_[0, :, :2]+0.5)*image_['img_size_new']
                    kp_vit  = (vit_pose_keypoint_[0, :, :2]+0.5)*image_['img_size_new']
                    if("fullframe" not in self.cfg.test_type):
                        kp_pred[:, 0] -= image_['left_']
                        kp_pred[:, 1] -= image_['top_']
                        kp_vit[:, 0]  -= image_['left_']
                        kp_vit[:, 1]  -= image_['top_']
                    for ii in range(25): 
                        if(ii in [8, 15, 16, 17, 18, 22, 23, 24, 19, 20, 21]): continue
                        cv2.circle(cv_ndarr, (int(kp_pred[ii][0]), int(kp_pred[ii][1])), color=[255, 255, 255], radius=3, thickness=3)
                        # cv2.circle(cv_ndarr, (int(kp_vit[ii][0]), int(kp_vit[ii][1])), color=[0, 255, 0], radius=3, thickness=3)
                        # cv2.putText(cv_ndarr, str(ii), (int(kp[ii][0]), int(kp[ii][1])), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    # cv2.imwrite(str(0) + ".png", cv_ndarr_)
                video_file.write(cv_ndarr)
            
            video_file.release()   
            
        return 0

    def on_train_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): self.trainer.datamodule.data_train.__getitem__(0)
        pass
    
    def step(self, batch: Any):
        input_data, output_data, _, _ = batch
        output, smpl_output, _        = self.forward(input_data, mask_type=self.cfg.mask_type)
        loss_dict                     = compute_loss(self.cfg, output, smpl_output, output_data, input_data, train=True)
        
        return loss_dict, output, smpl_output

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict, output, smpl_output = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()])
        if("img" in batch[0].keys()): batch[0].pop('img')
        if(self.cfg.compute_acc and "kinetics" in self.cfg.action_space):
            # for kinetics
            pred_actions_kinetics = smpl_output['pred_actions_kinetics'][:, 0, :]
            gt_actions_kinetics   = batch[1]['action_label_kinetics'][:, 0, 0, 0, 0].long()
            self.train_acc_1(pred_actions_kinetics, gt_actions_kinetics)
            self.train_acc_5(pred_actions_kinetics, gt_actions_kinetics)            
            self.log("train/acc/top_1", self.train_acc_1, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc/top_5", self.train_acc_5, on_step=False, on_epoch=True, prog_bar=True)
            
        self.train_loss(loss.item())
        for key in loss_dict.keys():
            self.log("train/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True)

        del loss_dict, output, smpl_output, batch
        
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")
        
    def on_validation_start(self):
        torch.cuda.empty_cache()
        self.rendered_videos = 0
        if(self.cfg.debug): self.trainer.datamodule.data_val.__getitem__(0)

    def validation_step(self, batch: Any, batch_idx: int):
        input_data, output_data, meta_data, video_name = batch
        if(self.cfg.render.enable): input_data_render = copy.deepcopy(batch[0])
        slowfast_paths = [self.cfg.storage_folder + "/slowfast/" + "".join(video_name[i].split(".jpg"))+".pkl" for i in range(len(video_name))]
        video_paths    = [self.cfg.storage_folder + "/videos/" + ("%04d")%(self.current_epoch) + "".join(video_name[i].split(".jpg"))+".mp4" for i in range(len(video_name))]
        
        output, smpl_output, _ = self.moving_window_smooth(input_data, mask_type=self.cfg.mask_type_test)
        if("img" in input_data.keys()): input_data.pop("img")
        loss_dict = compute_loss(self.cfg, output, smpl_output, output_data, input_data, train=False)
        loss = sum([v for k,v in loss_dict.items()])
        if(self.cfg.render.enable): input_data_render['mask_detection'] = copy.deepcopy(input_data['mask_detection'])
        
        if(self.cfg.compute_acc and "kinetics" in self.cfg.action_space):
            # for kinetics
            pred_actions_kinetics = smpl_output['pred_actions_kinetics'][:, 0, :]
            gt_actions_kinetics   = batch[1]['action_label_kinetics'][:, 0, 0, 0, 0].long()
            self.val_acc_1(pred_actions_kinetics, gt_actions_kinetics)
            self.val_acc_5(pred_actions_kinetics, gt_actions_kinetics)            
            self.log("val/acc/top_1", self.val_acc_1, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc/top_5", self.val_acc_5, on_step=False, on_epoch=True, prog_bar=True)
        
        
        if(self.cfg.compute_map and "ava" in self.cfg.action_space): 
            self.store_results_batch(input_data, output_data, meta_data, smpl_output, video_name, slowfast_paths, output=copy.deepcopy(output[:, self.cfg.max_people:, :]))

        # for post-hoc analysis
        if(self.cfg.debug and self.cfg.store_svm_vectors):

            gt_label = copy.deepcopy(output_data['action_label_ava'][:, :, 0, 0, :60])
            has_gt   = copy.deepcopy(output_data['has_gt'][:, :, 0, 0, :])
            pred_out = copy.deepcopy(output[:, self.cfg.max_people:, :])
            
            for ix in range(pred_out.shape[0]):
                gt_ = has_gt[ix, :, 0]==2
                gt_idx = gt_.nonzero()
                if(gt_idx.shape[0]> 0 ):
                    mid_idx = gt_idx[gt_idx.shape[0]//2]
                    pred_vect = pred_out[ix, mid_idx[0], :].detach().cpu().numpy()
                    gt_label_ = gt_label[ix, mid_idx[0], :].detach().cpu().numpy().astype(np.int)
                    try:     self.svm_vectors.append([pred_vect, gt_label_])
                    except:  self.svm_vectors = [pred_vect, gt_label_]
                        
        # For visualization
        if(self.cfg.render.enable): 
            self.render_with_smpl(input_data_render, output_data, meta_data, smpl_output, video_name, video_path = video_paths)

        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        del output, smpl_output, loss_dict, input_data, output_data, meta_data, video_name, batch
        if(self.cfg.render.enable): del input_data_render
            
        return {"loss": loss}

    @rank_zero_only
    def validation_epoch_end(self, outputs: List[Any]):
        
        if(self.cfg.debug and self.cfg.store_svm_vectors):
            print("svm vectors saved", self.cfg.storage_folder + "/svm_vectors.npy")
            try:    np.save(self.cfg.storage_folder + "/svm_vectors.npy", self.svm_vectors)
            except: pass
            
            try:    np.save(self.cfg.storage_folder + "/svm_pred_vectors.npy", self.svm_pred)
            except: pass
                
        # logic to read the results and compute the metrics
        if(self.cfg.test_batch_id<0 and self.cfg.compute_map and "ava" in self.cfg.action_space):
            self.write_ava_csv(self.cfg.storage_folder + "/slowfast/", self.cfg.storage_folder + "/ava_val.csv")
            a1 = open("/datasets01/AVA/080720/frame_list/ava_action_list_v2.2_for_activitynet_2019.pbtxt", "r")
            a2 = open("/datasets01/AVA/080720/frame_list/ava_val_v2.2.csv", "r")
            a3 = open( self.cfg.storage_folder + "/ava_val.csv", "r")
            aaaa = run_evaluation(a1, a2, a3)
            self.log("mAP : ", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
            log.info("mAP : " + str(aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
            print("mAP : " + str(aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
            self.log("val/mAP", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0, prog_bar=True)
            joblib.dump(aaaa, self.cfg.storage_folder + "/results/" + str(self.current_epoch) + ".pkl")
            
            if(self.best_val_acc < aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0):
                os.system("cp " + self.cfg.storage_folder + "/ava_val.csv" + " " + self.cfg.storage_folder + "/ava_val_best.csv")
                self.best_val_acc = aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0
        else:
            log.info("mAP : " + str(0))
            self.log("val/mAP", 0, prog_bar=True)
            
    def get_param_groups(self):
        def _get_layer_decay(name):
            layer_id = None
            if name in ("encoder.class_token", "encoder.pose_token", "encoder.mask_token"):
                layer_id = 0
            elif ("_encoder" in name):
                layer_id = 0
            elif ("_head" in name):
                layer_id = self.cfg.vit.depth + 1
            elif name.startswith("encoder.pos_embedding"):
                layer_id = 0
            elif name.startswith("encoder.transformer1.layers"):
                layer_id = int(name.split("encoder.transformer1.layers.")[1].split(".")[0]) + 1
            else:
                layer_id = self.cfg.vit.depth + 1
            layer_decay = self.cfg.layer_decay ** (self.cfg.vit.depth + 1 - layer_id)
            return layer_id, layer_decay

        # for m in self.modules():
        #     assert not isinstance(m, torch.nn.modules.batchnorm._NormBase), "BN is not supported with layer decay"

        non_bn_parameters_count = 0
        zero_parameters_count = 0
        no_grad_parameters_count = 0
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, p in self.named_parameters():
            if not p.requires_grad:
                group_name = "no_grad"
                no_grad_parameters_count += 1
                continue
            name = name[len("module."):] if name.startswith("module.") else name
            if ((len(p.shape) == 1 or name.endswith(".bias")) and self.cfg.ZERO_WD_1D_PARAM):
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "zero")
                weight_decay = 0.0
                zero_parameters_count += 1
            else:
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "non_bn")
                weight_decay = self.cfg.weight_decay
                non_bn_parameters_count += 1

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.lr * layer_decay,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.lr * layer_decay,
                }
            parameter_group_names[group_name]["params"].append(name)
            parameter_group_vars[group_name]["params"].append(p)

        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if(self.cfg.layer_decay is not None):
            optim_params = self.get_param_groups()
        else:
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.cfg.lr}]
        
        
        if(self.cfg.solver=="AdamW"):
            optimizer = optim.AdamW(params=optim_params, weight_decay=self.cfg.weight_decay, betas=(0.9, 0.95))
        elif(self.cfg.solver=="SGD"):
            optimizer = optim.SGD(params=optim_params, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        else:
            raise NotImplementedError("Unknown solver : " + self.cfg.solver)

        def warm_start_and_cosine_annealing(epoch):
            if epoch < self.cfg.warmup_epochs:
                lr = (epoch+1) / self.cfg.warmup_epochs
            else:
                lr = 0.5 * (1. + math.cos(math.pi * ((epoch+1) - self.cfg.warmup_epochs) / (self.trainer.max_epochs - self.cfg.warmup_epochs )))
            return lr

        if(self.cfg.scheduler == "cosine"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))], verbose=False)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.decay_steps, gamma=self.cfg.decay_gamma, verbose=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                'frequency': 1,
            }
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
