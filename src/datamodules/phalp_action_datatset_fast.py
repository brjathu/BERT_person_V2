import ast
import copy
import os
import pickle as pkl
import socket
import time
import traceback
from builtins import breakpoint
from functools import lru_cache

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

# import utils.rotation_conversions as geometry
from src.utils.utils import task_divider

MAX_CACHE_SIZE = 1000
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

def get_ego_rot(pose_shape, end_frame, start_frame):
    ego_location = pose_shape.copy()[:, :, 226:229].reshape(-1, 3)
    ego_rotation = pose_shape.copy()[:, :, :9].reshape(-1, 3, 3)
    M_ego = np.zeros((end_frame-start_frame, 4, 4))
    M_ego[:, 0, 0] = M_ego[:, 1, 1] = M_ego[:, 2, 2] = M_ego[:, 3, 3] = 1
    M_ego[:, :3, :3] = ego_rotation
    M_ego[:, :3, 3] = ego_location
    
    loca_ = np.sum(ego_rotation, (1,2))==0
    M_ego[loca_, 0, 0] = M_ego[loca_, 1, 1] = M_ego[loca_, 2, 2] = M_ego[loca_, 3, 3] = 1
    # import pdb; pdb.set_trace()
    return M_ego
        
class PHALP_action_dataset_fast(Dataset):
    def __init__(self, opt, train=True):
        
        self.opt              = opt
        self.data             = []
        self.track2video      = []
        self.frame_length     = self.opt.frame_length
        self.max_tokens       = self.opt.max_people
        self.train            = train
        self.mean_, self.std_ = np.load("data/mean_std.npy")
        self.mean_pose_shape  = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_pose_shape   = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        
        
        # ##### FOR Kinetics #####
        # if(self.opt.action_space=="kinetics400"):
        #     self.kinetics_class_list     = np.load("data/kinetics_400_classes.npy")

        # temp arguments
        self.opt.img_size     = 256
        self.pixel_mean_      = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape(3,1,1)
        self.pixel_std_       = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape(3,1,1)

        self.dataset_roots        = {
            "ava_val_1"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.307_ava_val/results_slowfast_v3/",         1],
            "ava_val_2"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_val/results_slowfast_v5/",         1],
            "ava_val_3"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v5/",         1],
            "ava_rend_"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6///",      1],            
            "ava_rend_climb"   : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v7////",      1],            
            "kinetics_train"   : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v6/",  self.opt.kinetics.sampling_factor],
            "kinetics_val"     : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results_slowfast_v6/",    1],
            "ava_train"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results_slowfast_v6/",       self.opt.ava.sampling_factor],
            "ava_val"          : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6/",         1],
            "ava_train_4"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results_slowfast_v7/",       self.opt.ava.sampling_factor],
            "ava_val_4"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v7/",         1],
            "kinetics_train_4" : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v7_2/",  self.opt.kinetics.sampling_factor],
            "ava_train_5"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.403_ava_train/results_slowfast_v12_3/",       self.opt.ava.sampling_factor],
            "ava_val_5"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.403_ava_val/results_slowfast_v11_3/",         1],
                  
                  
            # 1Hz
            "kinetics_train_7" : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v21_1/",  self.opt.kinetics.sampling_factor],
            "ava_train_7"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v21_1/",       self.opt.ava.sampling_factor],
            "avaK_train_7"     : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21_1/",      self.opt.ava.sampling_factor],
            "ava_val_8"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v21_1/",         1],            
            
            # 5 Hz
            "kinetics_train_9" : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v22_1/",  self.opt.kinetics.sampling_factor],
            "ava_train_9"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v22_2/",       self.opt.ava.sampling_factor],
            "ava_val_9"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v22_1/",         1],            
            
            # MAE
            "kinetics_train_11" : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v23_1/",  self.opt.kinetics.sampling_factor],
            "ava_train_11"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23_1/",       self.opt.ava.sampling_factor],
            "ava_val_11"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v23_2/",         1],            
            
            # 15 Hz
            "ava_val_10"       : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v23_1/",         1],            
            
            "ava_train_hug"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v22_2//",       self.opt.ava.sampling_factor],
            "ava_val_hug"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v22_1//",         1],            
            
            "ava_val_render100": ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v22_1//",         1],            
            "ava_val_render_good": ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v22_1///",         1],            
            
            "youtube": ["",             "/checkpoint/jathushan/TENET/out/Videos_v4.404_youtube/results_slowfast_v22_1/",         1],            
            
            
        }
        
        if(self.train):
            
            self.list_of_datasets = opt.train_dataset.split(",")
            for dataset in self.list_of_datasets:
                print(self.dataset_roots[dataset][0])
                self.get_dataset(root_dir=self.dataset_roots[dataset][1], 
                                filter_seq=self.dataset_roots[dataset][0],
                                num_sample=self.dataset_roots[dataset][2], 
                                min_track_length=1, 
                                total_num_tracks=None)        

        else:
            self.list_of_datasets = opt.test_dataset.split(",")
            for dataset in self.list_of_datasets:
                self.get_dataset(root_dir=self.dataset_roots[dataset][1], 
                                filter_seq=self.opt.test_class, 
                                min_track_length=1, 
                                total_num_tracks=None)
            
            self.data = np.array(self.data)
            self.track2video = np.array(self.track2video)
            # idx = np.argsort(self.data)
            # self.data = self.data[idx]
            # self.track2video = self.track2video[idx]
            
            self.data = task_divider(self.data, self.opt.test_batch_id, self.opt.number_of_processes)
            self.track2video = task_divider(self.track2video, self.opt.test_batch_id, self.opt.number_of_processes)

        self.ava_valid_classes = np.load("data/ava_valid_classes.npy")
        self.a1                = np.load("data/ava_clases_bad1.npy")
        self.kinetics_annotations = joblib.load("data/kinetics_annot_train.pkl")
        print("Number of tracks: ", len(self.data))
        
        if(self.opt.use_optimized_pose):
            self.pose_key = "opt_pose_shape"
        else:
            self.pose_key = "pose_shape"
            
        # if(self.opt.load_other_tracks or not(self.train)):
        #     self.ego_id = 0
        # else:
        #     self.ego_id = np.random.randint(0, self.max_tokens)
        self.ego_id = 0


    def __len__(self):
        return len(self.data)

    def get_dataset(self, root_dir="", filter_seq="posetrack-train", num_sample=1, min_track_length=20, total_num_tracks=None):
        count    = 0
        count_f  = 0
        path_npy = "data/"+"fast_".join(root_dir.split("/"))+".npy"
        
        # to store all the files in a list
        if(os.path.exists(path_npy)):
            list_of_files = np.load(path_npy)
        else:
            list_of_files = os.listdir(root_dir)
            np.save(path_npy, list_of_files)
            
        # to store all the tracks from same video in a dictionary
        list_of_t2v = {}
        list_of_v2t = {}
        for file_i in list_of_files:
            video_name = "_".join(file_i.split("_")[:-2])
            list_of_v2t.setdefault(video_name, []).append(file_i)
        for video_name in list_of_v2t.keys():
            all_tracks = list_of_v2t[video_name]
            for track_i in all_tracks:
                list_of_t2v[track_i] = [os.path.join(root_dir, i) for i in all_tracks if i!=track_i]
        if(self.opt.debug):
            import ipdb; ipdb.set_trace()
            
        for i_, video_ in enumerate(list_of_files):
            if(video_.endswith(".pkl") and filter_seq in video_):
                if(int(video_.split("_")[-1][:-4])>min_track_length):
                    for _ in range(num_sample): 
                        self.data.append(os.path.join(root_dir, video_))
                        self.track2video.append(list_of_t2v[video_])
                    count += 1
                    count_f += int(video_.split("_")[-1][:-4])
                    if(total_num_tracks is not None):
                        if(count>=total_num_tracks):
                            break
        print("Total number of tracks: ", count)
        print("Total number of frames: ", count_f)
    
    def get_start_end_frame(self, list_of_frames, f_):

        if(self.train):
            start_frame  = np.random.choice(len(list_of_frames)-(self.frame_length+f_), 1)[0] if(len(list_of_frames)>self.frame_length+f_) else 0
            end_frame    = start_frame + self.frame_length if(len(list_of_frames)>self.frame_length+f_) else  len(list_of_frames)-f_
            key_frame    = (start_frame+end_frame)//2
        else:
            start_frame  = 0
            end_frame    = len(list_of_frames) if(self.opt.full_seq_render) else min(len(list_of_frames), self.frame_length)
            key_frame    = (start_frame+end_frame)//2

        return start_frame, end_frame, key_frame

    def read_from_phalp_fast(self, idx):
        try:
            detection_data   = joblib.load(self.data[idx])
        except:
            np.save("_TMP/bad_files/" + self.data[idx].split("/")[-1].split(".")[0] + ".npy", [self.data[idx]])
            detection_data   = joblib.load(self.data[0])
            
        list_of_frames   = list(range(len(detection_data["frame_name"])))
        if(self.opt.frame_rate_range>1 and self.train):
            frame_rate     = np.random.randint(1, self.opt.frame_rate_range)
            list_of_frames = list_of_frames[::frame_rate]
        return detection_data, list_of_frames, self.data[idx].split("/")[-1][:-4]
    
    def read_from_phalp_other(self, other_track):
        try:
            detection_data   = joblib.load(other_track)
        except:
            detection_data   = joblib.load(self.data[0])
        return detection_data

    def initiate_dict(self, frame_length, f_):
        
        input_data = { 
            'pose_shape'            : np.zeros((frame_length, self.max_tokens, 229))*0.0,
            'relative_pose'         : np.zeros((frame_length, self.max_tokens, 16))*0.0,
            'has_detection'         : np.zeros((frame_length, self.max_tokens, 1))*0.0,
            'mask_detection'        : np.zeros((frame_length, self.max_tokens, 1))*0.0,
            'fid'                   : np.zeros((frame_length, self.max_tokens, 1))*0.0,
        }

        output_data = {
            'pose_shape'            : np.zeros((frame_length, self.max_tokens, f_, 229))*0.0,
            'action_label_ava'      : np.zeros((frame_length, self.max_tokens, f_, 80))*0.0,
            'action_label_kinetics' : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_detection'         : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_gt'                : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_gt_kinetics'       : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0
        }

        meta_data = {
            'frame_name'            : [],
            'frame_bbox'            : [],
            'frame_size'            : [],
            'frame_conf'            : [],
        }
        
        # additional supervision keys based on loss string, add it to output data.
        if("kp_l1" in self.opt.loss_type or self.opt.render.walker=="PL"):
            output_data['vitpose'] = np.zeros((frame_length, self.max_tokens, f_, 25, 3))*0.0


        # argment input data for extra features in the input.
        if("img" in self.opt.extra_feat.enable):
            input_data['img'] = np.zeros((224, 224, 3))*0.0

        if("action" in self.opt.extra_feat.enable):
            input_data['action_emb'] = np.zeros((frame_length, self.max_tokens, 80))*0.0
            
        if("mvit" in self.opt.extra_feat.enable):
            input_data['mvit_emb'] = np.zeros((frame_length, self.max_tokens, 1152))*0.0

        if("hmr" in self.opt.extra_feat.enable):
            input_data['hmr_emb'] = np.zeros((frame_length, self.max_tokens, 2048))*0.0

        if("appe" in self.opt.extra_feat.enable or "T" in self.opt.render.walker):
            input_data['appe_emb'] = np.zeros((frame_length, self.max_tokens, 4096))*0.0
            
        if("objects" in self.opt.extra_feat.enable):
            input_data['objects_emb'] = np.zeros((frame_length, self.max_tokens, 80))*0.0
            
        if("clip" in self.opt.extra_feat.enable):
            input_data['clip_emb'] = np.zeros((frame_length, self.max_tokens, 512))*0.0
            
        if("vitpose" in self.opt.extra_feat.enable):
            input_data['vitpose_emb'] = np.zeros((frame_length, self.max_tokens, 75))*0.0
        
        if("Dpose" in self.opt.extra_feat.enable):
            input_data['D_pose_shape'] = np.zeros((frame_length, self.max_tokens, 229))*0.0
            
        if("mae" in self.opt.extra_feat.enable):
            input_data['mae_emb'] = np.zeros((frame_length, self.max_tokens, 768))*0.0


        
        return input_data, output_data, meta_data
    
    def read_image(self, img_path):
        img_       = cv2.imread(img_path)
        img_       = cv2.resize(img_, (224, 224))
        
        def normalize(im):
            """Performs image normalization."""
            # [0, 255] -> [0, 1]
            im = im.astype(np.float32) / 255.0
            # Color norm
            im = color_norm(im, _MEAN, _STD)
            # HWC -> CHW
            im = im.transpose([2, 0, 1])
            return im
        
        def color_norm(im, mean, std):
            """Performs per-channel normalization."""
            for i in range(3):
                im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
            return im
        

        return normalize(img_)


    def __getitem__(self, idx):

        t1 = time.time()
        f_ = self.opt.num_smpl_heads
        detection_data, list_of_frames, video_name = self.read_from_phalp_fast(idx)
        start_frame, end_frame, keyframe   = self.get_start_end_frame(list_of_frames, f_)
        # print("T1:", time.time()-t1)
        
        if(self.train): frame_length_ = self.opt.frame_length
        else:           frame_length_ = max(end_frame - start_frame, self.opt.frame_length)
       
        input_data, output_data, meta_data = self.initiate_dict(frame_length_, f_)
        
        # for n>1 setting, read all other tracks.
        other_tracks = []
        if(self.max_tokens>1):
            tracks_ = self.track2video[idx]
            tracks_tmp = tracks_.copy()
            np.random.shuffle(tracks_tmp)
            for i in range(min(self.max_tokens-1, len(tracks_tmp))):
                other_tracks.append(self.read_from_phalp_other(tracks_tmp[i]))
                
        delta = 0
        if(end_frame>frame_length_):
            end_frame = end_frame - start_frame
            start_frame = 0
    
    
        input_data['pose_shape'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]            = (detection_data[self.pose_key][start_frame:end_frame].copy() - self.mean_pose_shape[None, :, :])/(self.std_pose_shape[None, :, :] + 1e-10)
        input_data['has_detection'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]         = detection_data["has_detection"][start_frame:end_frame]
        input_data['fid'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]                   = detection_data["fid"][start_frame:end_frame]
        
        if(self.opt.use_relative_pose):
            m_ego = get_ego_rot(detection_data[self.pose_key][start_frame:end_frame].copy(), end_frame, start_frame)

        output_data['pose_shape'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]        = detection_data[self.pose_key][start_frame:end_frame]
        output_data['has_detection'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]     = detection_data["has_detection"][start_frame:end_frame]
        output_data['has_gt'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]            = detection_data["has_gt"][start_frame:end_frame]
        
        # add kinetics labels
        if("kinetics" in self.opt.action_space and not("ava" in video_name)):
            class_label = self.kinetics_annotations[video_name.split("kinetics-train_")[1][:11]]
            output_data['has_gt_kinetics'][:, self.ego_id:self.ego_id+1, 0, :] = 1.0
            output_data['action_label_kinetics'][:, self.ego_id:self.ego_id+1, 0, :] = class_label[1]
        
        if(self.opt.mixed_training>0):
            TMP_1 = detection_data["action_label_ava"][start_frame:end_frame].copy()
            TMP_2 = detection_data["action_emb"][start_frame:end_frame].copy()
            if(self.opt.mixed_training==1):
                TMP_ = (TMP_1 + TMP_2)/2.0
            if(self.opt.mixed_training==2):
                TMP_1[:, :, self.a1-1] = TMP_2[:, :, self.a1-1]
                TMP_ = TMP_1.copy()
            if(self.opt.mixed_training==3):
                TMP_1[:, :, self.a1-1] = (TMP_2[:, :, self.a1-1]+TMP_1[:, :, self.a1-1])/2.0
                TMP_ = TMP_1.copy()
            if(self.opt.mixed_training==4):
                TMP_1 = 0.75*TMP_1+0.25*TMP_2
                TMP_ = TMP_1.copy()
            if(self.opt.mixed_training==5):
                TMP_1 = 0.5*TMP_1+0.5*TMP_2
                TMP_ = TMP_1.copy()
            if(self.opt.mixed_training==6):
                TMP_1 = 0.9*TMP_1+0.1*TMP_2
                TMP_ = TMP_1.copy()
            if(self.opt.mixed_training==10):
                hug_class = np.array([70])
                TMP_2[:, :, hug_class-1] = TMP_1[:, :, hug_class-1]
                TMP_ = TMP_2.copy()
        else:
            TMP_ = detection_data["action_label_ava"][start_frame:end_frame].copy()
            
        if(self.opt.ava.predict_valid):
            action_label_ava_ = np.zeros((end_frame-start_frame, 1, self.opt.ava.num_action_classes))
            # TMP_ = detection_data["action_label_ava"][start_frame:end_frame].copy()
            action_label_ava_[:, :, :self.opt.ava.num_valid_action_classes] = TMP_[:, :, self.ava_valid_classes-1]
            output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :] = action_label_ava_.copy()
        else:
            action_label_ava_ = np.zeros((end_frame-start_frame, 1, self.opt.ava.num_action_classes))
            action_label_ava_[:, :, :] = TMP_[:, :, :]
            output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]  = action_label_ava_.copy()
        
        
    
        
        # if(self.opt.full_gt_supervision>0):
        #     idx_2 = np.where(output_data['has_gt'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :] == 2)[0]
        #     if(len(idx_2)>0 and self.opt.full_gt_supervision==1):
        #         output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :] = action_label_ava_[idx_2[0]].copy()
                
        #     if(len(idx_2)>0 and self.opt.full_gt_supervision>=200 and self.opt.full_gt_supervision<300):
        #         wx_ = self.opt.full_gt_supervision-200
        #         a1 = np.min(idx_2)
        #         b1 = np.max(idx_2)
        #         s_  = a1 - wx_//2 if a1 - wx_//2 >= 0 else 0
        #         e_  = b1 + wx_//2 if b1 + wx_//2 < end_frame-start_frame else end_frame-start_frame
        #         output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :][s_:e_] = action_label_ava_[idx_2[0]].copy()
                
        # if(self.opt.full_pesudo_supervision>0 and not("ava" in video_name)):
        #     idx_2 = np.where(output_data['has_gt'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :] == 1)[0]
            
        #     if(len(idx_2)>0 and self.opt.full_pesudo_supervision==1):
        #         output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :][idx_2] = action_label_ava_[idx_2, :, :].mean(0)
        #     if(len(idx_2)>0 and self.opt.full_pesudo_supervision==2):
        #         output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :] = action_label_ava_[idx_2, :, :].mean(0)
        #     if(len(idx_2)>0 and self.opt.full_pesudo_supervision==3):
        #         for i_ in range(len(idx_2)//12):
        #             output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :][idx_2[i_*12:(i+1)*12]] = action_label_ava_[idx_2[i_*12:(i+1)*12], :, :].mean(0)
        #     if(len(idx_2)>0 and self.opt.full_pesudo_supervision==4):
        #         output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :][idx_2] = np.clip(action_label_ava_[idx_2, :, :] + 0.2*(action_label_ava_[idx_2, :, :] - action_label_ava_[idx_2, :, :].mean(0)), 0.0, 1.0)
                
        #     # 200 for averging. if 212, then window size is 12
        #     if(len(idx_2)>0 and self.opt.full_pesudo_supervision>=200 and self.opt.full_pesudo_supervision<300):
        #         wx_ = self.opt.full_pesudo_supervision-200
        #         for i_ in range(len(idx_2)//wx_+1):
        #             output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :][idx_2[i_*wx_:(i_+1)*wx_]] = action_label_ava_[idx_2[i_*wx_:(i_+1)*wx_], :, :].mean(0)
                    
        #     # 400 for spikes. if 412, then window size is 12
        #     if(len(idx_2)>0 and self.opt.full_pesudo_supervision>=400 and self.opt.full_pesudo_supervision<500):
        #         wx_ = self.opt.full_pesudo_supervision-400
        #         for i_ in range(len(idx_2)//wx_+1):
        #             output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :][idx_2[i_*wx_:(i_+1)*wx_]] = np.clip(action_label_ava_[idx_2[i_*wx_:(i_+1)*wx_], :, :] + self.opt.full_pesudo_supervision_c1 * (action_label_ava_[idx_2[i_*wx_:(i_+1)*wx_], :, :] - action_label_ava_[idx_2[i_*wx_:(i_+1)*wx_], :, :].mean(0)), 0.0, 1.0)

        # extra features.
        if("mvit_emb" in input_data.keys()):
            input_data['mvit_emb'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]              = detection_data["mvit_emb"][start_frame:end_frame]
        if("vitpose_emb" in input_data.keys()):
            input_data['vitpose_emb'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]           = detection_data["vitpose"][start_frame:end_frame].reshape(end_frame-start_frame, 1, 75)
        if("mae_emb" in input_data.keys()):
            input_data['mae_emb'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]               = detection_data["mae_emb"][start_frame:end_frame]#.reshape(end_frame-start_frame, 1, 75)
        if("img" in input_data.keys()):
            input_data['img'] = self.read_image(detection_data['frame_name'][keyframe])
            
            
        if(self.max_tokens>1):
            # for n>1 setting, read all other tracks.
            base_idx = detection_data['fid'][start_frame:end_frame]
            for ot in range(len(other_tracks)):
                other_detection_data = other_tracks[ot]
                other_base_idx = other_detection_data['fid']
                # import pdb; pdb.set_trace()
                if(other_base_idx[0]>base_idx[-1]): continue
                elif(other_base_idx[-1]<base_idx[0]): continue
                elif(other_base_idx[0]>=base_idx[0] and other_base_idx[-1, 0, 0]<=base_idx[-1, 0, 0]):
                    other_start_frame = 0
                    other_end_frame = len(other_base_idx)
                    delta = other_base_idx[0, 0, 0] - base_idx[0, 0, 0]
                elif(other_base_idx[0, 0, 0]>=base_idx[0, 0, 0] and other_base_idx[-1, 0, 0]>base_idx[-1, 0, 0]):
                    other_start_frame = 0
                    other_end_frame = len(other_base_idx) - (other_base_idx[-1, 0, 0] - base_idx[-1, 0, 0])
                    delta = other_base_idx[0, 0, 0] - base_idx[0, 0, 0]
                elif(other_base_idx[0, 0, 0]<base_idx[0, 0, 0] and other_base_idx[-1, 0, 0]<=base_idx[-1, 0, 0]):
                    other_start_frame = base_idx[0, 0, 0] - other_base_idx[0, 0, 0]
                    other_end_frame = len(other_base_idx)
                    delta = -other_start_frame
                else:
                    continue
                
                other_start_frame = int(other_start_frame)
                other_end_frame = int(other_end_frame)
                delta = int(delta)
                
                # print("other_start_frame:", other_start_frame, "other_end_frame:", other_end_frame, "delta:", delta)
                input_data['pose_shape'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]            = (other_detection_data[self.pose_key][other_start_frame:other_end_frame].copy() - self.mean_pose_shape[None, :, :])/(self.std_pose_shape[None, :, :] + 1e-10)
                input_data['has_detection'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]         = other_detection_data["has_detection"][other_start_frame:other_end_frame]
                input_data['fid'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]                   = other_detection_data["fid"][other_start_frame:other_end_frame]
                
                if(self.opt.use_relative_pose):
                    m_other = get_ego_rot(other_detection_data[self.pose_key][other_start_frame:other_end_frame].copy(), other_end_frame, other_start_frame)
                    M_other = np.zeros((end_frame-start_frame, 4, 4))
                    # M_other[:, 0, 0] = M_other[:, 1, 1] = M_other[:, 2, 2] = M_other[:, 3, 3] = 1
                    M_other[delta+other_start_frame:delta+other_end_frame, :, :] = m_other.copy()
                    rela_pose_shape = np.matmul(np.linalg.inv(m_ego), M_other)
                    input_data['relative_pose'][start_frame:end_frame, ot+1:ot+2, :] = rela_pose_shape.reshape((end_frame-start_frame, 1, 16)).copy()
                    
                    del m_other, M_other, rela_pose_shape
                    # joblib.dump([rela_pose_shape, m_ego, M_other, m_other, output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :].copy(), other_detection_data[self.pose_key][other_start_frame:other_end_frame].copy(), output_data['has_gt'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :].copy()], os.path.join("stats/" + str(idx) + "_" + str(ot) + ".pkl"))
                    # import pdb; pdb.set_trace()
                    
                output_data['pose_shape'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :]        = other_detection_data[self.pose_key][other_start_frame:other_end_frame]
                output_data['has_detection'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :]     = other_detection_data["has_detection"][other_start_frame:other_end_frame]
                if(self.opt.loss_on_others_action):
                    output_data['has_gt'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :]            = other_detection_data["has_gt"][other_start_frame:other_end_frame]
                
                if(self.opt.ava.predict_valid):
                    action_label_ava_ = np.zeros((other_end_frame-other_start_frame, 1, self.opt.ava.num_action_classes))
                    TMP_ = other_detection_data["action_label_ava"][other_start_frame:other_end_frame].copy()
                    action_label_ava_[:, :, :self.opt.ava.num_valid_action_classes] = TMP_[:, :, self.ava_valid_classes-1]
                    output_data['action_label_ava'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :] = action_label_ava_.copy()
                else:
                    output_data['action_label_ava'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :] = other_detection_data["action_label_ava"][other_start_frame:other_end_frame]
                
                # extra features.
                if("mvit_emb" in input_data.keys()):
                    input_data['mvit_emb'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]             = other_detection_data["mvit_emb"][other_start_frame:other_end_frame]
                if("vitpose_emb" in input_data.keys()):
                    input_data['vitpose_emb'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]          = other_detection_data["vitpose"][other_start_frame:other_end_frame].reshape(other_end_frame-other_start_frame, 1, 75)
            
                del other_detection_data, other_base_idx, action_label_ava_
        
            
        if(self.opt.debug):
            import ipdb; ipdb.set_trace()
          
        if(not(self.train)):
            # add meta data for rendering
            meta_data['frame_name'] = detection_data["frame_name"][start_frame:end_frame].copy()
            meta_data['frame_size'] = detection_data["frame_size"][start_frame:end_frame].copy()
            meta_data['frame_bbox'] = detection_data["frame_bbox"][start_frame:end_frame].copy()
            meta_data['frame_conf'] = detection_data["frame_conf"][start_frame:end_frame].copy()
            
            if(end_frame-start_frame<frame_length_):
                for i in range((frame_length_)-(end_frame-start_frame)):
                    meta_data['frame_name'].append("-1")
                    meta_data['frame_size'].append(np.array([0, 0]))
                    meta_data['frame_bbox'].append(np.array([0.0, 0.0, 0.0, 0.0]))
                    meta_data['frame_conf'].append(0)
                    
        del detection_data
        if(self.opt.use_relative_pose): 
            del m_ego
            
        if(self.opt.use_optimized_pose):
            output_data['pose_shape'] = np.nan_to_num(output_data['pose_shape'])
            input_data['pose_shape']  = np.nan_to_num(input_data['pose_shape'])
            input_data['relative_pose'] = np.nan_to_num(input_data['relative_pose'])
            
        
        return input_data, output_data, meta_data, video_name

