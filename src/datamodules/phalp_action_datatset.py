import copy
import os
import socket
import time
import traceback

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

# import utils.rotation_conversions as geometry
from src.utils.utils import task_divider

# from joblib import Parallel, delayed

MAX_CACHE_SIZE = 1000

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

class PHALP_action_dataset(Dataset):
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
            "ava_val_1"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.307_ava_val/results_slowfast_v3/",         1, 20],
            "ava_val_2"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_val/results_slowfast_v5/",         1, 20],
            "ava_val_3"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v5/",         1, 20],
            "ava_rend_"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6///",       1, 20],            
            "ava_rend_climb"   : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v7////",      1, 20],            
            "kinetics_train"   : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v6/",  self.opt.kinetics.sampling_factor, 20],
            "kinetics_val"     : ["",                "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results_slowfast_v6/",    1, 20],
            "ava_train"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results_slowfast_v6/",       self.opt.ava.sampling_factor, 20],
            "ava_val"          : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6/",         1, 20],
            "ava_train_4"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results_slowfast_v7/",       self.opt.ava.sampling_factor, 20],
            "ava_val_4"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v7/",         1, 20],
            "ava_val_5"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.403_ava_val/results_slowfast_v11/",        1, 1],
            "avaK_train_6"     : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v20/",        1, 1],
            "ava_val_6"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_val/results_slowfast_v20/",        1, 1],  # to be deleted
            
            
            
            
            "kinetics_train_7" : ["",               "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v21/",  self.opt.kinetics.sampling_factor, 20],
            "ava_train_7"      : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v21/",       1, 1],
            "avaK_train_7"     : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21/",      1, 1],
            "ava_val_8"        : ["ava",             "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v21/",         1, 1],
            
        }
        
        if(self.train):
            
            self.list_of_datasets = opt.train_dataset.split(",")
            for dataset in self.list_of_datasets:
                print(self.dataset_roots[dataset][0])
                self.get_dataset(root_dir=self.dataset_roots[dataset][1], 
                                filter_seq=self.dataset_roots[dataset][0],
                                num_sample=self.dataset_roots[dataset][2], 
                                min_track_length=self.dataset_roots[dataset][3], 
                                total_num_tracks=None)        

        else:
            self.list_of_datasets = opt.test_dataset.split(",")
            for dataset in self.list_of_datasets:
                self.get_dataset(root_dir=self.dataset_roots[dataset][1], 
                                filter_seq=self.opt.test_class, 
                                min_track_length=1, 
                                total_num_tracks=None)
            
            self.data = task_divider(self.data, self.opt.test_batch_id, self.opt.number_of_processes)
            self.data = np.sort(self.data)
            if(self.opt.test_batch_id==-2):
                self.data = self.data[:1000] 
            elif(self.opt.test_batch_id==-3):
                self.data = self.data[:30]
            elif(self.opt.test_batch_id==-10):
                self.data = self.data[:10000]
            elif(self.opt.test_batch_id==-4):
                self.data = self.data[:4000]
            else:
                self.data = self.data

        self.ava_valid_classes = np.load("data/ava_valid_classes.npy")
        print("Number of tracks: ", len(self.data))
        
        self.stable_datapoint = None

    def __len__(self):
        return len(self.data)

    def get_dataset(self, root_dir="", filter_seq="posetrack-train", num_sample=1, min_track_length=20, total_num_tracks=None):
        count    = 0
        count_f  = 0
        path_npy = "data/"+"_".join(root_dir.split("/"))+".npy"
        path_pkl = "data/"+"_".join(root_dir.split("/"))+".pkl"
        
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

        for i_, video_ in enumerate(list_of_files):
            if(video_.endswith(".pkl") and filter_seq in video_):
                if(int(video_.split("_")[-1][:-4])>=min_track_length):
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

    def parse_phalp(self, frame_data, p_, key='smpl', input_is_true=True, load_rgb=False, keyframe=False, extra_keys=None):

        out = {
            "has_detection" : [0],
            "has_gt"        : [0],
            "is_keyframe"   : [0],
        }
        
        if(frame_data['has_detection']):
            has_gt          = 0
            gloabl_ori_     = frame_data[key]['global_orient'].reshape(1, -1)
            body_pose_      = frame_data[key]['body_pose'].reshape(1, -1)
            betas_          = frame_data[key]['betas'].reshape(1, -1)
            location_       = copy.deepcopy(frame_data['embedding'][4096+4096+90:4096+4096+93].reshape(1, -1))
            # keypoints_      = copy.deepcopy(frame_data['embedding'][4096+4096:4096+4096+90].reshape(1, -1))-0.5 
            location_[0, 2] = location_[0, 2]/200.0
            pose_shape_     = np.hstack([gloabl_ori_, body_pose_, betas_, location_]) 
            if(self.opt.use_mean_std and input_is_true): 
                pose_shape_ = (pose_shape_ - self.mean_pose_shape)/(self.std_pose_shape + 1e-10)

            # Load Action labels AVA, and Kinetics (needs to be fixed!)
            action_label_ava = np.zeros((1, self.opt.ava.num_action_classes))
            action_label_kinetics = 0
            if("ava" in self.opt.action_space):
                if("gt_class_2" in frame_data.keys()): #  and self.opt.ava.gt_type in ["gt", "both", "all"]
                    action_label = frame_data['gt_class_2']
                    if(action_label is not None):
                        action_label_ava[0, action_label-1] = 1
                        has_gt = 2
                elif("gt_class" in frame_data.keys()): #  and self.opt.ava.gt_type in ["gt", "both", "all"]
                    action_label = frame_data['gt_class']
                    if(action_label is not None):
                        action_label_ava[0, action_label-1] = 1
                        has_gt = 2
                elif("action_emb_p" in frame_data.keys()):
                    action_label = frame_data['action_emb_p'].reshape(1, -1)
                    action_label_ava = action_label[:, 1:]
                    has_gt = 1
                else:
                    has_gt = 0
                
                if(self.opt.ava.predict_valid):
                    action_label_ava_ = np.zeros((1, self.opt.ava.num_action_classes))
                    action_label_ava_[:, :self.opt.ava.num_valid_action_classes] = action_label_ava[:, self.ava_valid_classes-1]
                    action_label_ava = copy.deepcopy(action_label_ava_)
                    
            if("kinetics" in self.opt.action_space):
                if("action_label_kinetics" in frame_data.keys()):
                    action_label_kinetics = frame_data['action_label_kinetics'][1]
                

            # keyframe check for validation data
            # if(not(self.train)):
            #     # TODO: give a list of is_keyframe for each frame
            #     import ipdb; ipdb.set_trace()
            #     pass
            
            # by default always include pose and shape vectors, and action label
            out['pose_shape']             = pose_shape_
            out['action_label_ava']       = action_label_ava
            out['action_label_kinetics']  = [action_label_kinetics]
            out['has_detection']          = [1]
            out['has_gt']                 = [has_gt]

            if(extra_keys is not None):
                # add output keys:
                if(not(input_is_true)):
                    if("vitpose" in extra_keys):
                        h, w = frame_data['size']
                        nis  = max(h, w)
                        top, left = (nis - h)//2, (nis - w)//2,
                        out['vitpose'] = copy.deepcopy(frame_data['vitpose'].reshape(1, 25, 3))
                        out['vitpose'][:, :, 0] = (out['vitpose'][:, :, 0] + left)/nis - 0.5
                        out['vitpose'][:, :, 1] = (out['vitpose'][:, :, 1] + top)/nis - 0.5
                
                # add input keys:
                else:
                    # Load RGB image and include it in the output
                    if("img" in extra_keys and keyframe):
                        img_       = cv2.imread(os.path.join("../TENET/", frame_data['img_path']))
                        img_       = cv2.resize(img_, (224, 224))
                        out['img'] = img_/255.0

                    if("action_emb" in extra_keys):     
                        try:
                            out['action_emb'] = frame_data['action_emb_p'].reshape(1, -1)[:, 1:]
                        except:
                            out['action_emb'] = np.zeros((1, self.opt.ava.num_action_classes))
                        
                    if("mvit_emb" in extra_keys):     
                        out['mvit_emb'] = frame_data['action_features'].reshape(1, -1)[:, :]

                    if("hmr_emb" in extra_keys):     
                        out['hmr_emb'] = frame_data['embedding'][4096:4096+2048].reshape(1, -1)
                    
                    if("appe_emb" in extra_keys):     
                        out['appe_emb'] = frame_data['embedding'][:4096].reshape(1, -1)
                        
                    if("objects_emb" in extra_keys):   
                        objects     = frame_data['objects'][0] 
                        objects_emb = np.zeros((1, 80))
                        for o_ in objects: objects_emb[:, o_] += 1
                        out['objects_emb'] = objects_emb
                    
                    if("clip_emb" in extra_keys):   
                        out['clip_emb'] = frame_data['clip_features'].reshape(1, -1) # 512
                        
                    if("vitpose_emb" in extra_keys):   
                        h, w = frame_data['size']
                        nis  = max(h, w)
                        top, left = (nis - h)//2, (nis - w)//2,
                        vitpose = copy.deepcopy(frame_data['vitpose'].reshape(1, 25, 3))
                        vitpose[:, :, 0] = (vitpose[:, :, 0] + left)/nis - 0.5
                        vitpose[:, :, 1] = (vitpose[:, :, 1] + top)/nis - 0.5
                        out['vitpose_emb'] = vitpose.reshape(1, -1)
        
        return out
    
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

    def read_from_phalp(self, idx):
        try:
            detection_data   = joblib.load(self.data[idx])
        except:
            np.save("_TMP/bad_files/" + self.data[idx].split("/")[-1].split(".")[0] + ".npy", [self.data[idx]])
            detection_data   = joblib.load(self.data[0])
            
        list_of_frames   = list(detection_data.keys())
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
        }

        output_data = {
            'pose_shape'            : np.zeros((frame_length, self.max_tokens, f_, 229))*0.0,
            'action_label_ava'      : np.zeros((frame_length, self.max_tokens, f_, 80))*0.0,
            'action_label_kinetics' : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_detection'         : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_gt'                : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'is_keyframe'           : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0
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

        
        return input_data, output_data, meta_data

    def __getitem__(self, idx):

        f_                   = self.opt.num_smpl_heads
        
        detection_data, list_of_frames, video_name = self.read_from_phalp(idx)
        
        start_frame, end_frame, keyframe   = self.get_start_end_frame(list_of_frames, f_)
        
        if(self.train): frame_length_ = self.opt.frame_length
        else:           frame_length_ = max(end_frame - start_frame, self.opt.frame_length)
       
        input_data, output_data, meta_data = self.initiate_dict(frame_length_, f_)
        
        # for n>1 setting, read all other tracks for finetuning.
        other_tracks = []
        if(self.max_tokens>1):
            tracks_ = self.track2video[idx]
            np.random.shuffle(tracks_)
            for i in range(min(self.max_tokens-1, len(tracks_))):
                other_tracks.append(self.read_from_phalp_other(tracks_[i]))
        
        for i_, t_ in enumerate(list(range(start_frame, end_frame))):
            
            # if(self.opt.load_other_tracks or not(self.train)):
                
            # else:
            #     ego_id = np.random.randint(0, self.max_tokens)
            ego_id = 0
              
            frame_name   = list_of_frames[t_]
            frame_data   = detection_data[frame_name]
            phalp_out    = self.parse_phalp(frame_data, 0, key = 'smpl', input_is_true=True, keyframe = i_==keyframe, extra_keys = input_data.keys())
            for key in phalp_out.keys(): 
                if(key in input_data.keys() and key!="img"):
                    input_data[key][i_, ego_id, :] = phalp_out[key][0] if phalp_out['has_detection'][0] else 0 
                if(key=="img" and phalp_out['has_detection'][0]):
                    input_data["img"] = phalp_out[key]

            for fi in range(f_):
                frame_name_f = list_of_frames[t_+fi]
                phalp_out_f  = self.parse_phalp(detection_data[frame_name_f], 0, key = 'smpl', input_is_true=False, extra_keys = output_data.keys())
                for key in phalp_out_f.keys(): 
                    if(key in output_data.keys() and key!="img"):
                        output_data[key][i_, ego_id, fi] = phalp_out_f[key][0] if phalp_out_f['has_detection'][0] else 0 
            
            # for n>1 setting, read all other tracks.
            if(self.max_tokens>1): # and self.opt.load_other_tracks
                for tid, track_other in enumerate(other_tracks):
                    if(frame_name not in track_other.keys()): continue
                    frame_data_other = track_other[frame_name]
                    phalp_out_other  = self.parse_phalp(frame_data_other, 0, key = 'smpl', input_is_true=True, keyframe = i_==keyframe, extra_keys = input_data.keys())
                    for key in phalp_out_other.keys(): 
                        if(key in input_data.keys() and key!="img"):
                            input_data[key][i_, tid+1, :] = phalp_out_other[key][0] if phalp_out_other['has_detection'][0] else 0 
                        if(key=="img" and phalp_out_other['has_detection'][0]):
                            input_data["img"] = phalp_out_other[key]

                    for fi in range(f_):
                        frame_name_f = list_of_frames[t_+fi]
                        if(frame_name_f not in track_other.keys()): continue
                        phalp_out_f_other  = self.parse_phalp(track_other[frame_name_f], 0, key = 'smpl', input_is_true=False, extra_keys = output_data.keys())
                        for key in phalp_out_f_other.keys(): 
                            if(key in output_data.keys() and key!="img"):
                                output_data[key][i_, tid+1, fi] = phalp_out_f_other[key][0] if phalp_out_f_other['has_detection'][0] else 0 
                    
            
            if(not(self.train)):
                # add meta data for rendering
                meta_data['frame_name'].append(frame_data['frame_path'])
                meta_data['frame_size'].append(np.array(frame_data['size']) if phalp_out['has_detection'][0] else np.array([0, 0]))
                meta_data['frame_bbox'].append(frame_data['bbox'] if phalp_out['has_detection'][0] else np.array([0.0, 0.0, 0.0, 0.0]))
                meta_data['frame_conf'].append(frame_data['conf'] if phalp_out['has_detection'][0] else 0)

        
        if(end_frame-start_frame<frame_length_ and not(self.train)):
            for i in range((frame_length_)-(end_frame-start_frame)):
                meta_data['frame_name'].append("-1")
                meta_data['frame_size'].append(np.array([0, 0]))
                meta_data['frame_bbox'].append(np.array([0.0, 0.0, 0.0, 0.0]))
                meta_data['frame_conf'].append(0)
        
        return input_data, output_data, meta_data, video_name
    
