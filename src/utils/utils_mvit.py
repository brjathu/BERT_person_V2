import argparse
import copy
import os

import joblib
import numpy as np
from optimization.main import optimize_track_pose
from tqdm import tqdm

from src.utils.rotation_conversions import *
from src.utils.utils import task_divider


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--batch_id', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--num_of_process', type=int, default=100, help='num of workers to use')
    parser.add_argument('--dataset_slowfast', type=str, default="kinetics-train", help='num of workers to use')
    parser.add_argument('--add_optimization', type=int)

    opt = parser.parse_args()
    return opt


def create_fast_tracklets(data, save_path, add_optimization=0):
    list_of_frames = list(data.keys())
    frame_length   = len(list_of_frames)
    
    array_fid              = np.zeros((frame_length, 1, 1))-1
    
    array_pose_shape       = np.zeros((frame_length, 1, 229))
    array_opt_pose_shape   = np.zeros((frame_length, 1, 229))
    array_action_label_ava = np.zeros((frame_length, 1, 80))
    # array_pseudo_label_ava = np.zeros((frame_length, 1, 80))
    array_has_detection    = np.zeros((frame_length, 1, 1))
    array_has_gt           = np.zeros((frame_length, 1, 1))
    
    array_vitpose          = np.zeros((frame_length, 1, 25, 3))
    
    array_action_emb       = np.zeros((frame_length, 1, 80))*0.0
    # array_mvit_emb         = []
    # array_mvit_emb_idx     = []
    array_mvit_emb         = np.zeros((frame_length, 1, 1152))*0.0
    # array_hmr_emb          = np.zeros((frame_length, 1, 2048))*0.0
    # array_appe_emb         = np.zeros((frame_length, 1, 4096))*0.0
    # array_objects_emb      = np.zeros((frame_length, 1, 80))*0.0
    # array_clip_emb         = np.zeros((frame_length, 1, 512))*0.0
    # array_vitpose          = np.zeros((frame_length, 1, 75))
    
    array_frame_name = []
    array_frame_bbox = []
    array_frame_size = []
    array_frame_conf = []

    if(add_optimization>0):
        optimized_poses = optimize_track_pose(track=file_path, bs=frame_length)
    
    for fid, frame_name in enumerate(list_of_frames):
        
        frame_data = data[frame_name]
        array_fid[fid, 0, 0] = frame_data["fid"]
        
        if(frame_data['has_detection']):
            has_gt          = 0
            gloabl_ori_     = frame_data['smpl']['global_orient'].reshape(1, -1)
            body_pose_      = frame_data['smpl']['body_pose'].reshape(1, -1)
            betas_          = frame_data['smpl']['betas'].reshape(1, -1)
            location_       = copy.deepcopy(frame_data['embedding'][4096+4096+90:4096+4096+93].reshape(1, -1))
            location_[0, 2] = location_[0, 2]/200.0
            pose_shape_     = np.hstack([gloabl_ori_, body_pose_, betas_, location_]) 

            
            # Load Action labels AVA, and Kinetics (needs to be fixed!)
            action_label_ava = np.zeros((1, 80))
            if("gt_class_2" in frame_data.keys()):
                action_label = frame_data['gt_class_2']
                if(action_label is not None):
                    action_label_ava[0, action_label-1] = 1
                    has_gt = 2
            elif("action_emb_p" in frame_data.keys()):
                action_label = frame_data['action_emb_p'].reshape(1, -1)
                action_label_ava = action_label[:, 1:]
                has_gt = 1
            else:
                has_gt = 0

            array_pose_shape[fid, 0, :] = pose_shape_
            array_action_label_ava[fid, 0, :] = action_label_ava
            array_has_detection[fid, 0, 0] = 1
            array_has_gt[fid, 0, 0] = has_gt

            h, w = frame_data['size']
            nis  = max(h, w)
            top, left = (nis - h)//2, (nis - w)//2,
            vitpose = copy.deepcopy(frame_data['vitpose'].reshape(1, 25, 3))
            vitpose[:, :, 0] = (vitpose[:, :, 0] + left)/nis - 0.5
            vitpose[:, :, 1] = (vitpose[:, :, 1] + top)/nis - 0.5
            array_vitpose[fid] = vitpose

            array_action_emb[fid, 0, :] = frame_data['action_emb_p'].reshape(1, -1)[:, 1:]
            array_mvit_emb[fid, 0, :]   = frame_data['action_features'].reshape(1, -1)
            
            
            #### for the optimized pose part ####
            if(add_optimization>0):
                ratio         = max(frame_data['size'])/256.0
                opt_location_ = copy.deepcopy(optimized_poses['camera_translation'][fid].reshape(1, -1))
                opt_location_[0, 2] = opt_location_[0, 2]*ratio/200.0
                opt_betas_    = copy.deepcopy(optimized_poses['betas'][fid].reshape(1, -1))
                opt_gloabl_ori_ = copy.deepcopy(optimized_poses['global_orient'][fid].reshape(1, -1))
                opt_gloabl_ori_ = axis_angle_to_matrix(torch.from_numpy(opt_gloabl_ori_)).numpy().reshape(1, -1)
                opt_body_pose_  = copy.deepcopy(optimized_poses['body_pose'][fid].reshape(23, 3))
                opt_body_pose_  = axis_angle_to_matrix(torch.from_numpy(opt_body_pose_)).reshape(1, -1).numpy()
                array_opt_pose_shape[fid, 0, :] = np.hstack([opt_gloabl_ori_, opt_body_pose_, opt_betas_, opt_location_])
            
        array_frame_name.append(frame_data['frame_path'])
        array_frame_size.append(np.array(frame_data['size']) if frame_data['has_detection'] else np.array([0, 0]))
        array_frame_bbox.append(frame_data['bbox'] if frame_data['has_detection'] else np.array([0.0, 0.0, 0.0, 0.0]))
        array_frame_conf.append(frame_data['conf'] if frame_data['has_detection'] else 0)
        
    joblib.dump(
        {
            'fid'                : array_fid,
            'pose_shape'         : array_pose_shape, 
            'opt_pose_shape'     : array_opt_pose_shape, 
            'action_label_ava'   : array_action_label_ava, 
            'has_detection'      : array_has_detection, 
            'has_gt'             : array_has_gt, 
            'vitpose'            : array_vitpose, 
            'action_emb'         : array_action_emb, 
            'mvit_emb'           : array_mvit_emb,
            'frame_name'         : array_frame_name,
            'frame_size'         : array_frame_size,
            'frame_bbox'         : array_frame_bbox,
            'frame_conf'         : array_frame_conf,
        }, 
    save_path)



if __name__ == '__main__':

    # # setup video model
    device           = 'cuda'
    args             = parse_option()

    if(args.dataset_slowfast == "ava-train"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23_1/"
    elif(args.dataset_slowfast == "ava-val"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v21/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v21_1/"
    elif(args.dataset_slowfast == "avaK-train"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21_1/"
    elif(args.dataset_slowfast == "kinetics-train"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v21/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v21_1/"
    else:
        raise ValueError("Invalid dataset")
    
    
    path_npy          = "data/phalp_fast_23_files_" + root_base.replace('/', '_') + '.npy'
    if(os.path.exists(path_npy)):
        phalp_files_ = np.load(path_npy)
    else:
        phalp_files_ = np.sort([i for i in os.listdir(root_base) if i.endswith('.pkl')])
        np.save(path_npy, phalp_files_)
        
    phalp_files       = phalp_files_.copy()
    np.random.seed(2)
    np.random.shuffle(phalp_files)
    phalp_files      = task_divider(phalp_files, args.batch_id, args.num_of_process)
        
    os.makedirs(root_new, exist_ok=True)
    
    for file_ in tqdm(phalp_files):
        file_path = root_base + file_
        save_path = root_new + file_
        
        # check for save path exist
        if(os.path.exists(save_path)):
            continue
        
        try:
            data = joblib.load(file_path)
        except:
            print("Error in file: ", file_path)
            continue
        
        create_fast_tracklets(data, save_path, args.add_optimization)
        
        