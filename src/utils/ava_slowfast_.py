from bdb import Breakpoint
from functools import partial
import os
import numpy as np

import os
import joblib
import cv2
import csv
import torch
import argparse
from tqdm import tqdm
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection, slowfast_r50_detection # Another option is slowfast_r50_detection
import warnings
warnings.filterwarnings("ignore")
from src.utils.utils_slowfast import get_person_bboxes, ava_inference_transform, get_tracks
from src.utils.visualization_slowfast import VideoVisualizer
from src.utils.utils import task_divider
from src.ActivityNet.Evaluation.ava.np_box_ops import iou

activation  = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--batch_id', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--num_of_process', type=int, default=100, help='num of workers to use')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt      = parse_option()

    # setup video model
    device = 'cuda'
    dataset_slowfast = "kinetics-train"

    video_model = slowfast_r50_detection(True) # Another option is slowfast_r50_detection
    video_model.detection_head.roi_layer.register_forward_hook(get_activation('roi_layer'))
    video_model = video_model.eval().to(device)
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/ava_action_list.pbtxt')
    video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.5)
    
    if(dataset_slowfast=="ava-val"):
        phalp_path = '/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_val/results/'
        save_path  = '/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_val/results_slowfast_v3/'
        os.makedirs(save_path, exist_ok=True)
        phalp_files = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
        phalp_files = task_divider(phalp_files, opt.batch_id, opt.num_of_process)

    if(dataset_slowfast=="ava-train"):
        phalp_path         = '/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results/'
        save_path          = '/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results_slowfast_v4/'
        os.makedirs(save_path, exist_ok=True)
        phalp_files        = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
        phalp_files        = task_divider(phalp_files, opt.batch_id, opt.num_of_process)
        annotated          = 0
        ava_annotations    = joblib.load("../TENET/_DATA/ava_train_annot.pkl")

    if(dataset_slowfast=="kinetics-train"):
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v5/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_video_v5/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        phalp_files = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
        phalp_files = task_divider(phalp_files, opt.batch_id, opt.num_of_process)
        kinetics_annotations    = joblib.load("data/kinetics_annot_train.pkl")
        
    if(dataset_slowfast=="kinetics-val"):
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results_slowfast_v5/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results_slowfast_video_v5/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        phalp_files = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
        phalp_files = task_divider(phalp_files, opt.batch_id, opt.num_of_process)
        kinetics_annotations    = joblib.load("data/kinetics_annot_val.pkl")

    for phalp_file in tqdm(phalp_files):
        
        if(dataset_slowfast=="ava-val"):
            ############################################################################################################
            # This is the only change from slowfast_v3.py for ava valiation
            ############################################################################################################
            key_frame          = phalp_file.split('ava-val_')[-1][:-4]+'.jpg'
            phalp_             = joblib.load(phalp_path + phalp_file)
            all_frames         = list(phalp_.keys())
            tracks_dict        = get_tracks(phalp_)
            for track_id in tracks_dict.keys():
                # check if the file exists
                if os.path.exists(save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    continue
                
                if(key_frame not in tracks_dict[track_id].keys()): continue
                
                if(tracks_dict[track_id][key_frame]['has_detection']):

                    loc_all        = np.where(np.array(all_frames)==key_frame)[0][0]
                    start_         = loc_all-32 if loc_all-32>0 else 0
                    end_           = loc_all+32 if loc_all+32<len(all_frames) else len(all_frames)
                    list_of_frames = all_frames[start_:end_]
                    images         = [cv2.imread(phalp_[i]['frame_path']) for i in list_of_frames]
                    images         = np.array(images)
                    images         = images.transpose(3,0,1,2)
                    images         = images[::-1, :, :, :].copy()

                    bboxes         = np.array(tracks_dict[track_id][key_frame]['bbox']).reshape(1,4)
                    bboxes_        = np.concatenate([bboxes[:, :2], bboxes[:, :2] + bboxes[:, 2:4]], 1)

                    inputs, inp_boxes, _ = ava_inference_transform(torch.from_numpy(images), bboxes_)
                    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                    preds = video_model(inputs, inp_boxes.to(device))

                    preds= preds.to('cpu')
                    preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

                    ab                      = video_visualizer.class_names.copy()
                    top_scores, top_classes = torch.topk(preds, k=80)
                    top_scores              = top_scores.detach().cpu().float().numpy()
                    top_classes             = top_classes.detach().cpu().int().numpy()
                    top_labels              = []
                    for i in range(len(top_classes[0])): top_labels.append(ab.get(top_classes[0][i], "n/a"))

                    for frame_ in tracks_dict[track_id].keys():
                        tracks_dict[track_id][frame_]['action_score'] = top_scores
                        tracks_dict[track_id][frame_]['action_class'] = top_classes
                        tracks_dict[track_id][frame_]['action_label'] = top_labels.copy()
                        tracks_dict[track_id][frame_]['action_emb_p'] = preds.detach().cpu().float().numpy()
                    joblib.dump(tracks_dict[track_id], save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", compress=3)



        if(dataset_slowfast=="ava-train"):
            ############################################################################################################
            # This is the only change from slowfast_v3.py for ava training
            ############################################################################################################
            
            phalp_             = joblib.load(phalp_path + phalp_file)
            all_frames         = list(phalp_.keys())
            tracks_dict        = get_tracks(phalp_)
            
            key_frame_id       = phalp_file.split('ava-train_')[-1][:-11]
            key_frame_         = int(phalp_file.split('ava-train_')[-1][-10:-4])
            key_frames         = [key_frame_id + '_%06d.jpg'%(key_frame_+i*30,) for i in range(-2,3)]
            
            for track_id in tracks_dict.keys():
                # check if the file exists
                if os.path.exists(save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    continue
                
                for key_frame in key_frames:
                    if(key_frame not in tracks_dict[track_id].keys()): continue
                    
                    if(tracks_dict[track_id][key_frame]['has_detection']):
                        
                        if(key_frame[:11]+"/"+key_frame in ava_annotations.keys()):
                            frame_annot    = np.array(ava_annotations[key_frame[:11]+"/"+key_frame])
                        else:
                            continue
                        bbox_annot     = frame_annot[:, 2:6].astype(np.float32)

                        bbox_detect    = tracks_dict[track_id][key_frame]['bbox'].reshape(1,4)
                        size_          = tracks_dict[track_id][key_frame]['size']
                        bbox_detect_   = np.concatenate([bbox_detect[:, :2], bbox_detect[:, :2] + bbox_detect[:, 2:4]], 1)
                        bbox_detect_norm = np.array([[bbox_detect_[0, 0]/size_[1], bbox_detect_[0, 1]/size_[0], bbox_detect_[0, 2]/size_[1], bbox_detect_[0, 3]/size_[0]]])
                        iou_overlap    = iou(bbox_detect_norm, bbox_annot)
                        ab             = iou_overlap[0]>0.5
                        frame_annot_   = frame_annot[ab, :]

                        loc_all        = np.where(np.array(all_frames)==key_frame)[0][0]
                        start_         = loc_all-32 if loc_all-32>0 else 0
                        end_           = loc_all+32 if loc_all+32<len(all_frames) else len(all_frames)
                        list_of_frames = all_frames[start_:end_]
                        images         = [cv2.imread(phalp_[i]['frame_path']) for i in list_of_frames]
                        images         = np.array(images)
                        images         = images.transpose(3,0,1,2)
                        images         = images[::-1, :, :, :].copy()

                        bboxes         = np.array(tracks_dict[track_id][key_frame]['bbox']).reshape(1,4)
                        bboxes_        = np.concatenate([bboxes[:, :2], bboxes[:, :2] + bboxes[:, 2:4]], 1)

                        inputs, inp_boxes, _ = ava_inference_transform(torch.from_numpy(images), bboxes_)
                        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                        preds = video_model(inputs, inp_boxes.to(device))

                        preds= preds.to('cpu')
                        preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

                        ab                      = video_visualizer.class_names.copy()
                        top_scores, top_classes = torch.topk(preds, k=80)
                        top_scores              = top_scores.detach().cpu().float().numpy()
                        top_classes             = top_classes.detach().cpu().int().numpy()
                        top_labels              = []
                        for i in range(len(top_classes[0])): top_labels.append(ab.get(top_classes[0][i], "n/a"))
                        
                        start_f          = loc_all-15 if loc_all-15>0 else 0
                        end_f            = loc_all+15 if loc_all+15<len(tracks_dict[track_id].keys()) else len(tracks_dict[track_id].keys())
                        list_of_frames_f = list(tracks_dict[track_id].keys())[start_f:end_f]
                        for frame_ in list_of_frames_f:
                            tracks_dict[track_id][frame_]['action_score'] = top_scores
                            tracks_dict[track_id][frame_]['action_class'] = top_classes
                            tracks_dict[track_id][frame_]['action_label'] = top_labels.copy()
                            tracks_dict[track_id][frame_]['action_emb_p'] = preds.detach().cpu().float().numpy()
                            if(len(frame_annot_)>0):
                                tracks_dict[track_id][frame_]['gt_annot']     = frame_annot_
                                tracks_dict[track_id][frame_]['gt_class']     = frame_annot_[:, 6].astype(np.int32)
                                annotated += 1.0/len(tracks_dict[track_id].keys())
                            else:
                                tracks_dict[track_id][frame_]['gt_annot']     = None
                                tracks_dict[track_id][frame_]['gt_class']     = None

                        joblib.dump(tracks_dict[track_id], save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", compress=3)




        if(dataset_slowfast=="kinetics-train" or dataset_slowfast=="kinetics-val"):
            ############################################################################################################
            # This is the only change from slowfast_v4.py for kinetics
            ############################################################################################################

            make_video         = 0
            key_frame          = phalp_file.split(dataset_slowfast + '_')[-1][:-4]
            phalp_             = joblib.load(phalp_path + phalp_file)
            all_frames         = list(phalp_.keys())
            tracks_dict        = get_tracks(phalp_)
            for track_id in tracks_dict.keys():

                # check if the file exists
                if os.path.exists(save_path + key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    continue
                
                for fname_x in tracks_dict[track_id].keys():
                    tracks_dict[track_id][fname_x]['action_label_kinetics'] = kinetics_annotations[key_frame] #preds.detach().cpu().float().numpy()
                
                list_of_frames   = np.array(list(tracks_dict[track_id].keys()))
                list_of_paths    = np.array([tracks_dict[track_id][i]['frame_path'] for i in list_of_frames ])
                list_of_bbox     = np.array([tracks_dict[track_id][i]['bbox'] for i in list_of_frames ])
                list_of_time     = np.array([tracks_dict[track_id][i]['time'] for i in list_of_frames ])
                tracked_time     = list_of_time == 0

                list_of_frames   = list_of_frames[tracked_time]
                list_of_paths    = list_of_paths[tracked_time]
                list_of_bbox     = list_of_bbox[tracked_time]

                gif_imgs         = []
                video_created    = 0
                NUM_STEPS        = 30
                NUM_FRAMES       = 64
                list_iter        = list(range(len(list_of_frames)//NUM_STEPS + 1))
                for t_, time_stamp in enumerate(list_iter):    
                    print("Generating predictions for time stamp: {} sec".format(time_stamp))
                    
                    start_      = time_stamp * NUM_STEPS
                    end_        = (time_stamp + 1) * NUM_STEPS if len(list_of_frames) > (time_stamp + 1) * NUM_STEPS else len(list_of_frames)
                    time_stamp_ = list_of_frames[start_:end_]
                    
                    mid_        = (start_ + end_)//2
                    mid_frame   = list_of_frames[mid_]
                    mid_id      = tracks_dict[track_id][mid_frame]['fid']
                    
                    start_id    = mid_id - NUM_FRAMES//2 if mid_id - NUM_FRAMES//2 > 0 else 0
                    end_id      = mid_id + NUM_FRAMES//2 if mid_id + NUM_FRAMES//2 < len(all_frames) else len(all_frames)
                    

                    list_of_paths_all = list([phalp_[i]['frame_path'] for i in all_frames[start_id:end_id]])
                    inp_imgs    = np.array([cv2.imread(i) for i in list_of_paths_all])
                    if(len(inp_imgs)==0): continue
                    inp_imgs    = inp_imgs.transpose(3,0,1,2)
                    inp_imgs    = inp_imgs[::-1, :, :, :].copy()
                    
                    inp_bboxes        = np.array([tracks_dict[track_id][mid_frame]['bbox']])
                    inp_bboxes        = np.concatenate([inp_bboxes[:, :2], inp_bboxes[:, :2] + inp_bboxes[:, 2:4]], 1)
                    predicted_boxes   = inp_bboxes 
                    predicted_boxes   = predicted_boxes.reshape(1, 4)
                    if len(predicted_boxes) == 0: continue
                        
                    # Preprocess clip and bounding boxes for video action recognition.
                    inputs, inp_boxes, _ = ava_inference_transform(torch.from_numpy(inp_imgs), predicted_boxes)
                    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                    
                    # The model here takes in the pre-processed video clip and the detected bounding boxes.
                    if isinstance(inputs, list):
                        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                    else:
                        inputs = inputs.unsqueeze(0).to(device)
                    preds      = video_model(inputs, inp_boxes.to(device))
                    action_emb = activation['roi_layer']  # torch.Size([13, 2304, 7, 7])
                    action_emb = action_emb.mean(dim=(2,3))

                    for fname in time_stamp_:
                        tracks_dict[track_id][fname]['action_emb'] = action_emb[0].detach().cpu().numpy()

                    preds = preds.to('cpu')
                    preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
                    
                    ab                      = video_visualizer.class_names.copy()
                    top_scores, top_classes = torch.topk(preds, k=80)
                    top_scores              = top_scores.detach().cpu().float().numpy()
                    top_classes             = top_classes.detach().cpu().int().numpy()
                    top_labels              = []
                    for i in range(len(top_classes[0])): 
                        top_labels.append(ab.get(top_classes[0][i], "n/a"))

                    for fname in time_stamp_:
                        tracks_dict[track_id][fname]['action_score'] = top_scores
                        tracks_dict[track_id][fname]['action_class'] = top_classes
                        tracks_dict[track_id][fname]['action_label'] = top_labels.copy()
                        tracks_dict[track_id][fname]['action_emb_p'] = preds.detach().cpu().float().numpy()
                        
                        


                    # Plot predictions on the video and save for later visualization.
                    if(make_video):
                        images   = torch.from_numpy(inp_imgs)
                        images   = images.permute(1,2,3,0)
                        images   = images/255.0
                        out_img_pred = video_visualizer.draw_clip_range(images, preds, torch.from_numpy(predicted_boxes))

                    if(video_created==0 and make_video):
                        height, width   = out_img_pred[0].shape[0], out_img_pred[0].shape[1]
                        video_save_path = save_path_video +  key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".mp4"
                        video_file      = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(width,height))
                        video_created   = 1

                    if(make_video and video_created==1):
                        for img_ in out_img_pred:
                            img_ = (255*img_).astype(np.uint8)
                            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                            video_file.write(img_)

                if(make_video): video_file.release()
                if(make_video): print('Predictions are saved to the video file: ', video_save_path)
                joblib.dump(tracks_dict[track_id], save_path + key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl")
        