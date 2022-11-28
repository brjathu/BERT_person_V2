import os
import warnings
from functools import partial

import detectron2
import numpy as np
import pytorchvideo
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import (  # Another option is slowfast_r50_detection
    slow_r50_detection, slowfast_r50_detection)
from pytorchvideo.transforms.functional import (clip_boxes_to_image,
                                                short_side_scale_with_boxes,
                                                uniform_temporal_subsample)
from torchvision.transforms._functional_video import normalize

warnings.filterwarnings("ignore")

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32
    crop_size = 256, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )
    
    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )
    
    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )
    
    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), ori_boxes


# def get_tracks(phalp_tracks):

#     tracks_ids     = []
#     tracks_dict    = {}
#     for frame_ in phalp_tracks:
#         tracks_ids += phalp_tracks[frame_]['tracked_ids']

#     tracks_ids = list(set(tracks_ids))
    
#     for track_id in tracks_ids:
#         tracks_dict[track_id] = {}
#         for fid, frame_name in enumerate(phalp_tracks.keys()):
#             list_tracks = np.array(phalp_tracks[frame_name]['tracked_ids'])
#             list_time   = np.array(phalp_tracks[frame_name]['tracked_time'])
#             idx_        = np.where(list_tracks==track_id)[0]
#             if(len(idx_)==1):
#                  time_       = list_time[idx_[0]]
#                  if(time_==0):
#                     track_results = {
#                         'track_id'   : track_id,
#                         'frame_name' : frame_name,
#                         'time'       : time_,
#                         'bbox'       : phalp_tracks[frame_name]['bbox'][idx_[0]],
#                         'center'     : phalp_tracks[frame_name]['center'][idx_[0]],
#                         'scale'      : phalp_tracks[frame_name]['scale'][idx_[0]],
#                         'conf'       : phalp_tracks[frame_name]['conf'][idx_[0]],
#                         # 'appe'       : phalp_tracks[frame_name]['appe'][idx_[0]],
#                         # 'pose'       : phalp_tracks[frame_name]['pose'][idx_[0]],
#                         # 'loca'       : phalp_tracks[frame_name]['loca'][idx_[0]],
#                         'embedding'  : phalp_tracks[frame_name]['embedding'][idx_[0]],
#                         'smpl'       : phalp_tracks[frame_name]['smpl'][idx_[0]],
#                         'camera'     : phalp_tracks[frame_name]['camera'][idx_[0]],
#                         'img_path'   : phalp_tracks[frame_name]['img_path'][idx_[0]],
#                         'img_name'   : phalp_tracks[frame_name]['img_name'][idx_[0]],
#                         'openpose'   : phalp_tracks[frame_name]['openpose'][idx_[0]],
#                         'mask_name'  : phalp_tracks[frame_name]['mask_name'][idx_[0]]
#                     }
#                     tracks_dict[track_id][frame_name] = track_results
#     return tracks_dict



def get_tracks(phalp_tracks):

    tracks_ids     = []
    tracks_dict    = {}
    for frame_ in phalp_tracks: tracks_ids += phalp_tracks[frame_]['tracked_ids']

    tracks_ids = list(set(tracks_ids))
    
    for track_id in tracks_ids:
        tracks_dict[track_id] = {}
        list_valid_frames     = []
        list_frame_names      = []
        for fid, frame_name in enumerate(phalp_tracks.keys()):
            list_tracks  = np.array(phalp_tracks[frame_name]['tid'])
            list_tracks2 = np.array(phalp_tracks[frame_name]['tracked_ids'])
            list_time   = np.array(phalp_tracks[frame_name]['tracked_time'])
            idx_        = np.where(list_tracks==track_id)[0]
            idx_2        = np.where(list_tracks2==track_id)[0]
            # if(len(idx_)==1 and list_time[idx_[0]]==0):
            if(len(idx_)>=1 and list_time[idx_[0]]==0):
                time_       = list_time[idx_[0]]
                track_results = {
                    'track_id'   : track_id,
                    'frame_name' : frame_name,
                    'time'       : time_,
                    'bbox'       : phalp_tracks[frame_name]['bbox'][idx_[0]],
                    'center'     : phalp_tracks[frame_name]['center'][idx_[0]],
                    'scale'      : phalp_tracks[frame_name]['scale'][idx_[0]],
                    'conf'       : phalp_tracks[frame_name]['conf'][idx_[0]],
                    'size'       : phalp_tracks[frame_name]['size'][idx_[0]],
                    'embedding'  : phalp_tracks[frame_name]['embedding'][idx_[0]],
                    'smpl'       : phalp_tracks[frame_name]['smpl'][idx_[0]],
                    'smpl_gt'    : phalp_tracks[frame_name]['smpl_gt'][idx_[0]],
                    'camera'     : phalp_tracks[frame_name]['camera'][idx_[0]],
                    'img_path'   : phalp_tracks[frame_name]['img_path'][idx_[0]],
                    'img_name'   : phalp_tracks[frame_name]['img_name'][idx_[0]],
                    'vitpose'   : phalp_tracks[frame_name]['vitpose'][idx_[0]],
                    'objects'   : phalp_tracks[frame_name]['objects'][idx_[0]],
                    # 'annot'     : phalp_tracks[frame_name]['annot'][idx_[0]],
                    
                    'mask_name'  : phalp_tracks[frame_name]['mask_name'][idx_[0]],
                    'has_detection' : True,
                    'fid'           : fid,
                    'frame_path'   : phalp_tracks[frame_name]['frame_path'],
                }
                
                if("annot" in phalp_tracks[frame_name].keys()):
                    track_results['annot'] = phalp_tracks[frame_name]['annot'][idx_[0]]
                else:
                    track_results['annot'] = []
            
                tracks_dict[track_id][frame_name] = track_results
                list_valid_frames.append(1)
            else:
                track_results = {
                    'track_id'      : -1,
                    'frame_name'    : frame_name,
                    'has_detection' : False,
                    'fid'           : fid,
                    'time'          : -1,
                    'bbox'          : None,
                    'size'          : None,
                    'action_gt'     : None,
                    'frame_path'    : phalp_tracks[frame_name]['frame_path'],
                }
                tracks_dict[track_id][frame_name] = track_results
                list_valid_frames.append(0)
            
            list_frame_names.append(frame_name)
        
        list_valid_frames2 = np.array(list_valid_frames)
        loc_ = np.where(list_valid_frames2==1)
        s_ = np.min(loc_)
        e_ = np.max(loc_)
        
        if(e_-s_<15):
            s_ = s_ - 5 if s_-5>=0 else 0
            e_ = e_ + 5 if e_+5<len(list_valid_frames) else len(list_valid_frames)-1
            
        for i, fname in enumerate(list_frame_names):
            if(i<s_ or i>e_):
                del tracks_dict[track_id][fname]

    return tracks_dict




