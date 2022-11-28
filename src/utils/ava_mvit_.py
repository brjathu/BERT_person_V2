import argparse
import csv
import os
import warnings
from bdb import Breakpoint
from functools import partial

import cv2
import detectron2
import joblib
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
from tqdm import tqdm

warnings.filterwarnings("ignore")
import clip
from PIL import Image
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils import logging
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from slowfast.visualization.async_predictor import (AsyncDemo, AsyncVis,
                                                    draw_predictions)
from slowfast.visualization.predictor import ActionPredictor, Predictor
from slowfast.visualization.utils import TaskInfo
# from src.utils.visualization_slowfast import VideoVisualizer
from slowfast.visualization.video_visualizer import VideoVisualizer

from src.ActivityNet.Evaluation.ava.np_box_ops import iou
from src.utils.utils import task_divider
from src.utils.utils_mvit import create_fast_tracklets
from src.utils.utils_slowfast import (ava_inference_transform,
                                      get_person_bboxes, get_tracks)
import copy


import cv2
import math
import numpy as np
import os
import pickle
import random

from tqdm import tqdm
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as transforms

# Per-channel mean and standard deviation (in RGB order)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_im_size = 224
_max_shift = 0
_lower_crop = False

def lower_center_crop(im, crop_size):
    """Performs lower-center cropping."""
    w = im.shape[1]
    x = math.ceil((w - crop_size) / 2)
    return im[-crop_size:, x:(x + crop_size), :]


def lower_random_shift_crop(im, crop_size, max_shift):
    w = im.shape[1]
    x = (w - crop_size) // 2
    assert x + max_shift + crop_size <= w
    assert x - max_shift >= 0
    shift = np.random.randint(-max_shift, max_shift + 1)
    x = x + shift
    return im[-crop_size:, x:(x + crop_size), :]


def resize_crop(im, size, max_shift=0, lower_crop=True):
    """Performs image resize and crop."""
    if max_shift > 0:
        # (480, 640, 3) -> (448, 448, 3) or (480, 480, 3)
        crop_size = 448 if lower_crop else 480
        im = lower_random_shift_crop(im, crop_size, max_shift)
        # (448, 448, 3) or (480, 480, 3) -> (size, size, 3)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        # (480, 640, 3) -> (448, 448, 3) or (480, 480, 3)
        crop_size = 448 if lower_crop else 480
        im = lower_center_crop(im, crop_size)
        # (448, 448, 3) or (480, 480, 3) -> (size, size, 3)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    return im

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

def process_image(im):
    im_pil = Image.fromarray(im).convert("RGB")
    # im_pil = self._augmentation(im_pil)
    im = np.array(im_pil).astype(np.float32)
    im = resize_crop(im, _im_size, max_shift=_max_shift, lower_crop=_lower_crop)
    im = normalize(im)
    return im





logger = logging.get_logger(__name__)
torch.cuda.empty_cache()

activation  = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--batch_id', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--num_of_process', type=int, default=100, help='num of workers to use')
    parser.add_argument('--dataset_slowfast', type=str, default="kinetics-train", help='num of workers to use')
    parser.add_argument('--add_clip', type=int, default=1)
    # parser.add_argument('--add_mvp', type=int, default=1)

    opt = parser.parse_args()
    return opt

def run_slowfast(t, list_of_imgs, mid_frame_bboxes, make_video):            
    images = []
    c = 0
    
    # import ipdb; ipdb.set_trace()
    
    for frame in list_of_imgs:
        img_ = cv2.imread(frame)
        
        # th_1 = 1.54
        # th_2 = 0.648
        
        th_1 = 1.5
        th_2 = 0.67
        
        # th_1 = 1.4
        # th_2 = 0.7
        
        if(center_crop):
            h, w = img_.shape[:2]
            r_   = w/h
            if(r_>th_1 and w>h):
                w_   = h*th_1
                l_   = int((w-w_)//2)
                img_ = img_[:, l_:-l_, ::]
                if(c==0):
                    mid_frame_bboxes = [[mid_frame_bboxes[0][0]-l_, mid_frame_bboxes[0][1], mid_frame_bboxes[0][2]-l_, mid_frame_bboxes[0][3]]]
                    mid_frame_bboxes = np.array(mid_frame_bboxes)
            elif(r_<th_2 and w<h):
                h_   = w/th_2
                t_   = int((h-h_)//2)
                img_ = img_[t_:-t_, :, ::]
                if(c==0):
                    mid_frame_bboxes = [[mid_frame_bboxes[0][0], mid_frame_bboxes[0][1]-t_, mid_frame_bboxes[0][2], mid_frame_bboxes[0][3]-t_]]
                    mid_frame_bboxes = np.array(mid_frame_bboxes)
        c += 1
        images.append(img_)
    
    task = TaskInfo()
    
    task.img_height    = images[0].shape[0]
    task.img_width     = images[0].shape[1]
    task.crop_size     = test_crop_size
    task.clip_vis_size = clip_vis_size
    task.add_frames(t, images)
    task.add_bboxes(torch.from_numpy(mid_frame_bboxes).float().cuda())
    with torch.no_grad(): 
        task = video_model(task)
    
    if(make_video):
        frames_out = draw_predictions(task, video_vis)
    else:
        frames_out = None
        
    return task, frames_out

def generate_pseudo_labels(t_, list_of_paths_all, list_of_paths_window, predicted_boxes, make_video, add_clip):
    
    ####################### pySlowFast ############################
    with torch.no_grad():
        task_, imgs_ = run_slowfast(t_, list_of_paths_all, predicted_boxes, make_video)
        preds      = task_.action_preds[0]
        feats      = task_.action_preds[1]
        preds      = preds.to('cpu')
        feats      = feats.to('cpu')
        preds      = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)

    ####################### CLIP ############################
    if(args.add_clip):
        image_features = []
        for i in range(len(list_of_paths_window)):
            clip_image = preprocess(Image.open(list_of_paths_window[i])).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features.append(clip_model.encode_image(clip_image))
                # text_features  = clip_model.encode_text(ava_class_text)
                # logits_per_image, logits_per_text = clip_model(clip_image, ava_class_text)
                # probs = logits_per_image.softmax(dim=-1).cpu()
                # probs = torch.cat([torch.zeros(probs.shape[0],1), probs], dim=1)
  
        return [preds, imgs_, feats], [0, image_features]
    
    ####################### MVP ############################
    # if(args.add_mvp):
    #     image_features = []
    #     for i in range(len(list_of_paths_window)):
    #         clip_image = preprocess(Image.open(list_of_paths_window[i])).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             image_features.append(clip_model.encode_image(clip_image))
    #             # text_features  = clip_model.encode_text(ava_class_text)
    #             # logits_per_image, logits_per_text = clip_model(clip_image, ava_class_text)
    #             # probs = logits_per_image.softmax(dim=-1).cpu()
    #             # probs = torch.cat([torch.zeros(probs.shape[0],1), probs], dim=1)
  
    #     return [preds, imgs_, feats], [0, image_features]
    
    return [preds, imgs_, feats], [0, 0]
    # return [preds, imgs_, feats], [0, 0], [0, 0]
        
if __name__ == '__main__':

    # # setup video model
    device           = 'cuda'
    # path_to_config   = "/private/home/jathushan/3D/slowfast/configs/AVA/c2/SLOWFAST_64x2_R101_50_50.yaml"
    path_to_config   = "/private/home/jathushan/3D/slowfast/configs/AVA/MViT-L-312_masked.yaml"
    center_crop = False
    if("MViT" in path_to_config): 
        center_crop = True
    args             = parse_option()
    args.cfg         = path_to_config
    args.opts        = None
    dataset_slowfast = args.dataset_slowfast
    make_video       = False

    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)
    
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run demo with config:")
    logger.info(cfg)
    
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=0.5,
        lower_thres=0.3,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )
    
    video_model    = Predictor(cfg=cfg, gpu_id=None)
    # video_model.model.register_forward_hook(get_activation('s1_roi'))
    # video_model.model.register_forward_hook(get_activation('projection'))
    buffer_size    = cfg.DEMO.BUFFER_SIZE
    seq_length     = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    test_crop_size = cfg.DATA.TEST_CROP_SIZE
    clip_vis_size  = cfg.DEMO.CLIP_VIS_SIZE
    
    if(args.add_clip):
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # clip_model, preprocess = clip.load("ViT-L/14", device=device)
        ava_class_labels       = video_vis.class_names.copy()
        ava_class_text         = clip.tokenize(ava_class_labels).to(device)
        
    # if(args.add_mvp):
    #     import mvp
    #     mvp_model = mvp.load("vitb-mae-egosoup")
    #     mvp_model.freeze()          
    
    
    if(dataset_slowfast=="kinetics-train"):
        root_path        = '/datasets01/Kinetics400_Frames/frames/'
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v22/'
        save_path_x      = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v22x/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_video_v22/'
        fast_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v22_1/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_x, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        os.makedirs(fast_path, exist_ok=True)
        path_npy          = "data/kinetics-train_20_phalp_files_" + save_path.replace('/', '_') + '.npy'
        if(os.path.exists(path_npy)):
            phalp_files_ = np.load(path_npy)
        else:
            phalp_files_ = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
            np.save(path_npy, phalp_files_)
            
        phalp_files       = phalp_files_.copy()
        np.random.seed(2)
        np.random.shuffle(phalp_files)
        
        phalp_files = task_divider(phalp_files, args.batch_id, args.num_of_process)
        kinetics_annotations    = joblib.load("data/kinetics_annot_train.pkl")
        
    if(dataset_slowfast=="kinetics-val"):
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results_slowfast_v7/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_val/results_slowfast_video_v7/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        phalp_files = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
        phalp_files = task_divider(phalp_files, args.batch_id, args.num_of_process)
        kinetics_annotations    = joblib.load("data/kinetics_annot_val.pkl")
    
    
    if(dataset_slowfast=="ava-val"):
        root_path        = '/datasets01/AVA/080720/frames/'
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v23/'
        save_path_x      = '/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v23_x/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_video_v23/'
        fast_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v23_1/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_x, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        os.makedirs(fast_path, exist_ok=True)
        path_npy          = "data/ava-val_21_phalp_files_" + save_path.replace('/', '_') + '.npy'
        if(os.path.exists(path_npy)):
            phalp_files_ = np.load(path_npy)
        else:
            phalp_files_ = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
            np.save(path_npy, phalp_files_)
            
        phalp_files       = phalp_files_.copy()
        np.random.seed(5)
        np.random.shuffle(phalp_files)
        phalp_files      = task_divider(phalp_files, args.batch_id, args.num_of_process)
        annotated        = 0
        ava_annotations  = joblib.load("../TENET/_DATA/ava_val_annot.pkl")
        # phalp_files      = ['ava-val_1j20qq1JyX4_002820.pkl']
        # phalp_files      = ['ava-val_1j20qq1JyX4_004200.pkl']
        
        
    if(dataset_slowfast=="ava-train"):
        root_path        = '/datasets01/AVA/080720/frames/'
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23/'
        save_path_x      = '/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23_x/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_video_v23/'
        fast_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23_1/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_x, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        os.makedirs(fast_path, exist_ok=True)
        path_npy          = "data/ava-train_23_phalp_files_" + save_path.replace('/', '_') + '.npy'
        if(os.path.exists(path_npy)):
            phalp_files_ = np.load(path_npy)
        else:
            phalp_files_ = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
            np.save(path_npy, phalp_files_)
            
        phalp_files       = phalp_files_.copy()
        np.random.seed(2)
        np.random.shuffle(phalp_files)
        phalp_files      = task_divider(phalp_files, args.batch_id, args.num_of_process)
        annotated        = 0
        ava_annotations  = joblib.load("../TENET/_DATA/ava_train_annot.pkl")

    if(dataset_slowfast=="avaK-train"):
        root_path        = '/datasets01/Kinetics400_Frames/frames/'
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21/'
        save_path_x      = '/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21_x/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_video_v21/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_x, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        path_npy          = "data/avaK-train_20_phalp_files_" + save_path.replace('/', '_') + '.npy'
        if(os.path.exists(path_npy)):
            phalp_files_ = np.load(path_npy)
        else:
            phalp_files_ = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
            np.save(path_npy, phalp_files_)
            
        phalp_files       = phalp_files_.copy()
        np.random.seed(2)
        np.random.shuffle(phalp_files)
        phalp_files      = task_divider(phalp_files, args.batch_id, args.num_of_process)
        annotated        = 0
        ava_annotations  = {}
        
    if(dataset_slowfast=="youtube"):
        root_path        = '/private/home/jathushan/3D/TENET/_DEMO/'
        phalp_path       = '/checkpoint/jathushan/TENET/out/Videos_v4.404_youtube/results/'
        save_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.404_youtube/results_slowfast_v22/'
        save_path_x      = '/checkpoint/jathushan/TENET/out/Videos_v4.404_youtube/results_slowfast_v22x/'
        save_path_video  = '/checkpoint/jathushan/TENET/out/Videos_v4.404_youtube/results_slowfast_video_v22/'
        fast_path        = '/checkpoint/jathushan/TENET/out/Videos_v4.404_youtube/results_slowfast_v22_1/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_x, exist_ok=True)
        os.makedirs(save_path_video, exist_ok=True)
        os.makedirs(fast_path, exist_ok=True)
        path_npy          = "data/youtube_20_phalp_files_" + save_path.replace('/', '_') + '.npy'
        if(os.path.exists(path_npy)):
            phalp_files_ = np.load(path_npy)
        else:
            phalp_files_ = np.sort([i for i in os.listdir(phalp_path) if i.endswith('.pkl')])
            np.save(path_npy, phalp_files_)
            
        phalp_files       = phalp_files_.copy()
        np.random.seed(2)
        np.random.shuffle(phalp_files)
        
        phalp_files = task_divider(phalp_files, args.batch_id, args.num_of_process)
        
    for phalp_file in tqdm(phalp_files):
    
        
        if(dataset_slowfast=="youtube"):
            
            try:
                phalp_             = joblib.load(phalp_path + phalp_file)
                all_frames         = list(phalp_.keys())
                tracks_dict        = get_tracks(phalp_)
                
                for track_id in tracks_dict.keys():
                    video_created  = 0
                    # check if the file exists
                    # if os.path.exists(fast_path + key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    #     continue
                    
                    # if os.path.exists(save_path + key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    #     continue
                    
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
                    NUM_STEPS        = 6
                    NUM_FRAMES       = seq_length
                    list_iter        = list(range(len(list_of_frames)//NUM_STEPS + 1))
                    for t_, time_stamp in enumerate(list_iter):    
                        print("Generating predictions for time stamp: {} sec".format(time_stamp))
                    
                    
                        start_      = time_stamp * NUM_STEPS
                        end_        = (time_stamp + 1) * NUM_STEPS if (time_stamp + 1) * NUM_STEPS < len(list_of_frames) else len(list_of_frames)
                        time_stamp_ = list_of_frames[start_:end_]
                        if(len(time_stamp_)==0): continue

                        mid_        = (start_ + end_)//2
                        mid_frame   = list_of_frames[mid_]
                        mid_bbox    = list_of_bbox[mid_]
                        mid_id      = tracks_dict[track_id][mid_frame]['fid']
                        start_id    = mid_id - NUM_FRAMES//2 if mid_id - NUM_FRAMES//2 > 0 else 0
                        end_id      = mid_id + NUM_FRAMES//2 if mid_id + NUM_FRAMES//2 < len(all_frames) else len(all_frames)
                            
                            
                        # import ipdb; ipdab.set_trace()
                        # get the frames for mvit
                        list_of_all_frames = []
                        list_of_all_paths  = []
                        for i in range(-NUM_FRAMES//2,NUM_FRAMES//2):
                            frame_id = int(mid_frame[:-4])
                            if frame_id+i<=0:      frame_id = 1
                            elif frame_id+i>128: frame_id = 128
                            else:                  frame_id = frame_id+i
                            frame_name_all = '%06d.jpg'%(frame_id,)
                            frame_key_all  = phalp_file[8:-4] + "/img/" + '%05d.jpg'%(frame_id,)
                            # import ipdb; ipdb.set_trace()
                            # import pdb; pdb.set_trace()
                            if frame_name_all not in list_of_all_frames:
                                if(os.path.exists(root_path + "/" + frame_key_all)):
                                    list_of_all_frames.append(frame_name_all)
                                    list_of_all_paths.append(root_path + "/" + frame_key_all)
                                        
                                        
                        list_of_paths_all = list([phalp_[i]['frame_path'] for i in all_frames[start_id:end_id]])
                        list_of_paths_window = list([phalp_[i]['frame_path'] for i in time_stamp_])
                        predicted_boxes   = mid_bbox.reshape(1, 4)
                        predicted_boxes   = np.concatenate([predicted_boxes[:, :2], predicted_boxes[:, :2] + predicted_boxes[:, 2:4]], 1)
                        slowfast_, clip_  = generate_pseudo_labels(t_, list_of_paths_all, list_of_paths_window, predicted_boxes, make_video, args.add_clip)
                        for i_, frame_ in enumerate(time_stamp_):
                            tracks_dict[track_id][frame_]['action_emb_p']      = slowfast_[0].cpu().float().numpy()                        
                            tracks_dict[track_id][frame_]['action_features']   = slowfast_[2].cpu().float().numpy()                        
                            if(args.add_clip): tracks_dict[track_id][frame_]['clip_features']     = clip_[1][i_].cpu().numpy()
                        
                        if(video_created==0 and make_video):
                            imgs_           = slowfast_[1]
                            height, width   = imgs_[0].shape[0], imgs_[0].shape[1]
                            video_save_path = save_path_video +  phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".mp4"
                            video_file      = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(width,height))
                            video_created   = 1

                        if(make_video and video_created==1):
                            for img in imgs_:
                                video_file.write(img)
    
                        print(t_, len(list_of_paths_all), len(predicted_boxes))
                        
                    if(make_video): video_file.release()
                    if(make_video): print('Predictions are saved to the video file: ', video_save_path)
                    # joblib.dump(tracks_dict[track_id], save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", compress=3)
                    create_fast_tracklets(copy.deepcopy(tracks_dict[track_id]), fast_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", 0)
            except:
                pass
        
        if(dataset_slowfast=="kinetics-train" or dataset_slowfast=="kinetics-val"):
            
            key_frame          = phalp_file.split(dataset_slowfast + '_')[-1][:-4]
            class_name         = kinetics_annotations[key_frame][0]
            phalp_             = joblib.load(phalp_path + phalp_file)
            all_frames         = list(phalp_.keys())
            tracks_dict        = get_tracks(phalp_)
            
            for track_id in tracks_dict.keys():
                video_created  = 0
                # check if the file exists
                if os.path.exists(fast_path + key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    continue
                
                # if os.path.exists(save_path + key_frame + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                #     continue
                
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
                NUM_STEPS        = 6
                NUM_FRAMES       = seq_length
                list_iter        = list(range(len(list_of_frames)//NUM_STEPS + 1))
                for t_, time_stamp in enumerate(list_iter):    
                    print("Generating predictions for time stamp: {} sec".format(time_stamp))
                  
                  
                    start_      = time_stamp * NUM_STEPS
                    end_        = (time_stamp + 1) * NUM_STEPS if (time_stamp + 1) * NUM_STEPS < len(list_of_frames) else len(list_of_frames)
                    time_stamp_ = list_of_frames[start_:end_]
                    if(len(time_stamp_)==0): continue

                    mid_        = (start_ + end_)//2
                    mid_frame   = list_of_frames[mid_]
                    mid_bbox    = list_of_bbox[mid_]
                    mid_id      = tracks_dict[track_id][mid_frame]['fid']
                    start_id    = mid_id - NUM_FRAMES//2 if mid_id - NUM_FRAMES//2 > 0 else 0
                    end_id      = mid_id + NUM_FRAMES//2 if mid_id + NUM_FRAMES//2 < len(all_frames) else len(all_frames)
                        
                        
                    # import ipdb; ipdab.set_trace()
                    # get the frames for mvit
                    list_of_all_frames = []
                    list_of_all_paths  = []
                    for i in range(-NUM_FRAMES//2,NUM_FRAMES//2):
                        frame_id = int(mid_frame[:-4])
                        if frame_id+i<=0:      frame_id = 1
                        elif frame_id+i>300: frame_id = 300
                        else:                  frame_id = frame_id+i
                        frame_name_all = '%05d.jpg'%(frame_id,)
                        frame_key_all  = class_name + "/" + key_frame + "/" + '%05d.jpg'%(frame_id,)
                        # import pdb; pdb.set_trace()
                        if frame_name_all not in list_of_all_frames:
                            if(os.path.exists(root_path + "/" + frame_key_all)):
                                list_of_all_frames.append(frame_name_all)
                                list_of_all_paths.append(root_path + "/" + frame_key_all)
                                    
                                    
                    list_of_paths_all = list([phalp_[i]['frame_path'] for i in all_frames[start_id:end_id]])
                    list_of_paths_window = list([phalp_[i]['frame_path'] for i in time_stamp_])
                    predicted_boxes   = mid_bbox.reshape(1, 4)
                    predicted_boxes   = np.concatenate([predicted_boxes[:, :2], predicted_boxes[:, :2] + predicted_boxes[:, 2:4]], 1)
                    slowfast_, clip_  = generate_pseudo_labels(t_, list_of_paths_all, list_of_paths_window, predicted_boxes, make_video, args.add_clip)
                    for i_, frame_ in enumerate(time_stamp_):
                        tracks_dict[track_id][frame_]['action_emb_p']      = slowfast_[0].cpu().float().numpy()                        
                        tracks_dict[track_id][frame_]['action_features']   = slowfast_[2].cpu().float().numpy()                        
                        if(args.add_clip): tracks_dict[track_id][frame_]['clip_features']     = clip_[1][i_].cpu().numpy()
                    
                    if(video_created==0 and make_video):
                        imgs_           = slowfast_[1]
                        height, width   = imgs_[0].shape[0], imgs_[0].shape[1]
                        video_save_path = save_path_video +  phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".mp4"
                        video_file      = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(width,height))
                        video_created   = 1

                    if(make_video and video_created==1):
                        for img in imgs_:
                            video_file.write(img)
 
                    print(t_, len(list_of_paths_all), len(predicted_boxes))
                    
                if(make_video): video_file.release()
                if(make_video): print('Predictions are saved to the video file: ', video_save_path)
                # joblib.dump(tracks_dict[track_id], save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", compress=3)
                create_fast_tracklets(copy.deepcopy(tracks_dict[track_id]), fast_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", 0)
        
    
        if(dataset_slowfast=="ava-train" or dataset_slowfast=="ava-val" or dataset_slowfast=="avaK-train"):
            
            add_psedo_labels   = True
            NUM_STEPS          = 2
            NUM_FRAMES         = seq_length
            phalp_             = joblib.load(phalp_path + phalp_file)
            all_frames         = list(phalp_.keys())
            tracks_dict        = get_tracks(phalp_)
            
            if(dataset_slowfast=="ava-train"):
                key_frame_id       = phalp_file.split('ava-train_')[-1][:-11]
                key_frame_         = int(phalp_file.split('ava-train_')[-1][-10:-4])
                # key_frames         = [key_frame_id + '_%06d.jpg'%(key_frame_+i*30,) for i in range(-2,3)]
                key_frames         = [key_frame_id + '_%06d.jpg'%(key_frame_+i*30,) for i in range(1)]
                key_frames_num     = [key_frame_+i*30 for i in range(1)]
                
            if(dataset_slowfast=="ava-val"):
                key_frame_id       = phalp_file.split('ava-val_')[-1][:-11]
                key_frame_         = int(phalp_file.split('ava-val_')[-1][-10:-4])
                key_frames         = [key_frame_id + '_%06d.jpg'%(key_frame_+i*30,) for i in range(1)]
                key_frames_num     = [key_frame_+i*30 for i in range(1)]
                
            if(dataset_slowfast=="avaK-train"):
                key_frame_class    = phalp_file.split('avaK-train_')[-1].split('_._')[0]
                key_frame_id       = phalp_file.split('avaK-train_')[-1].split('_._')[1]
                key_frame_         = int(phalp_file.split('avaK-train_')[-1].split('_._')[-1][:-4])
                key_frames         = ['%05d.jpg'%(key_frame_+i*30,) for i in range(1)]
                key_frames_num     = [key_frame_+i*30 for i in range(1)]
            

            for track_id in tracks_dict.keys():
                save_track         = True
                video_created      = 0
                
                # # # check if the file exists
                if os.path.exists(fast_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    print("File already exists: ", fast_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl")
                    continue
                if os.path.exists(save_path_x + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl"):
                    print("File already exists in x : ", save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl")
                    continue
                
                list_of_frames   = np.array(list(tracks_dict[track_id].keys()))
                list_of_paths    = np.array([tracks_dict[track_id][i]['frame_path'] for i in list_of_frames ])
                list_of_bbox     = np.array([tracks_dict[track_id][i]['bbox'] for i in list_of_frames ])
                list_of_time     = np.array([tracks_dict[track_id][i]['time'] for i in list_of_frames ])
                tracked_time     = list_of_time == 0

                list_of_frames   = list_of_frames[tracked_time]
                list_of_paths    = list_of_paths[tracked_time]
                list_of_bbox     = list_of_bbox[tracked_time]
                
                if(add_psedo_labels):
                    list_iter       = list(range(len(list_of_frames)//NUM_STEPS + 1))
                    for t_, time_stamp in enumerate(list_iter):    
                        print("Generating predictions for time stamp: {} sec".format(time_stamp))
                        
                        start_      = time_stamp * NUM_STEPS
                        end_        = (time_stamp + 1) * NUM_STEPS if (time_stamp + 1) * NUM_STEPS < len(list_of_frames) else len(list_of_frames)
                        time_stamp_ = list_of_frames[start_:end_]
                        if(len(time_stamp_)==0): continue

                        mid_        = (start_ + end_)//2
                        mid_frame   = list_of_frames[mid_]
                        mid_bbox    = list_of_bbox[mid_]
                        mid_id      = tracks_dict[track_id][mid_frame]['fid']
                        start_id    = mid_id - NUM_FRAMES//2 if mid_id - NUM_FRAMES//2 > 0 else 0
                        end_id      = mid_id + NUM_FRAMES//2 if mid_id + NUM_FRAMES//2 < len(all_frames) else len(all_frames)
                        
                        # get the frames for mvit
                        list_of_all_frames = []
                        list_of_all_paths  = []
                        for i in range(-NUM_FRAMES//2,NUM_FRAMES//2):
                            if(dataset_slowfast=="avaK-train"):
                                frame_id = int(mid_frame[:-4])
                                if frame_id+i<=0:      frame_id = 1
                                elif frame_id+i>300: frame_id = 300
                                else:                  frame_id = frame_id+i
                                frame_name_all = '%05d.jpg'%(frame_id,)
                                frame_key_all  = key_frame_class + "/" + key_frame_id + "/" + '%05d.jpg'%(frame_id,)
                                # import pdb; pdb.set_trace()
                                if frame_name_all not in list_of_all_frames:
                                    if(os.path.exists(root_path + "/" + frame_key_all)):
                                        list_of_all_frames.append(frame_name_all)
                                        list_of_all_paths.append(root_path + "/" + frame_key_all)
                            else:
                                frame_id = int(mid_frame[12:18])
                                if frame_id+i<=0:      frame_id = 1
                                elif frame_id+i>27030: frame_id = 27030
                                else:                  frame_id = frame_id+i

                                frame_name_all = key_frame_id + "_" + '%06d.jpg'%(frame_id,)
                                frame_key_all  = key_frame_id + "/" + key_frame_id + "_" + '%06d.jpg'%(frame_id,)
                                if frame_name_all not in list_of_all_frames:
                                    list_of_all_frames.append(frame_name_all)
                                    list_of_all_paths.append(root_path + "/" + frame_key_all)


                        # list_of_paths_all = list([phalp_[i]['frame_path'] for i in all_frames[start_id:end_id]])
                        list_of_paths_window = list([phalp_[i]['frame_path'] for i in time_stamp_])
                        predicted_boxes   = mid_bbox.reshape(1, 4)
                        predicted_boxes   = np.concatenate([predicted_boxes[:, :2], predicted_boxes[:, :2] + predicted_boxes[:, 2:4]], 1)
                        slowfast_, clip_  = generate_pseudo_labels(t_, list_of_all_paths, list_of_paths_window, predicted_boxes, make_video, args.add_clip)
                        for i_, frame_ in enumerate(time_stamp_):
                            tracks_dict[track_id][frame_]['action_emb_p']      = slowfast_[0].cpu().float().numpy()      
                            tracks_dict[track_id][frame_]['action_features']   = slowfast_[2].cpu().float().numpy()                       
                            if(args.add_clip): tracks_dict[track_id][frame_]['clip_features']     = clip_[1][i_].cpu().numpy()
                            # if(args.add_mvp): tracks_dict[track_id][frame_]['mvp_features']     = mvp_[1][i_].cpu().numpy()
                        
                        if(video_created==0 and make_video):
                            imgs_           = slowfast_[1]
                            height, width   = imgs_[0].shape[0], imgs_[0].shape[1]
                            video_save_path = save_path_video +  phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".mp4"
                            video_file      = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(width,height))
                            video_created   = 1

                        if(make_video and video_created==1):
                            for img in imgs_:
                                video_file.write(img)
                    if(make_video): 
                        video_file.release()
                        print('Predictions are saved to the video file: ', video_save_path)       


                # attach gt labels
                video_created      = 0
                frames_to_keep     = []
                t_                 = 0
                
                for tmp_key in tracks_dict[track_id].keys():
                    tracks_dict[track_id][tmp_key]['is_keyframe'] = 0
                    
                
                for key_frame, key_frame_num in zip(key_frames, key_frames_num):
                    t_ += 1
                    if(key_frame not in tracks_dict[track_id].keys()): 
                        # if(dataset_slowfast=="ava-val"):
                        save_track = False
                        continue
                    
                    tracks_dict[track_id][key_frame]['is_keyframe'] = 1
                    
                    if(tracks_dict[track_id][key_frame]['has_detection']):
                        
                        if(key_frame[:11]+"/"+key_frame in ava_annotations.keys()):
                            frame_annot    = np.array(ava_annotations[key_frame[:11]+"/"+key_frame])
                            bbox_annot     = frame_annot[:, 2:6].astype(np.float32)
                            bbox_detect    = tracks_dict[track_id][key_frame]['bbox'].reshape(1,4)
                            size_          = tracks_dict[track_id][key_frame]['size']
                            bbox_detect_   = np.concatenate([bbox_detect[:, :2], bbox_detect[:, :2] + bbox_detect[:, 2:4]], 1)
                            bbox_detect_norm = np.array([[bbox_detect_[0, 0]/size_[1], bbox_detect_[0, 1]/size_[0], bbox_detect_[0, 2]/size_[1], bbox_detect_[0, 3]/size_[0]]])
                            iou_overlap    = iou(bbox_detect_norm, bbox_annot)
                            ab             = iou_overlap[0]>0.5
                            frame_annot_   = frame_annot[ab, :]
                        else:
                            frame_annot_   = np.array([])

                        # list of all frames of the VIDEO
                        list_of_all_frames = []
                        list_of_all_paths  = []
                        for i in range(-NUM_FRAMES//2,NUM_FRAMES//2):
                            if(dataset_slowfast=="avaK-train"):
                                frame_id = key_frame_num
                                if frame_id+i<=0:      frame_id = 1
                                elif frame_id+i>300: frame_id = 300
                                else:                  frame_id = frame_id+i
                                frame_name_all = '%05d.jpg'%(frame_id,)
                                frame_key_all  = key_frame_class + "/" + key_frame_id + "/" + '%05d.jpg'%(frame_id,)
                                if frame_name_all not in list_of_all_frames:
                                    if(os.path.exists(root_path + "/" + frame_key_all)):
                                        list_of_all_frames.append(frame_name_all)
                                        list_of_all_paths.append(root_path + "/" + frame_key_all)
                            else:
                                frame_id = key_frame_num
                                if frame_id+i<=0:      frame_id = 1
                                elif frame_id+i>27030: frame_id = 27030
                                else:                  frame_id = frame_id+i

                                frame_name_all = key_frame_id + "_" + '%06d.jpg'%(frame_id,)
                                frame_key_all  = key_frame_id + "/" + key_frame_id + "_" + '%06d.jpg'%(frame_id,)
                                if frame_name_all not in list_of_all_frames:
                                    list_of_all_frames.append(frame_name_all)
                                    list_of_all_paths.append(root_path + "/" + frame_key_all)

                        
                        # list of all frames of the TRACK to be updated
                        list_of_track_frames = list_of_frames
                        loc_all              = np.where(np.array(list_of_track_frames)==key_frame)[0][0]
                        mid_bbox             = list_of_bbox[loc_all]
                        assert key_frame == list_of_track_frames[loc_all]
                        start_         = loc_all - NUM_STEPS//2 if loc_all - NUM_STEPS//2 > 0 else 0
                        end_           = loc_all + NUM_STEPS//2 if loc_all + NUM_STEPS//2 < len(list_of_track_frames) else len(list_of_track_frames)
                        time_stamp_    = list_of_frames[start_:end_]
                        if(len(time_stamp_)==0): continue
                        
                        list_of_paths_window = list([phalp_[i]['frame_path'] for i in time_stamp_])
                        predicted_boxes   = mid_bbox.reshape(1, 4)
                        predicted_boxes   = np.concatenate([predicted_boxes[:, :2], predicted_boxes[:, :2] + predicted_boxes[:, 2:4]], 1)
                        
                        slowfast_, clip_  = generate_pseudo_labels(t_, list_of_all_paths, list_of_paths_window, predicted_boxes, make_video, args.add_clip)
                        for i_, frame_ in enumerate(time_stamp_):
                            tracks_dict[track_id][frame_]['action_emb_p']      = slowfast_[0].cpu().float().numpy()        
                            tracks_dict[track_id][frame_]['action_features']   = slowfast_[2].cpu().float().numpy()                     
                            if(args.add_clip): tracks_dict[track_id][frame_]['clip_features']     = clip_[1][i_].cpu().numpy()
                            # if(args.add_mvp): tracks_dict[track_id][frame_]['mvp_features']     = mvp_[1][i_].cpu().numpy()
                            if(len(frame_annot_)>0):
                                tracks_dict[track_id][frame_]['gt_annot']     = frame_annot_
                                tracks_dict[track_id][frame_]['gt_class']     = frame_annot_[:, 6].astype(np.int32)
                                annotated += 1.0/len(time_stamp_)
                            
                            if(tracks_dict[track_id][key_frame]['annot'] is not None and len(tracks_dict[track_id][key_frame]['annot'][0])>0):
                                tracks_dict[track_id][frame_]['gt_annot_2']     = np.array(tracks_dict[track_id][key_frame]['annot'], dtype=np.int32)
                                tracks_dict[track_id][frame_]['gt_class_2']     = np.array(tracks_dict[track_id][key_frame]['annot'], dtype=np.int32)
                                
                        # if(video_created==0):
                        #     imgs_ = [cv2.imread(i) for i in list_of_paths_window ]
                        #     height, width   = imgs_[0].shape[0], imgs_[0].shape[1]
                        #     video_save_path = save_path_video +  phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".mp4"
                        #     video_file      = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=(width,height))
                        #     video_created   = 1
                        #     # if(len(frame_annot_)>0):
                        #     if(tracks_dict[track_id][key_frame]['annot'] is not None and len(tracks_dict[track_id][key_frame]['annot'][0])>0):
                        #         # tracks_dict[track_id][key_frame]['annot'][0]
                        #         # import ipdb; ipdb.set_trace()
                        #         x1 = int(mid_bbox[0])
                        #         y1 = int(mid_bbox[1])
                        #         x2 = int(mid_bbox[0] + mid_bbox[2])
                        #         y2 = int(mid_bbox[1] + mid_bbox[3])
                        #         cv2.rectangle(imgs_[len(imgs_)//2], (x1, y1), (x2, y2), (255,0,0), 4)
                        #         str_ = "_".join([str(i) for i in tracks_dict[track_id][key_frame]['annot']])
                        #         cv2.putText(imgs_[len(imgs_)//2], str(str_), (x1+20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        #         # for j_, annot in enumerate(frame_annot_):
                        #         #     x1 = int(float(annot[2])*width)
                        #         #     y1 = int(float(annot[3])*height)
                        #         #     x2 = int(float(annot[4])*width)
                        #         #     y2 = int(float(annot[5])*height)
                        #         #     cv2.rectangle(imgs_[len(imgs_)//2], (x1, y1), (x2, y2), (255,0,0), 4)
                        #         #     # add text
                        #         #     cv2.putText(imgs_[len(imgs_)//2], str(annot[-2]), (x1+20*j_, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        #         # import ipdb; ipdb.set_trace()
                        #         # pass

                        # if(video_created==1):
                        #     for img in imgs_:
                        #         video_file.write(img)
                    
                    else:
                        # if(dataset_slowfast=="ava-val"):
                        save_track = False 

                if(video_created==1): 
                    video_file.release()
                    print('Predictions are saved to the video file: ', video_save_path)       

                if(save_track): 
                    create_fast_tracklets(copy.deepcopy(tracks_dict[track_id]), fast_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", 0)
                    # joblib.dump(tracks_dict[track_id], save_path + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", compress=3)
                else:
                    joblib.dump([], save_path_x + phalp_file[:-4] + "_" + str(track_id) + "_" + str(len(tracks_dict[track_id].keys())) + ".pkl", compress=3)
                    