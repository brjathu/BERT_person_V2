import argparse
import copy
import os

import joblib
import numpy as np
# from optimization.main import optimize_track_pose
from tqdm import tqdm

# from src.utils.rotation_conversions import *
from src.utils.utils import task_divider
from src.utils.vit_timm import *

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--batch_id', type=int, default=-1, help='num of workers to use')
    parser.add_argument('--num_of_process', type=int, default=100, help='num of workers to use')
    parser.add_argument('--dataset_slowfast', type=str, default="kinetics-train", help='num of workers to use')
    parser.add_argument('--add_optimization', type=int)

    opt = parser.parse_args()
    return opt


def edit_features(data, save_path):
    
    # import ipdb; ipdb.set_trace()
    
    bs = 64
    frame_names = data['frame_name']
    data["mae_emb"] = np.zeros((len(frame_names), 1, 768))*0.0
    
    for i in range(0, len(frame_names), bs):
        frame_names_batch = frame_names[i:i+bs]
        imgs = []
        for j in range(len(frame_names_batch)):
            frame_name = frame_names_batch[j]
            img_ = cv2.imread(frame_name)
            img_ = process_image(img_)
            imgs.append(img_)
        imgs = np.array(imgs)
        imgs = torch.from_numpy(imgs).float().cuda()
        with torch.no_grad():
            feat = vit_hands.forward(imgs)
        feat = feat.cpu().numpy()
        for j in range(len(frame_names_batch)):
            data["mae_emb"][i+j] = feat[j]
    
    data.pop("mvit_emb")
    joblib.dump(data, save_path)            
    


if __name__ == '__main__':

    # # setup video model
    device           = 'cuda'
    args             = parse_option()

    if(args.dataset_slowfast == "ava-train"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v21_1/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.500_ava_train/results_slowfast_v23_1/"
    elif(args.dataset_slowfast == "ava-val"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v21_1/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.501_ava_val/results_slowfast_v23_2/"
    elif(args.dataset_slowfast == "avaK-train"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.500_avaK_train/results_slowfast_v21_1/"
    elif(args.dataset_slowfast == "kinetics-train"):
        root_base = "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v21_1/"
        root_new  = "/checkpoint/jathushan/TENET/out/Videos_v4.400_kinetics_train/results_slowfast_v23_1/"
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
    
    vit_hands, _ = vit_b16("/private/home/jathushan/3D/mvp/mvp-b.pth")
    vit_hands.eval()
    vit_hands.to('cuda')
    
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
        
        edit_features(data, save_path)
        
        