import joblib
import cv2
import csv
from tqdm import tqdm
import os
import numpy as np
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from src.ActivityNet.Evaluation.get_ava_performance import run_evaluation

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import joblib
import pprint
import sys

def compute_map(slowfast_path, th_conf=0.7, scale=1.0):
    exp_name       = slowfast_path.split("/")[-3] + "_" + str(th_conf) + "_" + str(scale)
    slowfast_files = [i for i in os.listdir(slowfast_path) if i.endswith(".pkl")]
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/ava_action_list.pbtxt')
    os.system('rm _TMP/'+exp_name+'.csv')
    f = open('_TMP/'+exp_name+'.csv', 'w')
    writer = csv.writer(f)

    counter = 0
    for slowfast_file in tqdm(slowfast_files):
        video_id  = slowfast_file.split("ava-val_")[1][:11]
        key_frame = slowfast_file.split("ava-val_")[1][12:18]
        frame_id  = "%04d"%(int(key_frame)//30 + 900,)
        data      = joblib.load(slowfast_path + slowfast_file)
        h, w      = data[-2][0][0], data[-2][0][1]
        conf      = data[-1]
        if(conf>=th_conf):
            for i in range(len(data[2])):
                x1, y1, x2, y2 = data[-3][i][0], data[-3][i][1], data[-3][i][2], data[-3][i][3]
                cx = (x1+x2)/2
                cy = (y1+y2)/2
                wx = (x2-x1)
                wy = (y2-y1)
                x1 = cx - scale*wx/2
                x2 = cx + scale*wx/2
                y1 = cy - scale*wy/2
                y2 = cy + scale*wy/2
                pred  = data[1][i]
                pred_ = np.argsort(pred)[::-1]
                conf  = pred[pred_]
                loc_  = conf>-1
                pred_ = pred_[loc_]
                conf  = conf[loc_]
                for j in range(len(pred_)):
                    if(pred_[j]!=0 and pred_[j] in allowed_class_ids):
                        result = [video_id, frame_id, x1/w, y1/h, x2/w, y2/h, pred_[j], conf[j]]
                        writer.writerow(result)
                counter += 1
    print("number of bbox: ", counter)
    f.close()

    a1 = open("/datasets01/AVA/080720/frame_list/ava_action_list_v2.2_for_activitynet_2019.pbtxt", "r")
    a2 = open("/datasets01/AVA/080720/frame_list/ava_val_v2.2.csv", "r")
    a3 = open('_TMP/'+exp_name+'.csv', "r")
    aaaa = run_evaluation(a1, a2, a3)
    # pprint.pprint(aaaa[0], indent=2)
    print(exp_name, " mAP : ", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
    joblib.dump(aaaa, "_TMP/"+exp_name+".pkl")



list_of_exp = [

    ["logs/1000_test/slowfast/", 0.70, 1],

]


    
# for exp in list_of_exp:
#     try:
#         compute_map(exp[0], exp[1], exp[2])
#     except:
#         pass



if __name__=="__main__":
    compute_map(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
    pass