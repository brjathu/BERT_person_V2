# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import json

# annot = np.load("../TENET/_DATA/kinetics400_train.npy")
# names = json.load(open("../slowfast/kinetics_names.json", "r"))
# annot_dict = {}
# for an in annot:
#     if("_" in an[0]):
#         key_ = "\"" + an[0].replace("_", " ") + "\""
#     else:
#         key_ = an[0].replace("_", " ")        
#     annot_dict[an[1]] = [an[0], names[key_] ]
# joblib.dump(annot_dict, "data/kinetics_annot_train.pkl")



# annot = np.load("../TENET/_DATA/kinetics400_val.npy")
# names = json.load(open("../slowfast/kinetics_names.json", "r"))
# annot_dict = {}
# for an in annot:
#     if("_" in an[0]):
#         key_ = "\"" + an[0].replace("_", " ") + "\""
#     else:
#         key_ = an[0].replace("_", " ")
        
#     annot_dict[an[1]] = [an[0], names[key_] ]
# joblib.dump(annot_dict, "data/kinetics_annot_val.pkl")






# from bdb import Breakpoint
# import numpy as np
# import os
# import csv
# from tqdm import tqdm
# import joblib

# classes   = os.listdir("/datasets01/AVA/080720/frames/")
# csv_file  = csv.reader(open("/private/home/jathushan/3D/BERT_person/_DATA/ava_kinetics_v1_0/ava_train_v2.2.csv"))

# list_of_videos = []
# list_of_annot  = {}
# for data in tqdm(csv_file):
#     frame_name = data[0] + "/" + data[0] + '_%06d.jpg'%((int(data[1])-900)*30,)
#     if(len(data)==8):
#         video_     = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], frame_name]
#         if(os.path.isfile("/datasets01/AVA/080720/frames/" + frame_name)):
#             list_of_videos.append(video_[-1])
#             list_of_annot.setdefault(frame_name, []).append(video_)

# list_of_videos = np.array(list_of_videos)
# list_of_videos = np.sort(list_of_videos)
# print(len(list_of_videos))
# print(len(np.unique(list_of_videos)))
# joblib.dump(list_of_annot, "data/ava_train_annot.pkl")


# classes   = os.listdir("/datasets01/AVA/080720/frames/")
# csv_file  = csv.reader(open("/private/home/jathushan/3D/BERT_person/_DATA/ava_kinetics_v1_0/ava_val_v2.2.csv"))

# list_of_videos = []
# list_of_annot  = {}
# for data in tqdm(csv_file):
#     frame_name = data[0] + "/" + data[0] + '_%06d.jpg'%((int(data[1])-900)*30,)
#     if(len(data)==8):
#         video_     = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], frame_name]
#         if(os.path.isfile("/datasets01/AVA/080720/frames/" + frame_name)):
#             list_of_videos.append(video_[-1])
#             list_of_annot.setdefault(frame_name, []).append(video_)

# list_of_videos = np.array(list_of_videos)
# list_of_videos = np.sort(list_of_videos)
# print(len(list_of_videos))
# print(len(np.unique(list_of_videos)))
# joblib.dump(list_of_annot, "data/ava_val_annot.pkl")







# rendering
# ############################################################################################################
# ############################################################################################################
# ############################################################################################################
# import os
# import joblib
# import numpy as np
# from tqdm import tqdm

# mesh = {}
# root_dir_ = "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6/"
# list_of_files = np.load("data/_checkpoint_jathushan_TENET_out_Videos_v4.401_ava_val_results_slowfast_v6_.npy")
# for file_ in tqdm(list_of_files):
#     id_       = file_.split("ava-val_")[1][:11]
#     key_      = file_.split("ava-val_")[1][12:18]
#     key_frame = id_ + "_" + key_ + ".jpg"      
#     ab = joblib.load(root_dir_ + file_)
#     if(key_frame in ab.keys()):
#         data = ab[key_frame]
#         try:
#             get_class = data['gt_class']
#             for gt_ in get_class:
#                 mesh.setdefault(gt_, []).append(file_)
#         except:
#             pass
# joblib.dump(mesh, "data/mesh.pkl")


# root_dir = "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6///"
# npy_name = "data/"+"_".join(root_dir.split("/"))+".npy"

# list_to_render = []
# for class_ in mesh.keys():
#     files_ = mesh[class_]
#     np.random.shuffle(files_)
#     for file_ in files_[:50]:
#         if(file_ not in list_to_render):
#             list_to_render.append(file_)
# np.save(npy_name, list_to_render)
# ############################################################################################################
# ############################################################################################################
# ############################################################################################################        




# import os
# import joblib
# import numpy as np
# from tqdm import tqdm

# mesh = {}
# # root_dir_ = "/checkpoint/jathushan/TENET/out/Videos_v4.401_ava_val/results_slowfast_v6/"
# # list_of_files = np.load("data/_checkpoint_jathushan_TENET_out_Videos_v4.401_ava_val_results_slowfast_v6_.npy")
# root_dir_ = "/checkpoint/jathushan/TENET/out/Videos_v4.400_ava_train/results_slowfast_v6/"
# list_of_files = np.load("data/_checkpoint_jathushan_TENET_out_Videos_v4.400_ava_train_results_slowfast_v6_.npy")
# pbar = tqdm(list_of_files)
# for file_ in pbar:
#     id_       = file_.split("ava-train_")[1][:11]
#     key_      = file_.split("ava-train_")[1][12:18]
#     frame_id  = "%04d"%(int(key_)//30 + 900,)
#     key_frame = id_ + "_" + key_ + ".jpg"      
#     ab = joblib.load(root_dir_ + file_)
#     pbar.set_description("Processing " +  str(len(list(mesh.keys()))) + " classes")
#     if(key_frame in ab.keys()):
#         data = ab[key_frame]
#         try:
#             get_annot = data['gt_annot']
#             get_class = data['gt_class']
#             bbox      = data['bbox']
#             size      = data['size']
#             bbox_     = np.concatenate([bbox[:2], bbox[:2] + bbox[2:]])
#             bb_n = [bbox_[0]/size[1], bbox_[1]/size[0], bbox_[2]/size[1], bbox_[3]/size[0]]
#             key_annot = id_ + "_" + frame_id + "_" + str(np.round(bb_n[0], 2)) + "_" + str(np.round(bb_n[1], 2)) + "_" + str(np.round(bb_n[2], 2)) + "_" + str(np.round(bb_n[3], 2))
#             # print(key_annot)
#             mesh[key_annot] = [get_annot, get_class, bbox, bbox_, bb_n, size]
#     #     # for gt_ in get_class:
#     #     #     mesh.setdefault(gt_, []).append(file_)
#     #     break
#         except:
#             # import pdb; pdb.set_trace()
#             pass
#         # break
#     # break
# joblib.dump(mesh, "data/mesh_3.pkl")



# plt.xlim(-1000,220000)
# for run in runs_all:
#     path  = "logs/" + run + "/tensorboard/lightning_logs/version_0/"
#     try:
#         files = np.sort(list([i for i in os.listdir(path) if "events" in i]))
#         for file_ in files:
#             try:
#                 ab    = parse_tensorboard(path + file_, [key_])
#                 if(len(ab[key_]["value"])>0):
#                     if(run in skip_runs): continue
#                     # plt.plot(list(range(1,len(ab[key_]["value"])+1)), ab[key_]["value"], label=run, alpha=0.8, linewidth=2, marker='o')
#                     plt.plot(ab[key_]["step"], ab[key_]["value"], label=run, alpha=0.8, linewidth=2, marker='o')
#                     # add text at the maximum value
#                     if(ab[key_]["value"].iloc[-1]>37):
#                         plt.text(ab[key_]["step"].iloc[-1], ab[key_]["value"].iloc[-1], run, fontsize=10)
#             except:
#                 pass
#     except:
#         pass

# plt.xlim(-1,5)
# plt.ylim(40,42)








# read csv file
import csv
import os
from tqdm import tqdm
import joblib
from src.ActivityNet.Evaluation.get_ava_performance import run_evaluation
import numpy as np

def read_predictions(csv_path):
    a3 = open(csv_path, "r")
    res_ = {}
    i = 0
    for row in tqdm(csv.reader(a3)):
        # res_.setdefault(row[0]+"_"+row[1]+"_"+str(np.round(float(row[2]),2))+"_"+str(np.round(float(row[3]),2))+"_"+str(np.round(float(row[4]),2))+"_"+str(np.round(float(row[5]),2)), {}).setdefault(row[-2], row)
        res_.setdefault(row[0]+"_"+row[1]+"_"+row[2]+"_"+row[3]+"_"+row[4]+"_"+row[5], {}).setdefault(row[-2], row)
    return res_



list_of_csv_files = [
    # "logs/1802/ava_val.csv",
    # "logs/1813/ava_val_best.csv",
    # "logs/1813a/ava_val_best.csv",
    # "logs/1813b/ava_val_best.csv",
    # "logs/1812/ava_val_best.csv",
    # "logs/1740/ava_val.csv",
    # "logs/1741/ava_val.csv",
    # "logs/1720_f4/ava_val.csv",
    # "logs/13005/ava_val.csv",
    
    
    # "logs/1962b/ava_val_best.csv",
    # "logs/1962c/ava_val_best.csv",
    # "logs/1961a/ava_val_best.csv",
    
    
    # "logs/1972e/0/ava_val_best.csv",
    # "logs/1972d/0/ava_val_best.csv",
    # "logs/1970d/0/ava_val_best.csv",
    # "logs/1970b/ava_val_best.csv",
    
    "/private/home/jathushan/3D/BERT_person_hydra/logs//1990_4p3_test1/0/ava_val_best.csv",
    "/private/home/jathushan/3D/BERT_person_hydra/logs//1990_4p3_test2/0/ava_val_best.csv",
    # "/private/home/jathushan/3D/BERT_person_hydra/logs/19612h4/0/ava_val_best.csv",
]

res_ = [read_predictions(csv_path) for csv_path in list_of_csv_files]
# annot_ = joblib.load("data/mesh_2.pkl")

f = open("_TMP/test.csv", 'w')
writer = csv.writer(f)
counter = 0
selected_videos = []
for key in tqdm(res_[0].keys()):
    
    data_1 = res_[0][key]
    for key_ in data_1.keys():
        row_1 = data_1[key_]
        pred  = []
        for i_ in range(len(res_)):
            try:
                pred.append(float(res_[i_][key][key_][-1]))
                if(row_1[6]=="20"):
                    if(float(res_[0][key][key_][-1])>float(res_[1][key][key_][-1])):
                        selected_videos.append(key)
            except:
                pass
        
        # if(np.mean(pred)<0.5):
        #     pred_ = np.min(pred)
        # else:
        #     pred_ = np.max(pred)
        
        # if(np.mean(pred)>0.95):
        #     pred_ = 1
        # elif(np.mean(pred)<0.05):
        #     pred_ = 0
        # else:
        
        pred_ = np.mean(pred)
            
        result = [row_1[0], row_1[1], row_1[2], row_1[3], row_1[4], row_1[5], row_1[6], pred_]
        writer.writerow(result)
f.close()


a1 = open("/datasets01/AVA/080720/frame_list/ava_action_list_v2.2_for_activitynet_2019.pbtxt", "r")
a2 = open("/datasets01/AVA/080720/frame_list/ava_val_v2.2.csv", "r")
a3 = open("_TMP/test.csv", "r")
aaaa = run_evaluation(a1, a2, a3)
print("mAP : ", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
print("mAP : " + str(aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
print("mAP : " + str(aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
print("val/mAP", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
joblib.dump(aaaa, "_TMP/en_1806.pkl")