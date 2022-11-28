

# #!/usr/bin/env python
# # coding: utf-8
# #
# # Author:   Kazuto Nakashima
# # URL:      https://github.com/kazuto1011
# # Created:  2017-08-16

# import argparse

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import make_blobs

# import numpy as np
# from tqdm import tqdm

# a1 = np.load("/private/home/jathushan/3D/BERT_person_hydra/logs/1990_3b_test3/0/svm_vectors.npy", allow_pickle=True)

# X_train = []
# Y_train = []

# for i in tqdm(range(len(a1))):
#     labels = np.where(a1[i][1]==1)[0]
#     for lab in labels:
#         X_train.append(a1[i][0])
#         Y_train.append(int(lab))

# X_train = np.array(X_train)
# Y_train = np.array(Y_train)


# # from sklearn.multiclass import OneVsRestClassifier
# # from sklearn.svm import SVC

# # clf = OneVsRestClassifier(SVC(probability=True)).fit(X_train, Y_train)
# # a1 = clf.predict_proba(X_train[0].reshape(1, -1))
# # print(a1)


# from cuml.svm import SVC, LinearSVC
# from cuml.multiclass import OneVsRestClassifier
# import cupy as cp

# # clf = OneVsRestClassifier(LinearSVC(kernel='linear', C=10, cache_size=10000))
# X_train = cp.array(X_train, dtype=cp.float32)
# Y_train = cp.array(Y_train, dtype=cp.int32)
# clf = LinearSVC(loss='squared_hinge', penalty='l1', C=1, multi_class='ovr', max_iter=1000, probability=True)
# clf.fit(X_train, Y_train)
# a1 = clf.predict_proba(X_train[0].reshape(1, -1))
# print(a1)



from tqdm import tqdm
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from cuml.svm import SVC, LinearSVC
from cuml.multiclass import OneVsRestClassifier
import cupy as cp
import joblib
import csv
from src.ActivityNet.Evaluation.get_ava_performance import run_evaluation


AVA_VALID_FRAMES = range(902, 1799)
ava_valid_classes = joblib.load("data/ava_class_mappping.pkl")
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('data/ava_action_list.pbtxt')







a1 = np.load("/private/home/jathushan/3D/BERT_person_hydra/logs/1990_3b_test3/0/svm_vectors.npy", allow_pickle=True)
X_train = []
Y_train = []
for i in tqdm(range(len(a1))):
    labels = np.where(a1[i][1]==1)[0]
    for lab in labels:
        X_train.append(a1[i][0])
        Y_train.append(int(lab))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = cp.array(X_train, dtype=cp.float32)
Y_train = cp.array(Y_train, dtype=cp.int32)
# clf = LinearSVC(loss='squared_hinge', probability=True, C=100, multi_class='ovr', max_iter=1000)
clf = SVC(probability=True, C=10, multi_class='ovr', cache_size=10000, verbose=True)
# , penalty='l1', C=1, , max_iter=1000, 
clf.fit(X_train, Y_train)
joblib.dump(clf, "_TMP/svm_model.pkl")




# clf = joblib.load("_TMP/svm_model.pkl")
f = open("_TMP/ava.csv", 'w')
writer = csv.writer(f)
counter = 0
        
b1 = np.load("/private/home/jathushan/3D/BERT_person_hydra/logs/1990_3b_test4/0/svm_pred_vectors.npy", allow_pickle=True)
X_pred = []
X_name = []
for i in tqdm(range(len(b1))):
    data = b1[i]
    slowfast_file = b1[i][0]

    # make predctions with SVM classifier
    # import ipdb; ipdb.set_trace()
    x_ = cp.array(data[2], dtype=cp.float32).reshape(1, -1)
    svm_predictions = clf.predict_proba(x_).reshape(-1)
    svm_predictions = cp.asnumpy(svm_predictions) 
    svm_predictions = np.concatenate((np.zeros(1), svm_predictions))
    # import ipdb; ipdb.set_trace()
    
    
    video_id  = slowfast_file.split("ava-val_")[1][:11]
    key_frame = slowfast_file.split("ava-val_")[1][12:18]
    frame_id  = "%04d"%(int(key_frame)//30 + 900,)
    if(int(key_frame)//30+900 not in AVA_VALID_FRAMES): continue
            
    h, w = data[-2][0][0], data[-2][0][1]
    det_conf_ = data[-1]
    if(det_conf_ < 0.80): continue

    x1, y1, x2, y2 = data[-3][0][0], data[-3][0][1], data[-3][0][2], data[-3][0][3]
    
    pred  = svm_predictions
    pred_ = np.argsort(pred)[::-1]
    conf  = pred[pred_]
    loc_  = conf>-1
    pred_ = pred_[loc_]
    conf  = conf[loc_]

    for j in range(len(pred_)):
        # if(len(pred_)==60+1):
        pred_class = ava_valid_classes[pred_[j]]
        # else:
            # pred_class = pred_[j]
        if(pred_class!=0 and pred_class in allowed_class_ids):
            result = [video_id, frame_id, x1/w, y1/h, x2/w, y2/h, pred_class, conf[j]]
            writer.writerow(result)
    counter += 1
print("number of bounding boxes: ", counter)           
f.close()




a1 = open("/datasets01/AVA/080720/frame_list/ava_action_list_v2.2_for_activitynet_2019.pbtxt", "r")
a2 = open("/datasets01/AVA/080720/frame_list/ava_val_v2.2.csv", "r")
a3 = open("_TMP/ava.csv", "r")
aaaa = run_evaluation(a1, a2, a3)
print("mAP : ", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
print("mAP : " + str(aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
print("mAP : " + str(aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
print("val/mAP", aaaa[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
joblib.dump(aaaa, "_TMP/en_1806.pkl")