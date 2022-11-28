import torch
import numpy as np
from src.utils.one_euro_filter import OneEuroFilter
import src.utils.rotation_conversions as geometry


def smooth_pose(pred_smpl_params, min_cutoff=0.1, beta=0.99):

    global_orientation = pred_smpl_params['global_orient'].cpu().numpy()
    pose               = pred_smpl_params['body_pose'].cpu().numpy()
    betas              = pred_smpl_params['betas']

    rotmats            = torch.from_numpy(np.concatenate([global_orientation, pose], 1)).cuda()
    axis_angles        = geometry.matrix_to_axis_angle(rotmats.cpu()).numpy()

    one_euro_filter = OneEuroFilter(np.zeros_like(axis_angles[0]), axis_angles[0],  min_cutoff=min_cutoff, beta=beta)

    pred_pose_hat   = np.zeros_like(axis_angles)

    pred_pose_hat[0] = axis_angles[0]

    for idx, pose in enumerate(axis_angles[1:]):
        idx += 1
        t    = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose

    rotmats_     = geometry.axis_angle_to_matrix(torch.from_numpy(pred_pose_hat)).cuda()

    pred_smpl_params = {'global_orient': rotmats[:, :1, :, :], 'body_pose': rotmats_[:, 1:, :, :],'betas': betas}   

    return pred_smpl_params


def smooth_tran(camera, min_cutoff=0.1, beta=0.99):

    camera          = camera.cpu().numpy()
    one_euro_filter = OneEuroFilter(np.zeros_like(camera[0]), camera[0],  min_cutoff=min_cutoff, beta=beta)
    camera_hat      = np.zeros_like(camera)
    camera_hat[0]   = camera[0]

    for idx, cam in enumerate(camera[1:]):
        idx += 1
        t    = np.ones_like(cam) * idx
        cam  = one_euro_filter(t, cam)
        camera_hat[idx] = cam

    camera_hat = torch.from_numpy(camera_hat).cuda()


    return camera_hat