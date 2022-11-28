import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.utils.utils import batch_rodrigues, draw_text, get_colors, str2bool

warnings.filterwarnings("ignore")
from torchvision.ops.focal_loss import sigmoid_focal_loss

from src.utils.class_balanced_loss import \
    CBLoss  # FocalLoss #CB_loss, focal_loss

# from balanced_loss import Loss

# def get_mask_tokens(opt, tokens, tokens_no_track, idx, mask_type="random"):
#     BS, T, P, dim  = tokens.size()
#     idx_bert       = copy.deepcopy(idx)
#     tokens_in      = copy.deepcopy(tokens)

#     if(mask_type=="random"):
#         idx_bert       = idx_bert.view(-1)
#         idx_bert       = idx_bert.cpu().numpy()
#         idx_mask       = copy.deepcopy(idx_bert)*0
#         active_tokens  = np.where(idx_bert==1)[0]
#         num_tokens     = len(active_tokens)
#         masked_tokens  = np.random.choice(active_tokens, int(num_tokens*opt.mask_ratio), replace=opt.with_replacement)
#         idx_mask[masked_tokens] = 1
#         idx_mask       = np.reshape(idx_mask, (BS, T, P))
#         idx_mask       = torch.from_numpy(idx_mask).cuda().float()

#     if(mask_type=="random_clip"):
#         w_             = 5
#         idx_stop       = list(range(T//w_))
#         idx_bert       = idx_bert.view(-1)
#         idx_bert       = idx_bert.cpu().numpy()
#         idx_mask       = copy.deepcopy(idx_bert)*0
#         masked_tokens  = np.random.choice(idx_stop, int(len(idx_stop)*opt.mask_ratio), replace=opt.with_replacement)
#         masked_tokens  = [ list(range(i*w_, (i+1)*w_)) for i in masked_tokens]
#         masked_tokens  = [item for sublist in masked_tokens for item in sublist]
#         idx_mask[masked_tokens] = 1
#         idx_mask       = np.reshape(idx_mask, (BS, T, P))
#         idx_mask       = torch.from_numpy(idx_mask).cuda().float()

#     if(mask_type=="clip"):
#         idx_mask       = copy.deepcopy(idx_bert)*0
    
#         st = np.random.choice(opt.frame_length-12, (1, BS))[0]
#         for i_, st_ in enumerate(st): 
#             idx_mask[i_, st_:st_+12, :] = 1
#         masked_tokens  = [0]
    
#     if(mask_type=="zero"):
#         idx_mask       = copy.deepcopy(idx)*0
#         masked_tokens  = [0]
        

#     tokens          = tokens.view(BS, T*P, -1)
#     tokens_in       = tokens_in.view(BS, T*P, -1)
#     tokens_no_track = tokens_no_track.view(BS, T*P, -1)
#     idx             = idx.view(BS, T*P)
#     idx_mask        = idx_mask.view(BS, T*P)
                
#     return tokens, tokens_in, tokens_no_track, idx, idx_mask, len(masked_tokens)
    
def compute_loss(opt, output, smpl_output, output_data, input_data, train=True):
    loss_dict      = {
        "pose"   : 0, 
        "loca"   : 0, 
        "action" : 0, 
        "kp"     : 0, 
    }

    samples_per_class = np.load("data/class_sum.npy")
    cb_loss = CBLoss(loss_type="focal_loss", beta=0.999, fl_gamma=2, samples_per_class=samples_per_class, class_balanced=False)

    for fi in range(opt.num_smpl_heads):
        # groundtruth data
        gt_pose_shape        = output_data['pose_shape'][:, :, :, fi, :226]
        gt_location          = output_data['pose_shape'][:, :, :, fi, 226:229]
        # gt_action_ava        = output_data['action_label_ava'][:, :, :, fi, :].float()
        gt_action_kinetics   = output_data['action_label_kinetics'][:, :, :, fi, :].float()
        gt_has_detection     = output_data['has_detection'][:, :, :, fi, :]
        if(opt.ava.predict_valid):
            gt_action_ava    = output_data['action_label_ava'][:, :, :, fi, :opt.ava.num_valid_action_classes].float()
        else:
            gt_action_ava    = output_data['action_label_ava'][:, :, :, fi, :].float()
  
        # predicted data  
        BS, T, P, _          = gt_pose_shape.shape
        masked_detection     = input_data['mask_detection']

        # import pdb; pdb.set_trace()
        pred_global_orient   = smpl_output['pred_smpl_params'][fi]['global_orient'].view(BS, T, P, 9)
        pred_body_pose       = smpl_output['pred_smpl_params'][fi]['body_pose'].view(BS, T, P, 207)
        pred_betas           = smpl_output['pred_smpl_params'][fi]['betas'].view(BS, T, P, 10)
        pred_pose_shape      = torch.cat((pred_global_orient, pred_body_pose, pred_betas), dim=3)
        pred_location        = smpl_output['cam_t'][:, fi, :].view(BS, T, P, 3)
        pred_action_ava      = smpl_output['pred_actions_ava'][:, fi, :].view(BS, T, P, -1)
        pred_action_kinetics = smpl_output['pred_actions_kinetics']
            
        if(train): 
            if(opt.masked):
                loca_loss = torch.logical_and(gt_has_detection==1, masked_detection==1)
            else:
                loca_loss = gt_has_detection==1
        else:      
            loca_loss = gt_has_detection==1
        loca_loss = loca_loss[:, :, :, 0]

        loss_pose   = torch.tensor(0.0).cuda()
        loss_loca   = torch.tensor(0.0).cuda()
        loss_action = torch.tensor(0.0).cuda()
        loss_kp     = torch.tensor(0.0).cuda()

        if("pose_l1" in opt.loss_type):
            loss_pose   = ( pred_pose_shape[loca_loss] - gt_pose_shape[loca_loss] ).abs().sum()/(torch.sum(loca_loss)+1)

        if("pose_l2" in opt.loss_type):
            loss_pose   = ( pred_pose_shape[loca_loss] - gt_pose_shape[loca_loss] ).pow(2).sum()/(torch.sum(loca_loss)+1)
            loss_pose = torch.nan_to_num(loss_pose, nan=0.0, posinf=0.0, neginf=0.0)
            
        if("loca_l1" in opt.loss_type):
            loss_loca = ( pred_location[loca_loss] - gt_location[loca_loss] ).abs().sum()/(torch.sum(loca_loss)+1)
            loss_loca = torch.nan_to_num(loss_loca, nan=0.0, posinf=0.0, neginf=0.0)
            
        if("loca_l2" in opt.loss_type):
            loss_loca = ( pred_location[loca_loss] - gt_location[loca_loss] ).pow(2).sum()/(torch.sum(loca_loss)+1)

        if("kp_l1" in opt.loss_type):
            gt_kpc    = output_data['vitpose'][:, :, 0, fi, :, :].float()
            pred_kp   = smpl_output['pred_keypoints'][:, fi, :, :].view(BS, T, P, 45, 2)
            pred_kp   = pred_kp[:, :, 0, :25, :]
            gt_kp     = gt_kpc[:, :, :, :2]
            gt_conf   = gt_kpc[:, :, :, 2:]
            
            # gt_kp     = torch.clamp(gt_kp, -0.5, 0.5)
            # pred_kp   = torch.clamp(pred_kp, -0.5, 0.5)
            # import ipdb; ipdb.set_trace()
            loss_kp   = 10.0*(gt_conf[loca_loss[:, :, 0]] * ( pred_kp[loca_loss[:, :, 0]] - gt_kp[loca_loss[:, :, 0]])).abs().sum()/(torch.sum(loca_loss[:, :, 0])+1)
            
        if("action" in opt.loss_type):
            if("ava" in opt.action_space):
                gt_has_annotation  = output_data['has_gt'][:, :, :, fi, :]
                if(opt.ava.gt_type=="both"):
                    loca_loss_annot_1 = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]==1)
                    loca_loss_annot_2 = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]==2)
                    
                    try:
                        if(opt.ava.distil_type=="both_bce"):
                            loss_action_1 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot_1], gt_action_ava[loca_loss_annot_1])
                            loss_action_2 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot_2], gt_action_ava[loca_loss_annot_2])
                            loss_action   = 0.5*torch.nan_to_num(loss_action_1) + torch.nan_to_num(loss_action_2)
                            
                        elif(opt.ava.distil_type=="10kl_bce"):
                            loss_action_1 = torch.sigmoid(pred_action_ava[loca_loss_annot_1] - gt_action_ava[loca_loss_annot_1]).pow(2).sum()/(torch.sum(loca_loss_annot_1)+1)
                            loss_action_2 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot_2], gt_action_ava[loca_loss_annot_2])
                            loss_action   = 0.1*torch.nan_to_num(loss_action_1) + torch.nan_to_num(loss_action_2)
                            
                        elif(opt.ava.distil_type=="kl_bce"):
                            loss_action_1 = torch.sigmoid(pred_action_ava[loca_loss_annot_1] - gt_action_ava[loca_loss_annot_1]).pow(2).sum()/(torch.sum(loca_loss_annot_1)+1)
                            loss_action_2 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot_2], gt_action_ava[loca_loss_annot_2])
                            loss_action   = torch.nan_to_num(loss_action_1) + torch.nan_to_num(loss_action_2)
                        else:
                            raise NotImplementedError
                    except:
                        loss_action_1 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot_1], gt_action_ava[loca_loss_annot_1])
                        loss_action_2 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot_2], gt_action_ava[loca_loss_annot_2])
                        loss_action   = 0.1*torch.nan_to_num(loss_action_1) + torch.nan_to_num(loss_action_2)
                        
                    
                else:
                    if(opt.ava.gt_type=="gt"):
                        loca_loss_annot = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]==2)
                    elif(opt.ava.gt_type=="pseduo_gt"):
                        loca_loss_annot = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]==1)
                    elif(opt.ava.gt_type=="all"):
                        loca_loss_annot = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]>=1)
                    else:
                        raise ValueError("Unknown ava gt type")

                    if("kl_l2" in opt.loss_type.split("action")[1]):
                        loss_action = 10*( torch.sigmoid(pred_action_ava[loca_loss_annot]) - gt_action_ava[loca_loss_annot] ).pow(2).sum()/(torch.sum(loca_loss_annot)+1)
                    if("bce" in opt.loss_type.split("action")[1]):
                        loss_action2 = F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot], gt_action_ava[loca_loss_annot])
                        loss_action = torch.nan_to_num(loss_action2)
                    if("BCE" in opt.loss_type.split("action")[1]):
                        loss_action = 10 * F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot], gt_action_ava[loca_loss_annot])
                    if("10BCE" in opt.loss_type.split("action")[1]):
                        loss_action = 100 * F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot], gt_action_ava[loca_loss_annot])
                    if("BCK" in opt.loss_type.split("action")[1]):
                        loss_action = 10 * F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot], F.sigmoid((gt_action_ava[loca_loss_annot]-0.5)/0.2))
                    if("FOC" in opt.loss_type.split("action")[1]):
                        # loss_action = 10 * sigmoid_focal_loss(pred_action_ava[loca_loss_annot], gt_action_ava[loca_loss_annot], alpha=opt.loss.focal.alpha, gamma=opt.loss.focal.gamma).mean()                
                        loss_action = cb_loss(pred_action_ava[loca_loss_annot], gt_action_ava[loca_loss_annot])
                    
            if("kinetics" in opt.action_space):
                # import ipdb; ipdb.set_trace()
                gt_has_annotation_k = output_data['has_gt_kinetics'][:, 0, :, 0, 0]==1
                # cross entropy loss with one-hot encoding
                # if("CE" in opt.loss_type.split("action")[1]):
                loss_action_k = F.cross_entropy(pred_action_kinetics[gt_has_annotation_k], gt_action_kinetics[:, 0][gt_has_annotation_k][:, 0].long()) # for 0th person
                loss_action += torch.nan_to_num(loss_action_k)
        
        # log the loss values
        loss_dict["pose"]    += loss_pose/opt.num_smpl_heads
        loss_dict["loca"]    += loss_loca/opt.num_smpl_heads
        loss_dict["action"]  += loss_action/opt.num_smpl_heads
        loss_dict["kp"]      += loss_kp/opt.num_smpl_heads
    
    return loss_dict

def train_disc(idx, BS, T, P, smpl_output, mosh_dataset, discriminator, optimizer_disc):
    
    idx_pose                 = copy.deepcopy(idx)
    idx_pose                 = idx_pose.view(BS*T*P,)
    body_pose                = smpl_output['smpl_output'].body_pose.detach()[idx_pose==1]
    betas                    = smpl_output['smpl_output'].betas.detach()[idx_pose==1]
    effective_BS             = torch.sum(idx==1)
    gt_body_pose, gt_betas   = mosh_dataset[effective_BS]
    gt_body_pose             = gt_body_pose.cuda()
    gt_betas                 = gt_betas.cuda()
    gt_rotmat                = batch_rodrigues(gt_body_pose.view(-1,3)).view(effective_BS, -1, 3, 3)
    disc_fake_out, loss_fake = discriminator(body_pose, betas, compute_loss=True, label=0.)
    disc_real_out, loss_real = discriminator(gt_rotmat, gt_betas, compute_loss=True, label=1.)
    loss_disc                = loss_fake + loss_real
    optimizer_disc.zero_grad()
    loss_disc.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.1)
    optimizer_disc.step()

    return loss_disc.detach()

    
class ParameterLoss(nn.Module):

    def __init__(self):
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param, glabl_ori, body_pose, betas, idx):
        # breakpoint()
        BS, TP             = idx.shape
        pred_global_orient = pred_param['smpl_output'].global_orient
        pred_body_pose     = pred_param['smpl_output'].body_pose
        pred_betas         = pred_param['smpl_output'].betas
        
        glabl_ori_         = glabl_ori.view(BS*TP, 1, 3, 3)
        body_pose_         = body_pose.view(BS*TP, 23, 3, 3)
        betas_             = betas.view(BS*TP, 10)
        idx_               = idx.view(BS*TP,)

        loss_param = 0
        loss_param += (self.loss_fn(pred_global_orient[idx_==1], glabl_ori_[idx_==1])).sum() / (1 + torch.sum(idx_))
        loss_param += (self.loss_fn(pred_body_pose[idx_==1], body_pose_[idx_==1])).sum() / (1 + torch.sum(idx_))
        loss_param += (self.loss_fn(pred_betas[idx_==1], betas_[idx_==1])).sum() / (1 + torch.sum(idx_))

        return loss_param

    
class Smoothness2DLoss(nn.Module):

    def __init__(self, loss_type='l1'):
        super(Smoothness2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
        
        self.frame_length = 50
        self.max_people   = 3
        self.smooth_mask  = torch.zeros(self.frame_length*self.max_people, self.frame_length*self.max_people, dtype=int)
        
        self.attend  = nn.Softmax(dim = -1)

        for t in range(1,self.frame_length-1):
            self.smooth_mask[(t-1)*self.max_people:(t+1)*self.max_people, (t-1)*self.max_people:(t+1)*self.max_people] = 1
    
    
    def forward(self, idx, idx_mask, idx_kp, smpl_output, attentions):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        BS, T, P  = idx_kp.shape
        
        keypoints_2d    = smpl_output['2d_points'].view(BS, T*P, 44, 2)
        keypoints_2d_   = keypoints_2d.unsqueeze(1).repeat(1, T*P, 1, 1, 1)
        loss_kp         = self.loss_fn(keypoints_2d_, keypoints_2d_.transpose(1,2))
        loss_kp         = torch.sum(loss_kp, (3,4))
        loss_kp_        = loss_kp.unsqueeze(1)
        
        attentions[0][:, :, self.smooth_mask==0] = -1e10
        atten3          = self.attend(attentions[-1])
        smooth_loss     = loss_kp_*atten3
        idx_            = rearrange(idx, 'b i -> b () i ()') * rearrange(idx, 'b j -> b () () j')
        idx_            = idx_.repeat(1, smooth_loss.shape[1], 1, 1)
        loss            = torch.sum(smooth_loss[idx_==1])/ (1 + torch.sum(idx_))
        
        return loss

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type='l1'):
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
    
    def forward(self, idx, idx_mask, idx_kp, num_masked, keypoints, smpl_output):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        BS, T, P, _, _  = keypoints.shape
        keypoints       = keypoints.cuda()
        keypoints       = keypoints.view(BS*T*P, 25, 3)
        pred_keypoints  = smpl_output['2d_points'][:, :25, :]
        pred_keypoints += 0.5
        
        idx_kp = idx_kp.view(BS*T*P,)
        conf = keypoints[:, :, -1].unsqueeze(-1).clone()
        loss = (conf[idx_kp==1] * self.loss_fn(pred_keypoints[idx_kp==1], keypoints[idx_kp==1][:, :, :-1])).sum() / (torch.sum(idx_kp==1) + 1)

        return loss

class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type='l1'):
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
    
    def forward(self, idx, idx_mask, idx_kp, num_masked, keypoints, smpl_output):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        BS, T, P, _, _  = keypoints.shape
        keypoints       = keypoints.cuda()
        keypoints       = keypoints.view(BS*T*P, 45, 3)
        pred_keypoints  = smpl_output['2d_points'][:, :45, :]
        pred_keypoints += 0.5
        
        idx_kp = idx_kp.view(BS*T*P,)
        # conf = keypoints[:, :, -1].unsqueeze(-1).clone()
        loss = (self.loss_fn(pred_keypoints[idx_kp==1], keypoints[idx_kp==1][:, :, :-1])).sum() / (torch.sum(idx_kp==1) + 1)

        return loss

    
class Location3DLoss(nn.Module):

    def __init__(self, loss_type='l1'):
        super(Location3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
    
    def forward(self, idx, idx_mask, num_masked, onehot_loca, loca_output):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        """
        BS, T, P, _        = onehot_loca.size()
        onehot_loca        = onehot_loca.view(BS*T*P, 3)
        
        pred_x             = loca_output[0].view(BS*T*P, 1000)
        pred_y             = loca_output[1].view(BS*T*P, 1000)
        pred_n             = loca_output[2].view(BS*T*P, 1000)
        idx                = idx.view(BS*T*P)
        loss = self.loss_fn(pred_x[idx==1], onehot_loca[idx==1][:, 0].cuda()) + self.loss_fn(pred_y[idx==1], onehot_loca[idx==1][:, 1].cuda()) + self.loss_fn(pred_n[idx==1], onehot_loca[idx==1][:, 2].cuda())

        return loss  
