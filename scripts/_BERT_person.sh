#!/bin/bash

# source ~/.zshrc
# conda activate HUMAN_BERT_2


# python src/train.py -m \
# trainer=ddp \
# task_name=3000_P1 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=3000_P2 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3,2e-3,4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1,0.2,0.3 \
# configs.masked=False \
# +seed=0,1,3 \





python src/train.py -m \
hydra/launcher=submitit_slurm \
launcher=slurm \
trainer=ddp \
task_name=3000_P3 \
configs.dataset_loader=fast \
configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
configs.test_dataset=\'ava_val_8\' \
configs.ava.predict_valid=True \
configs.bottle_neck="conv2k" \
configs.ava.gt_type="all" \
configs.ava.distil_type="both_bce" \
configs.lr=1e-3,2e-3,4e-3 \
configs.mask_type_test="zero" \
configs.frame_length=125 \
configs.max_people=1 \
configs.loss_type=\'action_bce\' \
configs.extra_feat.enable=\'mvit\' \
configs.extra_feat.mvit.mid_dim=1024 \
configs.extra_feat.mvit.en_dim=1024 \
configs.extra_feat.pose_shape.mid_dim=229 \
configs.extra_feat.pose_shape.en_dim=256 \
configs.extra_feat.relative_pose.mid_dim=512 \
configs.extra_feat.relative_pose.en_dim=1280 \
configs.use_relative_pose=True \
configs.in_feat=1280 \
configs.weight_decay=5e-2 \
trainer.gradient_clip_val=2.0 \
configs.mask_ratio=0.1 \
configs.masked=False \
configs.vit.depth=3,9 \
configs.vit.heads=8,12 \
configs.vit.mlp_dim=128,512 \
configs.vit.dim_head=32,128 \
configs.vit.dropout=0.1 \
configs.vit.droppath=0.1 \






# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=video_1 \
# configs.wights_path="/private/home/jathushan/3D/BERT_person_hydra/logs/1990_4p3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'youtube\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.render.enable=True \
# configs.render.num_videos=55 \
# configs.render.vis_action_label="all" \
# configs.compute_map=False \
# trainer.devices=1 \
# callbacks.rich_progress_bar.refresh_rate=0 \





