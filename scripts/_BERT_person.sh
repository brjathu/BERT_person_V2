#!/bin/bash

# source ~/.zshrc
# conda activate HUMAN_BERT_2






# python src/train.py \
# task_name=1207 \
# configs.train_dataset=\'ava_train,kinetics_train\' \
# configs.action_space="ava" \
# configs.decay_steps=[15,25] \
# configs.render.enable=True \
# configs.render.background=False \
# configs.render.texture=True \
# configs.render.engine="NMR" \
# configs.bottle_neck="conv" \
# configs.extra_feat="appe" \
# configs.frame_length=50 \
# # callbacks.rich_progress_bar.refresh_rate=0 \


# 1220_test - PL walker
# 1221_test - SMPL walker
# 1222_test - SMPL+Texture walker
# 1223_test - Background+SMPL+Texture walker

# python src/train.py \
# task_name=1220_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'kinetics_val\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="PL" \
# configs.frame_length=50 \
# configs.test_batch_id=-1 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=1 \



# python src/train.py \
# task_name=1221_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'ava_rend_climb\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="SMPL" \
# configs.frame_length=50 \
# configs.test_batch_id=-1 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=1 \



# python src/train.py \
# task_name=1222_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'kinetics_val\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="SMPL+T" \
# configs.render.engine="NMR" \
# configs.frame_length=50 \
# configs.test_batch_id=-1 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=1 \



# python src/train.py \
# task_name=1223_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'kinetics_val\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="SMPL+T+B" \
# configs.render.engine="NMR" \
# configs.frame_length=50 \
# configs.test_batch_id=-1 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=1 \

















# python src/train.py \
# train=False \
# task_name=1300_1 \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'ava_val_\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.extra_feat="action" \
# configs.render.enable=False \
# configs.frame_length=50 \
# configs.test_type="track.fullframe|gt"  \
# # callbacks.rich_progress_bar.refresh_rate=0 \


# python src/train.py \
# train=False \
# task_name=1300_2 \
# configs.train_dataset=\'ava_val\' \
# configs.test_dataset=\'ava_val\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.test_type="track.fullframe|gt"  \
# trainer.limit_val_batches=100 \
# # callbacks.rich_progress_bar.refresh_rate=0 \


# python src/train.py \
# train=False \
# task_name=1300_3 \
# configs.train_dataset=\'ava_val\' \
# configs.test_dataset=\'ava_val\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="clip" \
# configs.extra_feat.clip.en_dim=229 \
# configs.test_type="track.fullframe|clip"  \
# # trainer.limit_val_batches=100 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py \
# train=False \
# task_name=1300_4 \
# configs.train_dataset=\'ava_val\' \
# configs.test_dataset=\'ava_val\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.extra_feat.clip.en_dim=229 \
# configs.test_type="track.fullframe|gt"  \
# # trainer.limit_val_batches=100 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py \
# train=False \
# task_name=1300_5 \
# configs.train_dataset=\'ava_val_4\' \
# configs.test_dataset=\'ava_val_4\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.extra_feat.clip.en_dim=229 \
# configs.test_type="track.fullframe|gt"  \
# # trainer.strategy=ddp \
# # trainer.sync_batchnorm=True \
# # trainer.devices=8 \
# # trainer.num_nodes=1 \
# # callbacks.model_checkpoint.monitor="step" \
# # callbacks.model_checkpoint.mode="max" \
# # callbacks.model_checkpoint.save_top_k=-1 \
# # configs.train_batch_size=8 \
# # configs.train_num_workers=8 \
# # configs.test_batch_size=8 \
# # configs.test_num_workers=8 \
# # trainer.limit_val_batches=8 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py \
# train=False \
# task_name=1300_6 \
# configs.train_dataset=\'ava_val_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.extra_feat.mvit.en_dim=256 \
# configs.in_feat=485 \
# configs.test_type="track.fullframe|gt"  \
# trainer.strategy=ddp \
# trainer.sync_batchnorm=True \
# trainer.devices=8 \
# trainer.num_nodes=1 \
# callbacks.model_checkpoint.monitor="step" \
# callbacks.model_checkpoint.mode="max" \
# callbacks.model_checkpoint.save_top_k=-1 \
# configs.train_batch_size=8 \
# configs.train_num_workers=8 \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \





# python src/train.py \
# trainer=ddp \
# train=False \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# task_name=1300_7 \
# configs.train_dataset=\'ava_val_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.in_feat=512 \
# configs.test_type="track.fullframe|gt" \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# task_name=1300_8 \
# configs.train_dataset=\'ava_val_6\' \
# configs.test_dataset=\'ava_val_6\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.in_feat=512 \
# configs.test_type="track.fullframe|gt" \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# task_name=1300_9 \
# configs.train_dataset=\'ava_val_7\' \
# configs.test_dataset=\'ava_val_7\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.in_feat=512 \
# configs.test_type="track.fullframe|gt" \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# task_name=1300_10 \
# configs.train_dataset=\'ava_val_7\' \
# configs.test_dataset=\'ava_val_7\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.in_feat=512 \
# configs.test_type="track.fullframe|gt" \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# task_name=1300_11 \
# configs.train_dataset=\'ava_val_8\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.in_feat=512 \
# configs.test_type="track.fullframe|gt" \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.test_batch_size=8 \
# configs.test_num_workers=8 \
# task_name=1300_12 \
# configs.train_dataset=\'ava_val_8\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.compute_map=True \
# configs.action_space="ava" \
# configs.render.enable=False \
# configs.frame_length=125 \
# configs.extra_feat.enable="action" \
# configs.in_feat=512 \
# configs.test_type="track.fullframe|gt" \
# configs.ava.predict_valid=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.limit_val_batches=8 \











# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1970h \
# configs.train_dataset=\'kinetics_train_4,ava_train_4\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,hmr\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.hmr.en_dim=1024 \
# configs.extra_feat.hmr.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \



# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1970h2 \
# configs.train_dataset=\'kinetics_train_4,ava_train_4\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_bce\' \
# configs.extra_feat.enable=\'mvit,hmr\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.hmr.en_dim=1024 \
# configs.extra_feat.hmr.mid_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \



# python src/train.py -m \
# train=False \
# trainer=ddp \
# task_name=1970h2_test \
# configs.wights_path="logs/1970h2/0/checkpoints/last.ckpt" \
# configs.train_dataset=\'kinetics_train_4,ava_train_4\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_bce\' \
# configs.extra_feat.enable=\'mvit,hmr\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.hmr.en_dim=1024 \
# configs.extra_feat.hmr.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# configs.test_type="track.fullframe|arg.30" \
# configs.render.enable=True \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \




# python src/train.py -m \
# configs.wights_path="logs/1970h2/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=1970h2_finetune2 \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.lr=1e-3 \
# configs.loss_type=\'pose_l2,loca_l1,kp_l1,action_bce\' \
# configs.extra_feat.enable=\'mvit,hmr\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.hmr.en_dim=1024 \
# configs.extra_feat.hmr.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.test_type="track.fullframe|arg.30" \
# # configs.render.enable=True \
# # configs.render.num_videos=1 \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \



# python src/train.py -m \
# configs.wights_path="logs/1970h2/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=1970h2_finetune3 \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.lr=1e-3 \
# configs.loss_type=\'loca_l1,pose_l2,kp_l1,action_bce\' \
# configs.extra_feat.enable=\'mvit,hmr\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.hmr.en_dim=1024 \
# configs.extra_feat.hmr.mid_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.4 \
# configs.masked=True \
# configs.test_type="track.fullframe|arg.30" \
# trainer.check_val_every_n_epoch=1000 \
# configs.compute_map=False \
# # configs.render.enable=True \
# # configs.render.num_videos=1 \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \




# python src/train.py -m \
# train=False \
# configs.wights_path="logs/1970h2_finetune3/0/checkpoints/epoch_001.ckpt" \
# trainer=ddp \
# task_name=1970h2_finetune3_test \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.lr=1e-3 \
# configs.loss_type=\'loca_l1,kp_l1,action_bce\' \
# configs.extra_feat.enable=\'mvit,hmr\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.hmr.en_dim=1024 \
# configs.extra_feat.hmr.mid_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.test_type="track.fullframe|arg.30" \
# trainer.check_val_every_n_epoch=1000 \
# configs.compute_map=False \
# configs.render.enable=True \
# configs.render.num_videos=100 \
# trainer.devices=1 \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \


























# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19610 \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.6 \
# configs.masked=False \
# configs.warmup_epochs=5 \
# trainer.max_epochs=50 \




# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19610a \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.6 \
# configs.masked=False \
# configs.warmup_epochs=5 \
# trainer.max_epochs=50 \



# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19610b \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.scheduler="cosine","step" \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0,5.0,10.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \



# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19610c \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.scheduler="cosine","step" \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0,5.0,10.0 \
# configs.mask_ratio=0.6 \
# configs.masked=True \



# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19610d \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.scheduler="cosine","step" \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_BCE\' \
# configs.test_type="track.fullframe|avg.30" \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0,5.0,10.0 \
# configs.mask_ratio=0.6 \
# configs.masked=True \





# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=19610e \
# configs.train_dataset=\'kinetics_train_4,ava_train_4\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.scheduler="cosine","step" \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_BCE\' \
# configs.test_type="track.fullframe|avg.30" \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0,5.0,10.0 \
# configs.mask_ratio=0.6 \
# configs.masked=True \





# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19611 \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'pose_l2,loca_l1,action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.6 \
# configs.masked=True \
# configs.warmup_epochs=5 \
# trainer.max_epochs=50 \





# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# configs.dataset_loader=fast \
# trainer=ddp \
# task_name=19611a \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# configs.warmup_epochs=5 \
# trainer.max_epochs=50 \





# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611b \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# configs.warmup_epochs=5 \
# trainer.max_epochs=50 \





# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# # trainer.devices=1 \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611c \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \




# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611d \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2i" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611e \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611g \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611h \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611i \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611j \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \








# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611k \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# train=False \
# trainer=ddp \
# task_name=19611l \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.test_type="track.fullframe|GT"  \
# # trainer.devices=1 \
# # configs.train_batch_size=1 \
# # configs.test_batch_size=1 \
# # configs.train_num_workers=1 \
# # configs.test_num_workers=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611m \
# configs.train_dataset=\'ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.test_type="track.fullframe|GT"  \
# # trainer.devices=1 \
# # configs.train_batch_size=1 \
# # configs.test_batch_size=1 \
# # configs.train_num_workers=1 \
# # configs.test_num_workers=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# # configs.wights_path="logs/19602/0/checkpoints/last.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19611n \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_val_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
# configs.load_other_tracks=True \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # configs.test_type="track.fullframe|GT"  \
# # trainer.devices=1 \
# # configs.train_batch_size=1 \
# # configs.test_batch_size=1 \
# # configs.train_num_workers=1 \
# # configs.test_num_workers=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




















































































































# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=19611n \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=1 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.warmup_epochs=10 \
# trainer.max_epochs=100 \
# trainer.check_val_every_n_epoch=10000 \


# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=19611o \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=1 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.warmup_epochs=10 \
# trainer.max_epochs=100 \
# trainer.check_val_every_n_epoch=10000 \


# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# hydra.launcher.timeout_min=300 \
# configs.wights_path="logs/1970d/0/checkpoints/epoch_023.ckpt" \
# task_name=19613c \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4,1e-3 \
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
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# +trainer.val_check_interval=100 \



# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# task_name=19612g \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
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
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \






# # configs.dataset_loader=fast \
# python src/train.py -m \
# train=False \
# trainer=ddp \
# task_name=19612g6_test \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_6\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
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
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \


















# python src/train.py -m \
# trainer=ddp \
# task_name=19612a \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # configs.warmup_epochs=10 \
# # trainer.max_epochs=100 \
# # trainer.check_val_every_n_epoch=10000 \
# # trainer.devices=1 \
# # configs.train_batch_size=1 \
# # configs.test_batch_size=1 \
# # configs.train_num_workers=1 \
# # configs.test_num_workers=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.test_type="track.fullframe|GT"  \


# python src/train.py -m \
# trainer=ddp \
# trainer.benchmark=False \
# task_name=19612b \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
# configs.loss_type=\'action_BCE\' \
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
# trainer=ddp \
# task_name=19612d \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
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
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \



# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612d/0/checkpoints/last.ckpt" \
# task_name=19612e \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# # trainer.limit_train_batches=100 \





# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612d/0/checkpoints/last.ckpt" \
# task_name=19612f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# # trainer.limit_train_batches=100 \





# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# configs.wights_path="logs/19612d/0/checkpoints/last.ckpt" \
# task_name=19612h \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# # trainer.limit_train_batches=100 \





# python src/train.py -m \
# trainer=ddp \
# task_name=19612i \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,clip\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.clip.mid_dim=512 \
# configs.extra_feat.clip.en_dim=256 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1536 \
# configs.use_relative_pose=False \
# configs.in_feat=1536 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \





# python src/train.py -m \
# trainer=ddp \
# task_name=19612j \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,vitpose\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.vitpose.mid_dim=64 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=256 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \




# python src/train.py -m \
# configs.wights_path="logs/19612d/0/checkpoints/epoch_012.ckpt" \
# trainer=ddp \
# task_name=19612h2 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="gt" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# configs.weight_decay=5e-6 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \






# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612g/4/checkpoints/epoch_018.ckpt" \
# task_name=19612h3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# # trainer.limit_train_batches=100 \





# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612g/4/checkpoints/epoch_018.ckpt" \
# task_name=19612h4 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=10 \
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
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=2 \
# configs.test_num_workers=2 \
# trainer.accumulate_grad_batches=8 \
# # trainer.limit_train_batches=100 \






# # train=False \
# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612h4/0/checkpoints/epoch_003.ckpt" \
# task_name=19612h4_test3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=10 \
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
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=2 \
# configs.test_num_workers=2 \
# trainer.accumulate_grad_batches=8 \
# trainer.devices=1 \
# # trainer.limit_train_batches=100 \






# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# task_name=19613g \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4.8,1.2,0.3 \
# configs.solver="SGD" \
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
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# task_name=19613g \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4.8,1.2,0.3 \
# configs.solver="SGD" \
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
# configs.weight_decay=5e-2,5e-4,5e-6 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# task_name=19613g3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=0.1,0.075,0.050,0.025 \
# configs.solver="SGD" \
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
# configs.weight_decay=5e-4,5e-6,5e-8 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \






# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# task_name=19614g \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.layer_decay=0.95,0.9,0.8,0.7 \
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
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# task_name=19614x \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.layer_decay=0.75 \
# configs.mask_type_test="zero_x" \
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
# trainer.devices=1 \
# trainer.limit_train_batches=100 \
# trainer.limit_val_batches=100 \
# configs.compute_map=False \
# # trainer.check_val_every_n_epoch=100 \
# # callbacks.rich_progress_bar.refresh_rate=0 \



# python src/train.py -m \
# configs.wights_path="logs/19611o/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=19611o_finetune1 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.layer_decay=0.7 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \






# python src/train.py -m \
# configs.wights_path="logs/19611o/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=19611o_finetune2 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.layer_decay=0.7 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \




# python src/train.py -m \
# configs.wights_path="logs/19611o/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=19614y \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.layer_decay=0.7 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.devices=1 \






# python src/train.py -m \
# trainer=ddp \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# task_name=19612k \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# configs.wights_path="logs/19611o/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=19612l \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.layer_decay=0.9 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \





# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# configs.wights_path="logs/19611o/0/checkpoints/last.ckpt" \
# trainer=ddp \
# task_name=19612m \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.layer_decay=0.9,0.8,0.7 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1,0.2,0.4 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# task_name=19612n \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.0 \
# # trainer.limit_train_batches=100 \






# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# task_name=19612n2 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.1 \
# # trainer.limit_train_batches=100 \




# # configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19612o \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,objects\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=192 \
# configs.extra_feat.objects.mid_dim=64 \
# configs.extra_feat.objects.en_dim=64 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.0 \
# # trainer.limit_train_batches=100 \



# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# task_name=19612n3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.0 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=null \
# # trainer.limit_train_batches=100 \



# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# task_name=19612n4 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.0 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.accumulate_grad_batches=null \
# # trainer.limit_train_batches=100 \




# # configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19612n5 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv_1j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
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
# configs.vit.droppath=0.0 \
# configs.train_batch_size=8 \
# configs.test_batch_size=8 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# # trainer.accumulate_grad_batches=null \
# # trainer.precision=16 \
# # trainer.limit_train_batches=100 \






# # # configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# trainer=ddp \
# task_name=19612n6 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv_1j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\',\'pose_l2,loca_l1,action_BCE\' \
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
# configs.mask_ratio=0.1,0.3,0.6 \
# configs.masked=False,True \
# configs.vit.droppath=0.0,0.2 \
# configs.train_batch_size=8 \
# configs.test_batch_size=8 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.accumulate_grad_batches=null,4,8 \




# # configs.wights_path="logs/19612d/0/checkpoints/epoch_009.ckpt" \
# python src/train.py -m \
# trainer=ddp \
# task_name=19612n7 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv_1j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_BCE\' \
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
# configs.mask_ratio=0.4 \
# configs.masked=True \
# configs.vit.droppath=0.0 \
# configs.train_batch_size=8 \
# configs.test_batch_size=8 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# # trainer.accumulate_grad_batches=null \
# # trainer.precision=16 \
# # trainer.limit_train_batches=100 \








# python src/train.py -m \
# trainer=ddp \
# task_name=1980 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \



# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1980_test \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1980/0/checkpoints/last.ckpt" \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.devices=1 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1980a \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \X
# configs.mask_ratio=0.1 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1981 \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_6\' \
# configs.test_dataset=\'ava_val_6\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1981a \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'kinetics_train_4,ava_train_5,avaK_train_6\' \
# configs.test_dataset=\'ava_val_6\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
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
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1982 \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_7\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test \
# configs.wights_path="logs/1981/0/checkpoints/epoch_005.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_7\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \



# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test3 \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_7\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# trainer.devices=1 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test4 \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test5 \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test6 \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|avg.30" \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test7 \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# configs.full_seq_render=True \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test9 \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_val_8\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test10 \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_val_8\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# configs.render.enable=True \
# configs.render.num_videos=100 \
# configs.debug=True \
# callbacks.rich_progress_bar.refresh_rate=0 \
# trainer.devices=1 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=1 \
# configs.test_num_workers=1 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1982_test11 \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_train_7\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# configs.render.enable=True \
# configs.render.num_videos=100 \
# configs.debug=True \
# callbacks.rich_progress_bar.refresh_rate=0 \
# trainer.devices=1 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=1 \
# configs.test_num_workers=1 \
# configs.compute_map=False \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1982a \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_5,avaK_train_7\' \
# configs.test_dataset=\'ava_val_7\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# # trainer.accumulate_grad_batches=2 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \































































# python src/train.py -m \
# trainer=ddp \
# task_name=1990_1a \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_1b \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_1c \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="gt" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \









# python src/train.py -m \
# trainer=ddp \
# task_name=1990_1d \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="gt" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \








# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990 \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt","a" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\',\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="both","gt" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4,1e-4,4e-4 \
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
# configs.use_optimized_pose=False,True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_2d \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="gt" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# +trainer.val_check_interval=1000 \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \



# python src/train.py -m \
# trainer=ddp \
# task_name=1990_2e \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="both" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_bce\' \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# # +trainer.val_check_interval=1000 \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \








# python src/train.py -m \
# trainer=ddp \
# task_name=1990_3a \
# configs.dataset_loader=fast \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \



# python src/train.py -m \
# trainer=ddp \
# task_name=1990_3b \
# configs.dataset_loader=fast \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \



# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.compute_map=False \
# configs.debug=True \
# configs.store_svm_vectors=True \
# configs.svm_folder="_TMP/svm_01" \
# task_name=1990_3b_test3 \
# configs.dataset_loader=fast \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_train_7\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# trainer.devices=1 \
# # trainer.limit_val_batches=100 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# configs.compute_map=True \
# configs.debug=True \
# configs.store_svm_vectors=True \
# configs.svm_folder="_TMP/svm_01" \
# task_name=1990_3b_test4 \
# configs.dataset_loader=fast \
# configs.wights_path="logs/19612g/6/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# trainer.devices=1 \
# # trainer.limit_val_batches=100 \
# # callbacks.rich_progress_bar.refresh_rate=0 \


# python src/train.py -m \
# trainer=ddp \
# task_name=1990_3c \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=16 \







# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1991 \
# configs.dataset_loader=fast \
# configs.train_dataset="kinetics_train_7",\'ava_train_7,avaK_train_7,kinetics_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero","zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=1,4,8 \
# # configs.layer_decay=0.9 \
# # configs.vit.droppath=0.1 \
# # configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \





# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1992 \
# configs.dataset_loader=fast \
# configs.train_dataset="kinetics_train_7",\'ava_train_7,avaK_train_7,kinetics_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero","zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'pose_l2,loca_l1,action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=1,4,8 \
# # configs.layer_decay=0.9 \
# # configs.vit.droppath=0.1 \
# # configs.test_type="track.fullframe|" \
# # configs.render.enable=True \
# # configs.render.num_videos=10 \
# # trainer.limit_train_batches=100 \
# # trainer.check_val_every_n_epoch=10000 \




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1993 \
# configs.dataset_loader=fast \
# configs.train_dataset="kinetics_train_7",\'ava_train_7,avaK_train_7,kinetics_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero","zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=1,8,16 \








# python src/train.py -m \
# trainer=ddp \
# task_name=1990_3d \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.weight_decay=5e-6 \
# trainer.gradient_clip_val=2000.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=1 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \



# python src/train.py -m \
# trainer=ddp \
# task_name=1990_3e \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.layer_decay=0.99 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=16 \
# configs.loss_on_others_action=False \




# python src/train.py -m \
# train=False \
# trainer=ddp \
# task_name=1990_3e_test3 \
# configs.wights_path="logs/1990_3e/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.layer_decay=0.99 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=16 \
# configs.loss_on_others_action=False \




# python src/train.py -m \
# train=False \
# trainer=ddp \
# task_name=1990_3e_test4 \
# configs.wights_path="logs/1990_3e/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero_x" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# configs.layer_decay=0.99 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=16 \
# configs.loss_on_others_action=False \






































































# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4a \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7\' \
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
# trainer=ddp \
# task_name=1990_4b \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4c \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'pose_l2,loca_l1,action_BCE\' \
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
# configs.mask_ratio=0.6 \
# configs.masked=True \





# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/1990_4a/0/checkpoints/last.ckpt" \
# task_name=1990_4d \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \




# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/1990_4a/0/checkpoints/last.ckpt" \
# task_name=1990_4e \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \





# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1994 \
# configs.wights_path="logs/1990_4a/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all","both" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.vit.droppath=0,0.1,0.2 \
# configs.layer_decay=null,0.99,0.9 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8,16 \







# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# trainer=ddp \
# task_name=1995 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
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
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.depth=3,6,9 \
# configs.vit.heads=4,8,12 \
# configs.vit.mlp_dim=128,512,1024 \
# configs.vit.dim_head=32,64,128 \










# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
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
# trainer=ddp \
# configs.wights_path="logs/1990_4a/0/checkpoints/last.ckpt" \
# task_name=1990_4g \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# # configs.layer_decay=0.9 \
# # configs.vit.droppath=0.1 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4h \
# configs.train_dataset=\'kinetics_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
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
# configs.action_space="ava,kinetics" \




# python src/train.py -m \
# trainer=ddp \
# configs.wights_path="logs/1990_4a/0/checkpoints/last.ckpt" \
# task_name=1990_4i \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# # configs.layer_decay=0.9 \
# # configs.vit.droppath=0.1 \



# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4j \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1981/0/checkpoints/epoch_009.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# # configs.loss_on_others_action=False \



# for i in {0..9}; do
# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4j_test_${i} \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1990_4j/0/checkpoints/last.ckpt" \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# trainer.devices=2 
# done





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_5f \
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
# trainer=ddp \
# train=False \
# task_name=1990_5f_test \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
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
# trainer=ddp \
# train=False \
# task_name=1990_5f_test2 \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
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
# configs.test_type="track.fullframe|max.15" \
# configs.debug=False \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # trainer.devices=1 \





# params1=(8 10 12 15 20 30)
# for i in {0..5}; do
# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_5f_test_x${params1[i]}$ \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
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
# configs.test_type="track.fullframe|avg.${params1[i]}" \
# configs.debug=False
# done
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_7f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \









# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1996 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\',\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3,8e-3 \
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
# configs.use_optimized_pose=False,True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.depth=6,9 \
# configs.vit.heads=8,12 \







# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1997 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\',\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3,4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_8f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_9f \
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
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_10f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=896 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \







# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
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





# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4p3_test2 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
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




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4p3_test3 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero_x" \
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






# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4p3_test4 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/epoch_028.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
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



# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4p3_test5 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/epoch_028.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=10 \
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
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=2 \
# configs.test_num_workers=2 \
# trainer.accumulate_grad_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p3_t1 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \









# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p10 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2m" \
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
# configs.mask_ratio=0.6 \
# configs.masked=True \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p11 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2m" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.loss_type=\'action_bce\' \
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
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \






# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4p12 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
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
# configs.use_optimized_pose=True \
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




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4p4 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
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
# configs.use_optimized_pose=True \
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



# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4p1 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p1_test \
# train=False \
# configs.wights_path="logs/1990_4p1/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.render.enable=True \
# configs.render.num_videos=1000 \
# trainer.devices=1 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p0 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
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







# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p20 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose_only\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.mid_dim=256 \
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



# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p20x \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose_only\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.mid_dim=256 \
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




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4p21x \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
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




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4p21y \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
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




# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p21z \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
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





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p21 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose_only\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=False \
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




# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p22 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.mid_dim=256 \
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





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p23 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \




# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p23x \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
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





# python src/train.py -m \
# launcher=slurm \
# task_name=1990_4p24 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
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




# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p30 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv3l" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose,img\' \
# configs.extra_feat.img.mid_dim=229 \
# configs.extra_feat.img.en_dim=256 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
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
# # trainer.devices=1 \
# # configs.debug=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p31 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv3l" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose,img\' \
# configs.extra_feat.img.mid_dim=229 \
# configs.extra_feat.img.en_dim=256 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.action_space=\'ava,kinetics\' \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.debug=True \








# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4p32 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
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





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4p33 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_BCE\' \
# configs.extra_feat.enable=\'vitpose\' \
# configs.extra_feat.vitpose.mid_dim=256 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.action_space="ava,kinetics" \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_11f \
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
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1280 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \









# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4s \
# configs.wights_path="logs/1990_4f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \






# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_5s \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_6s \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \







# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_8s \
# configs.wights_path="logs/1990_8f/0/checkpoints/epoch_020.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \









# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_4t \
# configs.wights_path="logs/1990_4f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4t_test2 \
# configs.wights_path="logs/1990_4t/0/checkpoints/epoch_009.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \



# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_4t_test3 \
# configs.wights_path="logs/1990_4t/0/checkpoints/epoch_009.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \






# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_5t \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_6t \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \







# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1990_8t \
# configs.wights_path="logs/1990_8f/0/checkpoints/epoch_020.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6r \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \





# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# train=False \
# task_name=1980_8t \
# configs.dataset_loader=fast \
# configs.wights_path="logs/1980/0/checkpoints/last.ckpt" \
# configs.train_dataset=\'kinetics_train_4,ava_train_5\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2j" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit_only\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=False \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6r2 \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4t2 \
# configs.wights_path="logs/1996/18/checkpoints/epoch_027.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.vit.depth=9 \
# configs.vit.heads=8 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1990_4t3 \
# configs.wights_path="logs/1990_4f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="both" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \




# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1998 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\',\'pose_l2,loca_l1,action_bce\',\'pose_l2,loca_l1,action_BCE\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False,True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1,0.3,0.6 \
# configs.masked=False,True \
# +seed=1,2 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6t2 \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \








# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_6t_test \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.full_gt_supervision=False \
# configs.full_pesudo_supervision=2 \
# configs.test_type="track.fullframe|GT" \
# # trainer.devices=1 \
# # configs.debug=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6t4 \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=2 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f2 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=1 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
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
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=4 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f4 \
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
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f5 \
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
# configs.max_people=5 \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f6 \
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
# configs.max_people=5 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \






# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=1999 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.full_gt_supervision=0,1,224,212,206 \
# configs.full_pesudo_supervision=0,212,218,224,412,418,424 \
# configs.full_pesudo_supervision_c1=0.1 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1999_6f1 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.full_pesudo_supervision=212 \
# configs.full_gt_supervision=224 \






# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=2000 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.full_gt_supervision=0 \
# configs.full_pesudo_supervision=0 \
# configs.full_pesudo_supervision_c1=0.1 \
# configs.vit.dropout=0.3 \
# configs.vit.droppath=0.3 \






# python src/train.py -m \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp \
# task_name=2001 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.full_gt_supervision=0,1,224,212,206 \
# configs.full_pesudo_supervision=0,212,218,224,412,418,424 \
# configs.full_pesudo_supervision_c1=0.1 \
# configs.vit.dropout=0.3 \
# configs.vit.droppath=0.3 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f4_t1 \
# configs.wights_path="logs/1990_6f4/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.loss_type=\'action_bce\' \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f4_t2 \
# configs.wights_path="logs/1999/31/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.loss_type=\'action_bce\' \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.full_gt_supervision=206 \
# configs.full_pesudo_supervision=224 \
# configs.full_pesudo_supervision_c1=0.1 \



# params1=(1 3 5 7 9 12 15 20 30)
# for i in {0..8}; do
# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1990_6f4_t2_${params1[i]} \
# configs.wights_path="logs/1999/31/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.loss_type=\'action_bce\' \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.full_gt_supervision=206 \
# configs.full_pesudo_supervision=224 \
# configs.full_pesudo_supervision_c1=0.1 \
# configs.test_type="track.fullframe|avg.${params1[i]}" 
# done





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f7 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \






# python src/train.py -m \
# trainer=ddp \
# task_name=1999_6f2 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.full_pesudo_supervision=212 \
# configs.full_gt_supervision=224 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f8 \
# configs.wights_path="logs/1990_6f/0/checkpoints/last.ckpt" \
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
# configs.max_people=2 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
# configs.loss_on_others_action=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \












































# python src/train.py -m \
# trainer=ddp \
# task_name=1990_5f8 \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.loss_on_others_action=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \



# python src/train.py -m \
# trainer=ddp \
# task_name=1999_6f3 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=0 \
# configs.full_gt_supervision=0 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1999_6f3_test \
# configs.wights_path="logs/1999_6f3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=0 \
# configs.full_gt_supervision=0 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1999_6f4 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=0 \
# configs.full_gt_supervision=0 \
# configs.ava.sampling_factor=2 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1999_6f5 \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.vit.depth=9 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1990_6f6_t1 \
# configs.wights_path="logs/1990_6f6/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \















# python src/train.py -m \
# trainer=ddp \
# task_name=1991_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=10.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=15 \
# configs.vit.depth=3 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1999_6f3_1 \
# configs.wights_path="logs/1999_6f3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_9,ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=4e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'mvit\' \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=512 \
# configs.extra_feat.relative_pose.en_dim=1280 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=False \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.full_pesudo_supervision=212 \
# configs.full_gt_supervision=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1992_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=10.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=15 \
# configs.vit.depth=3 \
# configs.mixed_training=2 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1993_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=10.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=15 \
# configs.vit.depth=3 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \









# python src/train.py -m \
# trainer=ddp \
# task_name=1994_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
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
# trainer.max_epochs=30 \
# configs.vit.depth=3 \
# configs.mixed_training=2 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1995_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
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
# trainer.max_epochs=30 \
# configs.vit.depth=3 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \








# python src/train.py -m \
# trainer=ddp \
# task_name=1996_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.max_epochs=30 \
# configs.vit.depth=3 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=1991_7f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=3 \
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
# trainer.gradient_clip_val=10.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=15 \
# configs.vit.depth=3 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.loss_on_others_action=False \




# python src/train.py -m \
# trainer=ddp \
# task_name=1997_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# # params1=(8 10 12 15 20 30)
# params1=(7 6 5 4 3 2 1)
# for i in {0..6}; do
# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=1997_5f_test_x${params1[i]}$ \
# configs.wights_path="logs/1997_5f/0/checkpoints/epoch_000.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
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
# configs.test_type="track.fullframe|avg.${params1[i]}" \
# configs.debug=False
# done
# # callbacks.rich_progress_bar.refresh_rate=0 \



# python src/train.py -m \
# trainer=ddp \
# task_name=1998_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1999_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1999_5f_test \
# train=False \
# configs.wights_path="logs/1999_5f/0/checkpoints/epoch_007.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py -m \
# trainer=ddp \
# task_name=1999_5f_test2 \
# train=False \
# configs.wights_path="logs/1999_5f/0/checkpoints/epoch_007.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|avg.6" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=1999_5f_test3 \
# train=False \
# configs.wights_path="logs/1999_5f/0/checkpoints/epoch_007.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|part.6" \
# configs.debug=True \
# trainer.devices=2 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# python src/train.py -m \
# trainer=ddp \
# task_name=1999_5f_test4 \
# train=False \
# configs.wights_path="logs/1999_5f/0/checkpoints/epoch_007.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|part.6" \
# # configs.debug=True \
# # trainer.devices=2 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2000_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \








# python src/train.py -m \
# trainer=ddp \
# task_name=2001_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|avg.6" \
# +trainer.val_check_interval=1000 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2001_5f_test2 \
# configs.wights_path="logs/1997_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|avg.6" \
# +trainer.val_check_interval=1000 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2002_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|avg.6" \
# +trainer.val_check_interval=1000 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \








# python src/train.py -m \
# trainer=ddp \
# task_name=2003_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=3 \
# configs.test_type="track.fullframe|avg.6" \
# +trainer.val_check_interval=1000 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




















































# python src/train.py -m \
# trainer=ddp \
# task_name=2004_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2005_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2006_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2007_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2008_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \








# python src/train.py -m \
# trainer=ddp \
# task_name=2009_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2009_7f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \



# python src/train.py -m \
# trainer=ddp \
# task_name=2010_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=5 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py -m \
# trainer=ddp \
# task_name=2011_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=5 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2012_5f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=6 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# # params1=(8 10 12 15 20 30)
# params1=(6 7 8 5 9 4 10 3 2 1 11 12 13 14 15)
# for i in {0..14}; do
# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_5f_test_y${params1[i]}$ \
# configs.wights_path="logs/2007_5f/0/checkpoints/epoch_009.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_10\' \
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
# configs.test_type="track.fullframe|avg.${params1[i]}" \
# configs.debug=False
# done
# # callbacks.rich_progress_bar.refresh_rate=0 \









# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2020_7f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=10 \
# configs.test_type="track.fullframe|avg.6" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \





# # train=False \
# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2021_7f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_hug\' \
# configs.test_dataset=\'ava_train_hug\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=10 \
# configs.test_type="track.fullframe|avg.6" \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.render.enable=True \
# configs.render.num_videos=10000 \
# trainer.devices=1 \
# configs.compute_map=False \
# # configs.debug=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \




# python src/train.py -m \
# trainer=ddp \
# task_name=2022_7f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=2 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=0 \
# configs.test_type="track.fullframe|avg.6" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2030_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=128 \
# configs.extra_feat.mvit.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=64 \
# configs.extra_feat.relative_pose.en_dim=128 \
# configs.use_relative_pose=True \
# configs.in_feat=128 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.action_space="ava,kinetics" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2031_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_11,ava_train_11\' \
# configs.test_dataset=\'ava_val_11\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mae\' \
# configs.extra_feat.mae.mid_dim=768 \
# configs.extra_feat.mae.en_dim=768 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=256 \
# configs.extra_feat.relative_pose.en_dim=1024 \
# configs.use_relative_pose=True \
# configs.in_feat=1024 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.action_space="ava" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2032_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,vitpose\' \
# configs.extra_feat.mvit.mid_dim=256 \
# configs.extra_feat.mvit.en_dim=32 \
# configs.extra_feat.vitpose.mid_dim=64 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=256 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=256 \
# configs.extra_feat.relative_pose.en_dim=288 \
# configs.use_relative_pose=True \
# configs.in_feat=288 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.action_space="ava" \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \







# python src/train.py -m \
# trainer=ddp \
# task_name=2033_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,vitpose\' \
# configs.extra_feat.mvit.mid_dim=512 \
# configs.extra_feat.mvit.en_dim=256 \
# configs.extra_feat.vitpose.mid_dim=64 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=256 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=256 \
# configs.extra_feat.relative_pose.en_dim=512 \
# configs.use_relative_pose=True \
# configs.in_feat=512 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.action_space="ava" \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2034_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,vitpose\' \
# configs.extra_feat.mvit.mid_dim=256 \
# configs.extra_feat.mvit.en_dim=32 \
# configs.extra_feat.vitpose.mid_dim=64 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=256 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=256 \
# configs.extra_feat.relative_pose.en_dim=288 \
# configs.use_relative_pose=True \
# configs.in_feat=288 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.action_space="ava" \
# configs.masked_mvit=True \
# configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \








# python src/train.py -m \
# trainer=ddp \
# task_name=2035_5f \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2l" \
# configs.ava.gt_type="all" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,vitpose\' \
# configs.extra_feat.mvit.mid_dim=512 \
# configs.extra_feat.mvit.en_dim=256 \
# configs.extra_feat.vitpose.mid_dim=64 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=256 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=256 \
# configs.extra_feat.relative_pose.en_dim=512 \
# configs.use_relative_pose=True \
# configs.in_feat=512 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.action_space="ava" \
# configs.masked_mvit=True \
# configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.vit.droppath=0.1 \
# # configs.layer_decay=0.9 \









# python src/train.py -m \
# trainer=ddp \
# task_name=1990_20f \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
# configs.test_dataset=\'ava_val_8\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
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
# configs.masked_mvit=True \









# python src/train.py -m \
# trainer=ddp \
# task_name=1990_21f \
# configs.wights_path="logs/1990_20f/0/checkpoints/epoch_018.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-4 \
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
# configs.masked_mvit=True \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \







# python src/train.py -m \
# trainer=ddp \
# task_name=2007_21f \
# configs.wights_path="logs/1990_20f/0/checkpoints/epoch_018.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# python src/train.py -m \
# trainer=ddp \
# task_name=2007_22f \
# configs.wights_path="logs/1990_20f/0/checkpoints/epoch_018.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9,avaK_train_7\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \







# python src/train.py -m \
# trainer=ddp \
# task_name=2007_23f \
# configs.wights_path="logs/2007_21f/0/checkpoints/epoch_008.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2007_24f \
# configs.wights_path="logs/2007_21f/0/checkpoints/epoch_008.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_test1 \
# configs.wights_path="logs/2007_24f/0/checkpoints/epoch_005.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_render100\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.render.enable=True \
# configs.render.num_videos=10000 \
# configs.render.vis_action_label="all" \
# trainer.devices=1 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2007_25f \
# configs.wights_path="logs/2007_21f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2007_26f \
# configs.wights_path="logs/2034_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit,vitpose\' \
# configs.extra_feat.mvit.mid_dim=256 \
# configs.extra_feat.mvit.en_dim=32 \
# configs.extra_feat.vitpose.mid_dim=64 \
# configs.extra_feat.vitpose.en_dim=128 \
# configs.extra_feat.pose_shape.mid_dim=256 \
# configs.extra_feat.pose_shape.en_dim=128 \
# configs.extra_feat.relative_pose.mid_dim=256 \
# configs.extra_feat.relative_pose.en_dim=288 \
# configs.use_relative_pose=True \
# configs.in_feat=288 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \




# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_test \
# configs.wights_path="logs/2007_24f/0/checkpoints/epoch_005.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_10\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# trainer.devices=1 \
# # configs.debug=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \





# python src/train.py -m \
# trainer=ddp \
# task_name=2007_7t \
# configs.wights_path="logs/1990_5f/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.masked_mvit=True \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.loss_on_others_action=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2007_8t \
# configs.wights_path="logs/2007_7t/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.masked_mvit=False \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.loss_on_others_action=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \








# python src/train.py -m \
# trainer=ddp \
# task_name=2007_9t \
# configs.wights_path="logs/2007_7t/0/checkpoints/epoch_005.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# trainer.gradient_clip_val=200.0 \
# configs.mask_ratio=0.1 \
# configs.masked=False \
# configs.masked_mvit=False \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.loss_on_others_action=False \
# # configs.debug=True \
# # trainer.devices=1 \
# # callbacks.rich_progress_bar.refresh_rate=0 \









# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_final_1 \
# configs.wights_path="logs/2007_24f/0/checkpoints/epoch_005.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_render_good\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.render.enable=True \
# configs.render.num_videos=10000 \
# configs.render.vis_action_label="all" \
# trainer.devices=1 \







# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_final_2 \
# configs.wights_path="logs/2007_7t/0/checkpoints/epoch_005.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_render_good\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=5 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=1 \
# configs.test_num_workers=1 \
# trainer.accumulate_grad_batches=8 \
# configs.render.enable=True \
# configs.render.num_videos=55 \
# configs.render.vis_action_label="all" \
# trainer.devices=1 \
# # configs.debug=True \
# # callbacks.rich_progress_bar.refresh_rate=0 \










# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_final_3 \
# configs.wights_path="logs/1990_4p0/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_render_good\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="all" \
# configs.ava.distil_type="both_bce" \
# configs.lr=1e-3 \
# configs.mask_type_test="zero" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.extra_feat.enable=\'\' \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.extra_feat.relative_pose.mid_dim=128 \
# configs.extra_feat.relative_pose.en_dim=256 \
# configs.use_relative_pose=True \
# configs.use_optimized_pose=True \
# configs.in_feat=256 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.2 \
# configs.masked=False \
# configs.render.enable=True \
# configs.render.num_videos=55 \
# configs.render.vis_action_label="all" \
# trainer.devices=1 \



# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_final_4 \
# configs.wights_path="logs/1990_4p3/0/checkpoints/last.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'kinetics_train_7,ava_train_7,avaK_train_7\' \
# configs.test_dataset=\'ava_val_render_good\' \
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
# trainer.devices=1 \






# python src/train.py -m \
# trainer=ddp \
# task_name=2007_24f1 \
# configs.wights_path="logs/2007_21f/0/checkpoints/epoch_008.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'ava_val_9\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# trainer.devices=1 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# # configs.debug=True \






# python src/train.py -m \
# trainer=ddp \
# train=False \
# task_name=2007_24f_final_youtube \
# configs.wights_path="logs/2007_24f/0/checkpoints/epoch_005.ckpt" \
# configs.dataset_loader=fast \
# configs.train_dataset=\'ava_train_9\' \
# configs.test_dataset=\'youtube\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2k" \
# configs.ava.gt_type="gt" \
# configs.lr=1e-4 \
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
# trainer.max_epochs=30 \
# configs.mixed_training=4 \
# configs.test_type="track.fullframe|avg.6" \
# configs.vit.droppath=0.1 \
# configs.layer_decay=0.9 \
# configs.load_strict=False \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=4 \
# configs.test_num_workers=4 \
# trainer.accumulate_grad_batches=8 \
# configs.render.enable=True \
# configs.render.num_videos=10000 \
# configs.render.vis_action_label="all" \
# trainer.devices=1 \



