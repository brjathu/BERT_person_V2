#!/bin/bash

# source ~/.zshrc
# conda activate HUMAN_BERT_2


python src/train.py -m \
trainer=ddp \
task_name=3000_P1 \
configs.dataset_loader=fast \
configs.train_dataset=\'kinetics_train_7,ava_train_7\' \
configs.test_dataset=\'ava_val_8\' \
configs.ava.predict_valid=True \
configs.bottle_neck="conv2k" \
configs.ava.gt_type="all" \
configs.ava.distil_type="both_bce" \
configs.lr=4e-3 \
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


