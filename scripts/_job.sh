#!/bin/bash

#SBATCH  --job-name=1962
#SBATCH  --output=/private/home/%u/3D/TENET/out/logs/tenet-eval-%A_%a.out
#SBATCH  --error=/private/home/%u/3D/TENET/out/logs/tenet-eval-%A_%a.err
#SBATCH  --gres=gpu:8
#SBATCH  --cpus-per-task=80
#SBATCH  --constraint=volta32gb
#SBATCH  --nodes=1
#SBATCH  --time=72:00:00
#SBATCH  --partition=learnfair
##SBATCH  --array=0-24
#SBATCH  --signal=B:CONT@60    
##SBATCH  --signal=SIGUSR1@90           
#SBATCH  --requeue

### Run your job.
CASE_NUM=`printf %03d $SLURM_ARRAY_TASK_ID`

source ~/.zshrc
conda activate HUMAN_BERT_2



# python src/train.py \
# trainer=ddp \
# task_name=1970d \
# ckpt_path=logs/1970d/checkpoints/last.ckpt \
# configs.train_dataset=\'kinetics_train_4,ava_train_4\' \
# configs.test_dataset=\'ava_val_5\' \
# configs.ava.predict_valid=True \
# configs.bottle_neck="conv2" \
# configs.ava.gt_type="all" \
# configs.frame_length=125 \
# configs.max_people=1 \
# configs.loss_type=\'action_bce\' \
# configs.extra_feat.enable=\'mvit\' \
# configs.extra_feat.mvit.mid_dim=1024 \
# configs.extra_feat.mvit.en_dim=1024 \
# configs.extra_feat.pose_shape.mid_dim=229 \
# configs.extra_feat.pose_shape.en_dim=256 \
# configs.in_feat=1280 \
# configs.weight_decay=5e-2 \
# trainer.gradient_clip_val=2.0 \
# configs.mask_ratio=0.0 \
# configs.masked=False \
# configs.test_type="track.fullframe|" \
# configs.warmup_epochs=10 \
# trainer.max_epochs=100 \
# # +trainer.val_check_interval=1000 \