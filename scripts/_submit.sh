#!/bin/bash

#SBATCH  --job-name=PHALP
#SBATCH  --output=/private/home/%u/3D/TENET/out/logs/tenet-eval-%A_%a.out
#SBATCH  --error=/private/home/%u/3D/TENET/out/logs/tenet-eval-%A_%a.err
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=10
##SBATCH  --mem-per-gpu=300
#SBATCH  --time=12:00:00
#SBATCH  --partition=devlab
#SBATCH  --constraint=volta32gb
#SBATCH  --array=0-15
#SBATCH  --signal=B:CONT@60    
##SBATCH  --signal=SIGUSR1@90           
#SBATCH  --requeue

### Run your job.
CASE_NUM=`printf %03d $SLURM_ARRAY_TASK_ID`

source ~/.zshrc

# conda activate slowfast2
# export PYTHONPATH=/private/home/jathushan/3D/BERT_person_hydra:$PYTHONPATH
# export PYTHONPATH=/private/home/jathushan/3D/multi-shot:$PYTHONPATH
# export PYTHONPATH=/private/home/jathushan/3D/multi-shot/optimization:$PYTHONPATH
# python src/utils/ava_mvit_.py \
# --batch_id          $SLURM_ARRAY_TASK_ID \
# --num_of_process    1000 \
# --dataset_slowfast  "ava-val" \
# --add_clip          0 \


# conda activate HUMAN_BERT_3
# export PYTHONPATH=/private/home/jathushan/3D/BERT_person_hydra:$PYTHONPATH
# export PYTHONPATH=/private/home/jathushan/3D/multi-shot:$PYTHONPATH
# export PYTHONPATH=/private/home/jathushan/3D/multi-shot/optimization:$PYTHONPATH
# python src/utils/utils_mvit.py \
# --batch_id          $SLURM_ARRAY_TASK_ID \
# --num_of_process    1000 \
# --dataset_slowfast  "kinetics-train" \
# --add_optimization  1 \




# conda activate HUMAN_BERT_2
# export PYTHONPATH=/private/home/jathushan/3D/BERT_person_hydra:$PYTHONPATH
# export PYTHONPATH=/private/home/jathushan/3D/multi-shot:$PYTHONPATH
# export PYTHONPATH=/private/home/jathushan/3D/multi-shot/optimization:$PYTHONPATH
# python src/utils/utils_mae.py \
# --batch_id          $SLURM_ARRAY_TASK_ID \
# --num_of_process    1000 \
# --dataset_slowfast  "ava-val" \
# --add_optimization  1 \





conda activate HUMAN_BERT_2
python src/train.py -m \
trainer=ddp \
train=False \
task_name=2007_24f_test2 \
configs.wights_path="logs/2007_24f/0/checkpoints/epoch_005.ckpt" \
configs.dataset_loader=fast \
configs.train_dataset=\'ava_train_9\' \
configs.test_dataset=\'ava_val_render100\' \
configs.ava.predict_valid=True \
configs.bottle_neck="conv2k" \
configs.ava.gt_type="gt" \
configs.lr=1e-4 \
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
trainer.max_epochs=30 \
configs.mixed_training=4 \
configs.test_type="track.fullframe|avg.6" \
configs.vit.droppath=0.1 \
configs.layer_decay=0.9 \
configs.load_strict=False \
configs.train_batch_size=4 \
configs.test_batch_size=4 \
configs.train_num_workers=4 \
configs.test_num_workers=4 \
trainer.accumulate_grad_batches=8 \
configs.render.enable=True \
configs.render.num_videos=10000 \
configs.render.vis_action_label="all" \
trainer.devices=1 \
configs.test_batch_id=$SLURM_ARRAY_TASK_ID \
configs.number_of_processes=16 \










# params=(1310 1311 1312 1313 1314)
# values=(1 3 5 7 9)
# python src/train.py \
# task_name=${params[$SLURM_ARRAY_TASK_ID]} \
# configs.train_dataset=\'ava_train,kinetics_train\' \
# configs.test_dataset=\'ava_val\' \
# configs.action_space="ava" \
# configs.bottle_neck="conv" \
# configs.frame_length=125 \
# configs.loss_type=\'pose_l2,loca_l1,action_BCE\' \
# # Submitted batch job 64899204                                                                     




# python src/train.py \
# train=False \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# task_name=1230_test \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'ava_rend_\' \
# configs.action_space="ava" \
# configs.render.walker="PL" \
# configs.frame_length=50 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=8 \
# configs.train_num_workers=8 \
# configs.test_batch_id=$SLURM_ARRAY_TASK_ID \
# configs.number_of_processes=100 \



# python src/train.py \
# task_name=1231_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'ava_rend_\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="SMPL" \
# configs.frame_length=50 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=8 \
# configs.train_num_workers=8 \
# configs.test_batch_id=$SLURM_ARRAY_TASK_ID \
# configs.number_of_processes=100 \


# python src/train.py \
# task_name=1232_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'ava_rend_\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="SMPL+T" \
# configs.render.engine="NMR" \
# configs.frame_length=50 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=8 \
# configs.train_num_workers=8 \
# configs.test_batch_id=$SLURM_ARRAY_TASK_ID \
# configs.number_of_processes=100 \


# python src/train.py \
# task_name=1233_test \
# ckpt_path="logs/1207/checkpoints/last.ckpt" \
# train=False \
# configs.train_dataset=\'kinetics_train\' \
# configs.test_dataset=\'ava_rend_\' \
# configs.action_space="ava" \
# configs.render.enable=True \
# configs.render.walker="SMPL+T+B" \
# configs.render.engine="NMR" \
# configs.frame_length=50 \
# configs.full_seq_render=True \
# configs.render.num_videos=100000 \
# configs.compute_map=False \
# configs.test_num_workers=8 \
# configs.train_num_workers=8 \
# configs.test_batch_id=$SLURM_ARRAY_TASK_ID \
# configs.number_of_processes=100 \


# Submitted batch job 64974554
# Submitted batch job 64974613
# Submitted batch job 64974615
# Submitted batch job 64974618










