#!/bin/bash

DEVICE=${1}
TRAIN_PATH=${2}
VALID_PATH=${3}
CPU_THREADS=8
CPU_THREADS_NP=4
MOMENTUM=0.9
NESTEROV=1
L2=0.0005

MODEL=${4}
EPOCH=200
BATCH=128
LR=0.1
LR_SCHEDULE="multistep"
MULTISTEP_LR_EPOCH=60,120,160
MULTISTEP_LR_RATE=0.2,0.2,0.2

R_LOWER=${5}
R_UPPER=${6}
RANK_MIN=5
SVD_FC_LAST=1
LOWRANK_LOSS=0.5
SORTING_CRITERION="sv"
SVD_DECOMPOSE=${7}
SVD_BP_EPS=0.99

for SEED in ${8} ${9} ${10} ${11} ${12}
do
NAME=logs_mom${MOMENTUM}-nag${NESTEROV}_l2-${L2}_${MODEL}_ep${EPOCH}_mb${BATCH}_lr${LR}-${LR_SCHEDULE}-${MULTISTEP_LR_EPOCH}-${MULTISTEP_LR_RATE}_r${R_LOWER}-${R_UPPER}_rmin${RANK_MIN}_svdfc${SVD_FC_LAST}_lrloss${LOWRANK_LOSS}_sorttr-${SORTING_CRITERION}_deco-${SVD_DECOMPOSE}_bpeps${SVD_BP_EPS}_${SEED}
python3 -u train.py \
--logdir="./${NAME}/" \
--device=${DEVICE} \
--train_path=${TRAIN_PATH} \
--valid_path=${VALID_PATH} \
--cpu_threads=${CPU_THREADS} \
--cpu_threads_np=${CPU_THREADS_NP} \
--momentum=${MOMENTUM} \
--use_nesterov=${NESTEROV} \
--l2_lambda=${L2} \
--random_seed=${SEED} \
--eval_inter_epochs_train=1 \
--eval_inter_epochs_valid=1 \
--checkpoint_max_keep=1 \
--drop_remainder=0 \
--model=${MODEL} \
--epochs=${EPOCH} \
--batch_size=${BATCH} \
--batch_size_eval=1024 \
--init_lr=${LR} \
--lr_scheduler=${LR_SCHEDULE} \
--multistep_lr_decay_epochs=${MULTISTEP_LR_EPOCH} \
--multistep_lr_decay_rate=${MULTISTEP_LR_RATE} \
--r_lower=${R_LOWER} \
--r_upper=${R_UPPER} \
--rank_min=${RANK_MIN} \
--svd_fc_last=${SVD_FC_LAST} \
--lowrank_loss=${LOWRANK_LOSS} \
--sorting_criterion=${SORTING_CRITERION} \
--svd_decomposition=${SVD_DECOMPOSE} \
--svd_bp_eps=${SVD_BP_EPS} \
2>&1 | tee -a ${NAME}.txt

CHECK_POINT="./${NAME}/model.ckpt-78200"
python3 -u eval.py \
--out_file_path="res_${NAME}.txt" \
--device=${DEVICE} \
--train_path=${TRAIN_PATH} \
--valid_path=${VALID_PATH} \
--checkpoint=${CHECK_POINT} \
--cpu_threads=${CPU_THREADS} \
--cpu_threads_np=${CPU_THREADS_NP} \
--random_seed=${SEED} \
--format="NCHW" \
--model=${MODEL} \
--batch_size_eval=1024 \
--bn_batch_num_divider=1 \
--r_ratio=0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1 \
--rank_min=${RANK_MIN} \
--svd_fc_last=${SVD_FC_LAST} \
--sorting_criterion=${SORTING_CRITERION} \
--svd_decomposition=${SVD_DECOMPOSE} \
2>&1 | tee -a ${NAME}.txt
done
