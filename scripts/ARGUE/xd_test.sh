#!/bin/bash

# custom config
DATA=/workspace/.cache/
TRAINER=ARGUE

DATASET=$1
SEED=$2

CFG=$3
SHOTS=16
LOADEP=25
SUB=all


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/all/train_all/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
DIR=output/all/test_${SUB}/${COMMON_DIR}
python argue_train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}