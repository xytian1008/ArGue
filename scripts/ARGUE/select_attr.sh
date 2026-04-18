#!/bin/bash

# custom config
DATA=/workspace/.cache/
TRAINER=ARGUE

DATASET=$1
SEED=$2


CFG=$3


DIR=output/attr_selection/${DATASET}/seed${SEED}
python select_attr.py \
--root ${DATA} \
--seed ${SEED} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.SUBSAMPLE_CLASSES all