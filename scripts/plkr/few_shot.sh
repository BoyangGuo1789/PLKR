#!/bin/bash

START_TIME=$(date +%s)
DATA="${DATA:-/path/to/dataset/folder}"
TRAINER=PLKR

DATASET=${1:-imagenet}
CFG=${2:-fewshot_25_15_1_ALL_ep25_batch4_4+4ctx}
export CUDA_VISIBLE_DEVICES=${3:-0}
SEED=$4
SHOTS=(1 2 4 8 16)

SEEDS=(1 2 3)
if [ -n "${SEED}" ]; then
    SEEDS=("${SEED}")
fi

for SHOT in "${SHOTS[@]}"
do
    for CUR_SEED in "${SEEDS[@]}"
    do
        DIR=output/fewshot/${DATASET}/${TRAINER}/${CFG}_${SHOT}shots/seed${CUR_SEED}
        if [ -d "$DIR" ]; then
            echo "The results exist at ${DIR}"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${CUR_SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOT}
        fi
    done
done

END_TIME=$(date +%s)

ELAPSED_TIME=$((END_TIME - START_TIME))

ELAPSED_DAYS=$((ELAPSED_TIME / 86400))
ELAPSED_TIME=$((ELAPSED_TIME % 86400))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_TIME=$((ELAPSED_TIME % 3600))
ELAPSED_MINUTES=$((ELAPSED_TIME / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

echo "Total script runtime: ${ELAPSED_DAYS} days ${ELAPSED_HOURS} hours ${ELAPSED_MINUTES} minutes ${ELAPSED_SECONDS} seconds"
