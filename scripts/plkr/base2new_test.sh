#!/bin/bash

DATA="${DATA:-/path/to/dataset/folder}"
TRAINER=PLKR

DATASET=${1:-imagenet}
CFG=${2:-base2new_20_13_05_05_FF_Two_Neighborhood}
SEED=$4
export CUDA_VISIBLE_DEVICES=${3:-0}
SHOTS=16
SEEDS=(1 2 3)
if [ -n "${SEED}" ]; then
    SEEDS=("${SEED}")
fi
LOADEP="${LOADEP:-12}"
SUB=new

for SEED in "${SEEDS[@]}"
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    DIR=output/base2new/test_${SUB}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Evaluating model"
        echo "Results are available in ${DIR}. Resuming..."

        python train.py \
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

    else
        echo "Evaluating model"
        echo "Runing the first phase job and save the output to ${DIR}"

        python train.py \
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
    fi
done
