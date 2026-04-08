# PLKR

Official implementation of Prompt Learning with Knowledge Regularization for Pre-trained Vision-Language Models.

<img width="1510" height="650" alt="image" src="https://github.com/user-attachments/assets/b6505f07-280c-4da3-92b4-217495ba0d5a" />


## Overview

- Trainer name: `PLKR`
- Main trainer file: `trainers/plkr.py`
- Main experiment scripts: `scripts/plkr/`
- Open-source release keeps only the 3 primary PLKR configs:
  - `configs/trainers/PLKR/base2new_20_13_05_05_FF_Two_Neighborhood.yaml`
  - `configs/trainers/PLKR/cross_datasets_18_10_01_ALL_E8_3dept_Nei.yaml`
  - `configs/trainers/PLKR/fewshot_25_15_1_ALL_ep25_batch4_4+4ctx.yaml`

## Setup

1. Install PyTorch matching your CUDA version.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch.git
```

## Data Preparation

Set your dataset root:

```bash
export DATA=/path/to/dataset/root
```

`$DATA` should contain the following folders (as used by `datasets/*.py`):

```text
caltech-101
dtd
eurosat
fgvc_aircraft
food-101
imagenet
imagenet-adversarial
imagenet-rendition
imagenet-sketch
imagenetv2
oxford_flowers
oxford_pets
stanford_cars
sun397
ucf101
```

## Main Experiments

The release keeps only these three scripts:

```text
scripts/plkr/base2new_train.sh
scripts/plkr/base2new_test.sh
scripts/plkr/few_shot.sh
```

1. Base-to-New Train (ImageNet base classes)

```bash
bash scripts/plkr/base2new_train.sh imagenet base2new_20_13_05_05_FF_Two_Neighborhood 0
```

2. Base-to-New Test (ImageNet novel classes)

```bash
bash scripts/plkr/base2new_test.sh imagenet base2new_20_13_05_05_FF_Two_Neighborhood 0
```

By default, `base2new_train.sh` and `base2new_test.sh` evaluate seeds `1/2/3` (CoOp setting).  
To run a single seed: `bash scripts/plkr/base2new_train.sh <DATASET> <CONFIG_NAME> <GPU_ID> <SEED>`  
or `bash scripts/plkr/base2new_test.sh <DATASET> <CONFIG_NAME> <GPU_ID> <SEED>`.

3. Few-Shot (ImageNet, shots = 1/2/4/8/16; seeds = 1/2/3)

```bash
bash scripts/plkr/few_shot.sh imagenet fewshot_25_15_1_ALL_ep25_batch4_4+4ctx 0
```

To run a single seed: `bash scripts/plkr/few_shot.sh <DATASET> <CONFIG_NAME> <GPU_ID> <SEED>`.

## Outputs

- Training/evaluation outputs: `output/`
- Result parsers: `parse_test_res.py`, `process_log_base2new.py`, `process_log_cross.py`

## Project Structure

```text
configs/
  datasets/
  trainers/PLKR/
scripts/
  plkr/
trainers/
  plkr.py
train.py
```

## Citation

If you use this codebase, please cite:

- *Prompt Learning With Knowledge Regularization for Pre-Trained Vision-Language Models* (IEEE Transactions on Multimedia, 2026)
- DOI: [10.1109/TMM.2025.3639886](https://doi.org/10.1109/TMM.2025.3639886)

```bibtex
@ARTICLE{11275895,
  author={Guo, Boyang and Li, Liang and Zhang, Jiehua and Sun, Yaoqi and Yan, Chenggang and Sheng, Xichun},
  journal={IEEE Transactions on Multimedia},
  title={Prompt Learning With Knowledge Regularization for Pre-Trained Vision-Language Models},
  year={2026},
  volume={28},
  number={},
  pages={1457-1468},
  keywords={Adaptation models;Computational modeling;Training;Sorting;Ranking (statistics);Optimization;Electronic mail;Visualization;Overfitting;Computational efficiency;Prompt learning;vision-language models;regularization;zero-shot generalization;cross-dataset transfer},
  doi={10.1109/TMM.2025.3639886}
}
```

## License

MIT License. See `LICENSE`.

## Acknowledgements

This codebase builds on open-source CLIP prompt learning infrastructure, including [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and related prior repositories from the community.
