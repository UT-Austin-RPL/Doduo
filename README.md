# Doduo: Dense Visual Correspondence from Unsupervised Semantic-Aware Flow

[Zhenyu Jiang](http://zhenyujiang.me), [Hanwen Jiang](https://hwjiang1510.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)

## News

**`2023-07-20`**: Initial code release.

## Installation

Create a conda environment and install required packages.

```bash
conda env create -f env.yaml
```
You can change the `pytorch` and `cuda` version in env.yaml.

## Data Preparation

### Training

We use frames from Youtube VOS dataset for training. Please refer to [this website](https://youtube-vos.org/dataset/vis/) for data downloading. We use [Mask2Former](https://github.com/facebookresearch/Mask2Former) to generate instance masks for visible region discovery. We also put the predicted masks [here]().

### Testing

#### Point Correspondence

We evaluate point correspondence on DAVIS val set from TAP-Vid dataset. Please download the data from [here](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip)

## Training

```Python
# single GPU debug
python src/train.py model.mixed_precision=True experiment=doduo_train debug=fdr

# multiple GPUs + wandb logging
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 src/train.py model.mixed_precision=True experiment=doduo_train logger=wandb_csv
```

## Testing

```Python
python src/eval.py experiment=doduo_train ckpt_path=/path/to/ckpt
```