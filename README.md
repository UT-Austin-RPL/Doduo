# Doduo: Dense Visual Correspondence from Unsupervised Semantic-Aware Flow

[Zhenyu Jiang](http://zhenyujiang.me), [Hanwen Jiang](https://hwjiang1510.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)

[Project](https://ut-austin-rpl.github.io/Doduo/) | arxiv | [Huggingface Model](https://huggingface.co/stevetod/doduo)

## News

**`2023-07-20`**: Initial code release.

## Huggingface Model

We host pre-trained Doduo on Huggingface model hub. You can load the pretrained model by running the following command.

```python
from transformers import AutoModel
from PIL import Image
model = AutoModel.from_pretrained("stevetod/doduo", trust_remote_code=True)
frame_src = Image.open("path/to/src/frame.png")
frame_dst = Image.open("path/to/dst/frame.png")
flow = model(frame_src, frame_dst)
```

## Installation

Create a conda environment and install required packages.

```bash
conda env create -f env.yaml
```
You can change the `pytorch` and `cuda` version in env.yaml.

Also, the path to data is stored in `.env`. You can run `cp .env.example .env` and edit the `.env` file to change the path to data.

## Data and Pretrained Model

### Training

We use frames from Youtube VOS dataset for training. Please refer to [this website](https://youtube-vos.org/dataset/vis/) for data downloading. We use [Mask2Former](https://github.com/facebookresearch/Mask2Former) to generate instance masks for visible region discovery. We also put the predicted masks [here](https://utexas.box.com/s/201u9q9ldstfsn3xe5nh09x2emnvmp7k). Please unzip this file and put it under `Youtube-VOS/train/`.

### Testing

#### Point Correspondence

We evaluate point correspondence on DAVIS val set from TAP-Vid dataset. Please download the data from [here](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip)

### Pretrained Model

The pretrained model can be downloaded from this [link](https://utexas.box.com/s/tbkm8ec7oi41iedz1n23kr65fsjfad1a).

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