# Doduo: Dense Visual Correspondence from Unsupervised Semantic-Aware Flow

[Zhenyu Jiang](http://zhenyujiang.me), [Hanwen Jiang](https://hwjiang1510.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)

[Project](https://ut-austin-rpl.github.io/Doduo/) | arxiv | [Huggingface Model](https://huggingface.co/stevetod/doduo)

## News

**`2023-09-26`**: Initial code release.

## Huggingface Model

We provide a pre-trained version of the Doduo model on the Huggingface model hub. To use this model, run the following Python command:

```python
from transformers import AutoModel
from PIL import Image
model = AutoModel.from_pretrained("stevetod/doduo", trust_remote_code=True)
frame_src = Image.open("path/to/src/frame.png")
frame_dst = Image.open("path/to/dst/frame.png")
flow = model(frame_src, frame_dst)
```

## Installation

1. Create a conda environment and install the necessary packages.
   You can modify the `pytorch` and `cuda` version in the `env.yaml` file.

```bash
conda env create -f env.yaml
```

2. The data path is stored in `.env`. Run `cp .env.example .env` command to create an `.env` file. You can modify this file to change your data path.

## Data and Pretrained Model

### Training

We use frames from Youtube VOS dataset for training. Download the data from [this source](https://youtube-vos.org/dataset/vis/).

Note: We use [Mask2Former](https://github.com/facebookresearch/Mask2Former) to generate instance masks for visible region discovery. You can find the predicted masks [here](https://utexas.box.com/s/201u9q9ldstfsn3xe5nh09x2emnvmp7k). After downloading, unzip the file and place it in the `Youtube-VOS/train/` directory.

### Testing

#### Point Correspondence

We evaluate point correspondence on DAVIS val set from TAP-Vid dataset. Please download the data from [here](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip).

### Pretrained Model

You can download the pretrained model using [this link](https://utexas.box.com/s/tbkm8ec7oi41iedz1n23kr65fsjfad1a).

## Demos

We provide two demonstration notebooks for Doduo:

1. [Visualizing correspondence with any local checkpoint](./notebooks/eg_demo_correspondence.ipynb): Make sure you have installed the necessary environment before you initiate this notebook.
2. [Visualizing correspondence with the Huggingface model](./notebooks/eg_demo_correspondence_huggingface.ipynb): No environment installation is required to initiate this notebook.

## Training

You can use the following Python commands to start training the model:

```Python
# single GPU debug
python src/train.py model.mixed_precision=True experiment=doduo_train debug=fdr

# multiple GPUs + wandb logging
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 src/train.py model.mixed_precision=True experiment=doduo_train logger=wandb_csv
```

## Testing

Apply the following Python command, replacing "/path/to/ckpt" with your specific path:

```Python
python src/eval.py experiment=doduo_train ckpt_path=/path/to/ckpt
```

## Related Repositories

1. Our code is based on this fantastic template [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

2. We use [Unimatch](https://github.com/autonomousvision/unimatch) as our backbone.

## Citing

```
@inproceedings{jiang2023doduo,
   title={Doduo: Dense Visual Correspondence from Unsupervised Semantic-Aware Flow},
   author={Jiang, Zhenyu and Jiang, Hanwen and Zhu, Yuke},
   booktitle={TODO},
   year={2023}
}
```
