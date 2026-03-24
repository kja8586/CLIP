# MiniCLIP — CLIP-style Model on Flickr30k

A minimal CLIP (Contrastive Language–Image Pretraining) implementation trained on the [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) dataset.

## Architecture

| Component       | Model                  | Embedding Dim |
|-----------------|------------------------|---------------|
| Image Encoder   | ConvNeXt-Small (timm)  | 768           |
| Text Encoder    | DistilBERT             | 768           |
| Projection Head | Linear → 512-d         | 512           |

Contrastive loss with learnable temperature parameter.

## Project Structure

```
├── config.py              # Hyperparameters and CLI argument parsing
├── dataset.py             # Flickr30k dataset and DataLoader utilities
├── models/
│   ├── image_encoder.py   # Image encoder (ConvNeXt via timm)
│   ├── text_encoder.py    # Text encoder (DistilBERT)
│   └── clip_model.py      # MiniCLIP (combines encoders + projection)
├── train.py               # Training script
├── evaluate.py            # Recall@K evaluation script
├── requirements.txt       # Dependencies
└── notebooks/             # Original Kaggle notebook
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and place it under `data/`:

```
data/
├── flickr30k_images/      # Image files
└── results.csv            # Captions
```

## Training

```bash
python train.py --data_path data/flickr30k_images --caption_path data/results.csv
```

**CLI options:**

| Flag             | Default                  | Description             |
|------------------|--------------------------|-------------------------|
| `--data_path`    | `data/flickr30k_images`  | Path to images          |
| `--caption_path` | `data/results.csv`       | Path to captions CSV    |
| `--batch_size`   | `32`                     | Batch size              |
| `--epochs`       | `5`                      | Number of epochs        |
| `--lr`           | `1e-4`                   | Learning rate           |
| `--save_path`    | `model_weights.pth`      | Output weights path     |

## Evaluation

```bash
python evaluate.py --weights model_weights.pth --data_path data/flickr30k_images --caption_path data/results.csv
```

## Results (Flickr30k validation set)

| Metric   | I2T     | T2I     | Mean    |
|----------|---------|---------|---------|
| R@1      | 27.69%  | 25.24%  | 26.47%  |
| R@5      | 53.99%  | 51.08%  | 52.53%  |
| R@10     | 65.85%  | 63.13%  | 64.49%  |

## Acknowledgements

- [Flickr30k Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
- [CLIP paper](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- [timm](https://github.com/huggingface/pytorch-image-models) / [HuggingFace Transformers](https://github.com/huggingface/transformers)
