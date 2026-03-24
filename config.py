"""Configuration for MiniCLIP training and evaluation."""

import argparse


class CFG:
    # Data paths (default to Kaggle paths, override via CLI)
    data_path = "data/flickr30k_images"
    caption_path = "data/results.csv"

    # Model settings
    imageModel = "convnext_small"
    textModel = "distilbert-base-uncased"
    textTokenizer = "distilbert-base-uncased"
    pre_trained = True
    trainable = True

    # Training settings
    batch_size = 32
    max_length = 77
    epochs = 5
    lr = 1e-4
    val_ratio = 0.2
    num_workers = 2

    # Embedding dimensions
    img_embed_size = 768
    text_embed_size = 768
    proj_embed_size = 512

    # Output
    save_path = "model_weights.pth"


def get_args():
    """Parse CLI arguments to override CFG defaults."""
    parser = argparse.ArgumentParser(description="MiniCLIP - CLIP-style model on Flickr30k")

    parser.add_argument("--data_path", type=str, default=CFG.data_path, help="Path to image directory")
    parser.add_argument("--caption_path", type=str, default=CFG.caption_path, help="Path to captions CSV")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--val_ratio", type=float, default=CFG.val_ratio)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    parser.add_argument("--save_path", type=str, default=CFG.save_path)
    parser.add_argument("--image_model", type=str, default=CFG.imageModel)
    parser.add_argument("--text_model", type=str, default=CFG.textModel)

    args = parser.parse_args()

    # Update CFG with parsed arguments
    CFG.data_path = args.data_path
    CFG.caption_path = args.caption_path
    CFG.batch_size = args.batch_size
    CFG.epochs = args.epochs
    CFG.lr = args.lr
    CFG.val_ratio = args.val_ratio
    CFG.num_workers = args.num_workers
    CFG.save_path = args.save_path
    CFG.imageModel = args.image_model
    CFG.textModel = args.text_model
    CFG.textTokenizer = args.text_model

    return args
