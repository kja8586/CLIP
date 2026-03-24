"""Evaluation script for MiniCLIP — Recall@K metrics."""

import argparse
from collections import defaultdict

import torch
from torch.amp import autocast
from tqdm import tqdm

from config import CFG
from dataset import get_dataloaders
from models import MiniCLIP


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract image and text embeddings from a dataloader."""
    model.eval()
    all_img_embeds = []
    all_txt_embeds = []
    all_image_ids = []

    for images, texts, image_ids in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        texts = {k: v.to(device) for k, v in texts.items()}

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(texts)

        all_img_embeds.append(img_emb.cpu())
        all_txt_embeds.append(txt_emb.cpu())
        all_image_ids.append(image_ids)

    img_embeds = torch.cat(all_img_embeds, dim=0)  # (N, proj_dim)
    txt_embeds = torch.cat(all_txt_embeds, dim=0)  # (N, proj_dim)
    image_ids = torch.cat(all_image_ids, dim=0)    # (N,)

    return img_embeds, txt_embeds, image_ids


def compute_recall_at_k(similarity, ground_truth, k_values=[1, 5, 10]):
    """Compute Recall@K for given similarity matrix and ground truth."""
    recalls = {}
    num_queries = similarity.shape[0]

    for k in k_values:
        top_k_indices = similarity.topk(k, dim=1).indices  # (num_queries, k)
        hits = 0
        for i in range(num_queries):
            retrieved = set(top_k_indices[i].tolist())
            if retrieved & ground_truth[i]:  # intersection
                hits += 1
        recalls[f"R@{k}"] = hits / num_queries

    return recalls


def evaluate_recall(img_embeds, txt_embeds, image_ids, k_values=[1, 5, 10]):
    """Evaluate Image→Text and Text→Image Recall@K."""
    N = len(image_ids)

    # Build ground truth mappings
    img_id_to_caption_indices = defaultdict(set)
    for cap_idx in range(N):
        img_id = image_ids[cap_idx].item()
        img_id_to_caption_indices[img_id].add(cap_idx)

    seen_images = {}
    i2t_query_indices = []
    i2t_ground_truth = {}

    for cap_idx in range(N):
        img_id = image_ids[cap_idx].item()
        if img_id not in seen_images:
            seen_images[img_id] = len(i2t_query_indices)
            i2t_query_indices.append(cap_idx)
            i2t_ground_truth[seen_images[img_id]] = img_id_to_caption_indices[img_id]

    t2i_ground_truth = {}
    for cap_idx in range(N):
        img_id = image_ids[cap_idx].item()
        t2i_ground_truth[cap_idx] = {seen_images[img_id]}

    # Compute similarities
    unique_img_embeds = img_embeds[i2t_query_indices]
    sim_i2t = unique_img_embeds @ txt_embeds.T  # Image → Text
    sim_t2i = txt_embeds @ unique_img_embeds.T  # Text → Image

    # Compute Recall
    i2t_recall = compute_recall_at_k(sim_i2t, i2t_ground_truth, k_values)
    t2i_recall = compute_recall_at_k(sim_t2i, t2i_ground_truth, k_values)

    return i2t_recall, t2i_recall


def print_results(i2t_recall, t2i_recall):
    """Pretty-print recall results."""
    print("=" * 45)
    print("  Image → Text Retrieval (I2T)")
    print("=" * 45)
    for k, v in i2t_recall.items():
        print(f"  {k}: {v:.4f}  ({v*100:.2f}%)")

    print()
    print("=" * 45)
    print("  Text → Image Retrieval (T2I)")
    print("=" * 45)
    for k, v in t2i_recall.items():
        print(f"  {k}: {v:.4f}  ({v*100:.2f}%)")

    print()
    print("=" * 45)
    print(f"  Mean R@1:  {(i2t_recall['R@1'] + t2i_recall['R@1']) / 2:.4f}")
    print(f"  Mean R@5:  {(i2t_recall['R@5'] + t2i_recall['R@5']) / 2:.4f}")
    print(f"  Mean R@10: {(i2t_recall['R@10'] + t2i_recall['R@10']) / 2:.4f}")
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniCLIP with Recall@K")
    parser.add_argument("--weights", type=str, default="model_weights.pth", help="Path to model weights")
    parser.add_argument("--data_path", type=str, default=CFG.data_path, help="Path to image directory")
    parser.add_argument("--caption_path", type=str, default=CFG.caption_path, help="Path to captions CSV")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    args = parser.parse_args()

    # Update config
    CFG.data_path = args.data_path
    CFG.caption_path = args.caption_path
    CFG.batch_size = args.batch_size
    CFG.num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = MiniCLIP().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    # Data
    _, val_loader, _, _ = get_dataloaders()

    # Extract embeddings and evaluate
    img_embeds, txt_embeds, image_ids = extract_embeddings(model, val_loader, device)

    print(f"Image embeddings: {img_embeds.shape}")
    print(f"Text embeddings:  {txt_embeds.shape}")
    print(f"Unique images:    {image_ids.unique().shape[0]}")
    print()

    i2t_recall, t2i_recall = evaluate_recall(img_embeds, txt_embeds, image_ids)
    print_results(i2t_recall, t2i_recall)


if __name__ == "__main__":
    main()
