#!/usr/bin/env python3
"""
extract_diverse_frames_v2.py

Pipeline per MP4:
1) Extract candidate frames at a chosen FPS
2) Downscale to 720p (keep aspect ratio)
3) Remove near-duplicates (perceptual hash)
4) Embed with CLIP
5) KMeans cluster and select N representative frames
6) Save exactly N frames (if possible) into output/<video_name>/

Notes:
- Requires ffmpeg? No, uses OpenCV for decode.
- Duplicate removal uses imagehash (pHash). Install dependencies below.
"""

import os
import math
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import clip
from sklearn.cluster import KMeans

import imagehash


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resize_to_720p_keep_ar(frame_bgr: np.ndarray, target_h: int = 720) -> np.ndarray:
    """Resize frame to target height while preserving aspect ratio."""
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return frame_bgr
    if h == target_h:
        return frame_bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def safe_interval(video_fps: float, sample_fps: float) -> int:
    """Compute a safe frame interval for sampling."""
    if video_fps <= 0 or math.isnan(video_fps):
        # Fallback if fps can't be read
        video_fps = 30.0
    if sample_fps <= 0:
        sample_fps = 1.0
    interval = int(round(video_fps / sample_fps))
    return max(1, interval)


def phash_image(pil_img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(pil_img)


def is_near_duplicate(curr_hash, prev_hash, max_hamming: int) -> bool:
    """Return True if hashes are within max_hamming distance."""
    if prev_hash is None:
        return False
    return (curr_hash - prev_hash) <= max_hamming


# -----------------------------
# Step 1-3: Extract + downscale + dedupe
# -----------------------------

def extract_candidates(
    video_path: Path,
    candidates_dir: Path,
    sample_fps: float,
    target_h: int,
    dedupe_hamming: int,
    max_candidates: int | None,
) -> list[Path]:
    """
    Extract candidate frames sampled at sample_fps, resized to target_h, and deduped via pHash.
    Dedupe is applied sequentially (good for removing near-identical adjacent frames).
    """
    ensure_dir(candidates_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = safe_interval(video_fps, sample_fps)

    saved_paths: list[Path] = []
    prev_hash = None

    frame_idx = 0
    saved_idx = 0

    # Rough progress estimate (may be 0 if CAP_PROP_FRAME_COUNT unavailable)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Extract+dedupe {video_path.name}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                frame = resize_to_720p_keep_ar(frame, target_h=target_h)

                # Compute pHash for dedupe decision
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                curr_hash = phash_image(pil_img)

                if not is_near_duplicate(curr_hash, prev_hash, dedupe_hamming):
                    out_path = candidates_dir / f"cand_{saved_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    saved_paths.append(out_path)
                    saved_idx += 1
                    prev_hash = curr_hash

                    if max_candidates is not None and len(saved_paths) >= max_candidates:
                        break

            frame_idx += 1
            if pbar.total is not None:
                pbar.update(1)
    finally:
        pbar.close()
        cap.release()

    return saved_paths


# -----------------------------
# Step 4: CLIP embeddings
# -----------------------------

def compute_clip_embeddings(
    image_paths: list[Path],
    device: str,
    model_name: str = "ViT-B/32",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Returns embeddings array shape [N, D] for each image.
    """
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = preprocess(Image.open(p).convert("RGB"))
                imgs.append(img)
            batch = torch.stack(imgs, dim=0).to(device)

            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize for kmeans stability
            all_embs.append(emb.detach().cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# -----------------------------
# Step 5: Cluster + select
# -----------------------------

def select_representatives_kmeans(
    image_paths: list[Path],
    embeddings: np.ndarray,
    n_select: int,
    random_state: int = 0,
) -> list[Path]:
    """
    KMeans with k=n_select. Pick closest image to each cluster center.
    """
    n = len(image_paths)
    if n == 0:
        return []
    if n_select >= n:
        return image_paths[:]  # can't select more than we have

    km = KMeans(n_clusters=n_select, random_state=random_state, n_init="auto")
    km.fit(embeddings)

    centers = km.cluster_centers_
    labels = km.labels_

    selected = []
    for k in range(n_select):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue
        cluster_embs = embeddings[idxs]
        center = centers[k]
        dists = np.linalg.norm(cluster_embs - center, axis=1)
        chosen_local = int(np.argmin(dists))
        chosen_idx = int(idxs[chosen_local])
        selected.append(image_paths[chosen_idx])

    # In rare cases KMeans can yield fewer clusters populated than k (should be uncommon).
    # If that happens, top up with farthest-from-selected greedy to reach n_select.
    if len(selected) < n_select:
        selected_set = set(selected)
        remaining = [p for p in image_paths if p not in selected_set]
        # Greedy: pick points farthest from current selected set mean
        sel_embs = embeddings[[image_paths.index(p) for p in selected]]
        mean_sel = sel_embs.mean(axis=0)
        rem_embs = embeddings[[image_paths.index(p) for p in remaining]]
        d = np.linalg.norm(rem_embs - mean_sel, axis=1)
        order = np.argsort(-d)
        for j in order:
            selected.append(remaining[int(j)])
            if len(selected) >= n_select:
                break

    return selected[:n_select]


# -----------------------------
# Step 6: Save outputs
# -----------------------------

def save_selected(selected_paths: list[Path], out_dir: Path, prefix: str) -> None:
    ensure_dir(out_dir)
    for i, p in enumerate(selected_paths):
        img = cv2.imread(str(p))
        out_path = out_dir / f"{prefix}_{i:04d}.jpg"
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def process_one_video(
    video_path: Path,
    output_root: Path,
    frames: int,
    sample_fps: float,
    target_h: int,
    dedupe_hamming: int,
    max_candidates: int | None,
    clip_model: str,
    batch_size: int,
    keep_temps: bool,
) -> None:
    name = video_path.stem
    out_dir = output_root / name
    temp_dir = output_root / f"{name}__temp"
    candidates_dir = temp_dir / "candidates"

    ensure_dir(output_root)
    ensure_dir(temp_dir)

    # 1-3
    candidates = extract_candidates(
        video_path=video_path,
        candidates_dir=candidates_dir,
        sample_fps=sample_fps,
        target_h=target_h,
        dedupe_hamming=dedupe_hamming,
        max_candidates=max_candidates,
    )

    if len(candidates) == 0:
        print(f"[WARN] No candidates extracted for {video_path.name}")
        return

    # If we don't have enough candidates, just save what we have
    if len(candidates) <= frames:
        print(f"[WARN] Only {len(candidates)} candidates after dedupe; saving all.")
        save_selected(candidates, out_dir, prefix=name)
        if not keep_temps:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = compute_clip_embeddings(
        image_paths=candidates,
        device=device,
        model_name=clip_model,
        batch_size=batch_size,
    )

    # 5
    selected = select_representatives_kmeans(
        image_paths=candidates,
        embeddings=embeddings,
        n_select=frames,
    )

    # 6
    save_selected(selected, out_dir, prefix=name)
    print(f"Saved {len(selected)} diverse 720p frames to: {out_dir}")

    if not keep_temps:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Extract 200 diverse 720p frames per MP4 with dedupe + CLIP + KMeans.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .mp4 files")
    parser.add_argument("--output_dir", required=True, help="Directory to write output frame folders")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to output per video")
    parser.add_argument("--sample_fps", type=float, default=3.0, help="Candidate extraction rate (fps). Higher = more novelty candidates.")
    parser.add_argument("--target_h", type=int, default=720, help="Resize extracted frames to this height (aspect ratio preserved)")
    parser.add_argument("--dedupe_hamming", type=int, default=6, help="pHash hamming distance threshold for near-duplicate removal (lower=more aggressive)")
    parser.add_argument("--max_candidates", type=int, default=12000, help="Cap candidate frames per video (keeps runtime bounded). Use 0 for no cap.")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=64, help="CLIP embedding batch size (reduce if GPU OOM)")
    parser.add_argument("--keep_temps", action="store_true", help="Keep temporary candidate frames")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    max_cand = None if args.max_candidates == 0 else args.max_candidates

    videos = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".mp4"])
    if not videos:
        raise SystemExit(f"No .mp4 files found in {in_dir}")

    for v in videos:
        print(f"\n=== Processing {v.name} ===")
        process_one_video(
            video_path=v,
            output_root=out_dir,
            frames=args.frames,
            sample_fps=args.sample_fps,
            target_h=args.target_h,
            dedupe_hamming=args.dedupe_hamming,
            max_candidates=max_cand,
            clip_model=args.clip_model,
            batch_size=args.batch_size,
            keep_temps=args.keep_temps,
        )


if __name__ == "__main__":
    main()