"""
Encode WikiAnswers (first 500 rows) with CLIP and save in clip_topic_vectors_topic format.
Each row is treated as its own cluster.
Uses streaming download — only the first ~500 lines are fetched, NOT the full 8 GB file.
"""
import csv
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import requests
import torch
import open_clip


# ── binary I/O helpers ──────────────────────────────────────────────

def write_fbin(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("fbin expects a 2D float32 matrix")
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<ii", n, d))
        f.write(arr.tobytes(order="C"))


def write_ibin(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("ibin expects a 1D/2D int32 matrix")
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<ii", n, d))
        f.write(arr.tobytes(order="C"))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


# ── CLIP encoding ───────────────────────────────────────────────────

@torch.no_grad()
def encode_queries_clip(queries, batch_size: int, device: str):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    vectors = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        vectors.append(feats)

    return np.vstack(vectors)


# ── cluster CSV builder ─────────────────────────────────────────────

def build_cluster_csv(path: Path, vectors_norm: np.ndarray, labels: np.ndarray, center_ids: np.ndarray):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "center_idx", "qpc", "sigma_deg", "theta_min_deg"])
        for c, center_idx in enumerate(center_ids.tolist()):
            member_idx = np.where(labels == c)[0]
            qpc = int(member_idx.size)
            if qpc == 0:
                writer.writerow([c, int(center_idx), 0, 0.0, 0.0])
                continue
            center_vec = vectors_norm[center_idx]
            sims = np.clip(vectors_norm[member_idx] @ center_vec, -1.0, 1.0)
            angles = np.degrees(np.arccos(sims))
            sigma = float(np.std(angles))
            theta_min = float(np.min(angles))
            writer.writerow([c, int(center_idx), qpc, f"{sigma:.6f}", f"{theta_min:.6f}"])


def extract_questions(row: dict) -> list[str]:
    questions = []
    for key in ("query", "pos"):
        val = row.get(key, [])
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    questions.append(item.strip())
        elif isinstance(val, str) and val.strip():
            questions.append(val.strip())
    if not questions:
        for v in row.values():
            if isinstance(v, str) and v.strip():
                questions.append(v.strip())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item.strip():
                        questions.append(item.strip())
    return questions


# ── main ────────────────────────────────────────────────────────────

def main():
    PREFIX = "s1"
    OUTDIR = Path("clip_wikianswers_vectors")
    BATCH_SIZE = 128
    SEED = 42

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Stream-download WikiAnswers JSONL
    print("Streaming WikiAnswers.jsonl.gz from HuggingFace...")
    url = "https://huggingface.co/datasets/embedding-data/WikiAnswers/resolve/main/WikiAnswers.jsonl.gz"

    cluster_queries = []
    total_queries = 0
    buf = ""  # decoded text buffer for partial lines
    dec = zlib.decompressobj(zlib.MAX_WBITS | 16)  # 16 = gzip header
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            raw = dec.decompress(chunk)
            buf += raw.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    row = json.loads(line)
                    qs = extract_questions(row)
                    if len(qs) > 10:
                        cluster_queries.append(qs)
                        total_queries += len(qs)
                        if total_queries >= 10000:
                            break
            if total_queries >= 10000:
                break
        
        # Handle last line without trailing newline if still needed
        if total_queries < 10000 and buf.strip():
            row = json.loads(buf.strip())
            qs = extract_questions(row)
            if len(qs) > 10:
                cluster_queries.append(qs)
                total_queries += len(qs)

    k = len(cluster_queries)  # number of clusters = number of rows
    print(f"Prepared {k} clusters, {total_queries} total queries")
    if k > 0:
        print(f"Cluster 0: {len(cluster_queries[0])} queries")
        print(f"  q0: {cluster_queries[0][0][:100]}")
        if len(cluster_queries[0]) > 1:
            print(f"  q1: {cluster_queries[0][1][:100]}")

    # 3. Flatten all queries, encode with CLIP, then assign cluster labels
    all_queries = []
    labels_list = []
    for c, qs in enumerate(cluster_queries):
        for q in qs:
            all_queries.append(q)
            labels_list.append(c)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(all_queries)} queries with CLIP on {device}...")
    vectors = encode_queries_clip(all_queries, batch_size=BATCH_SIZE, device=device)
    vectors = l2_normalize(vectors)
    n = vectors.shape[0]
    labels = np.array(labels_list, dtype=np.int32)
    print(f"Encoded {n} vectors, dim={vectors.shape[1]}, {k} clusters")

    # 4. Find center for each cluster (closest to centroid)
    center_ids = np.empty(k, dtype=np.int32)
    cluster_sizes = []
    per_cluster_avg_l2 = []
    for c in range(k):
        member_idx = np.where(labels == c)[0]
        cluster_sizes.append(int(member_idx.size))
        members = vectors[member_idx]
        centroid = np.mean(members, axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = members @ centroid
        best_local = int(np.argmax(sims))
        center_ids[c] = int(member_idx[best_local])
        # avg L2 distance from center
        center_vec = vectors[center_ids[c]]
        l2 = np.linalg.norm(members - center_vec, axis=1)
        per_cluster_avg_l2.append(float(np.mean(l2)))

    # 5. Save output
    OUTDIR.mkdir(parents=True, exist_ok=True)

    write_fbin(OUTDIR / f"{PREFIX}.xq.fbin", vectors)
    write_ibin(OUTDIR / f"{PREFIX}.labels.ibin", labels.reshape(-1, 1))
    write_ibin(OUTDIR / f"{PREFIX}.centers.ibin", center_ids.reshape(-1, 1))
    build_cluster_csv(OUTDIR / f"{PREFIX}.clusters.csv", vectors, labels, center_ids)

    # cluster_topic_map
    with (OUTDIR / f"{PREFIX}.cluster_topic_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "topicid"])
        for i in range(k):
            writer.writerow([i, i])

    # queries manifest
    with (OUTDIR / f"{PREFIX}.queries.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "topicid", "query"])
        for i, q in enumerate(all_queries):
            writer.writerow([i, labels_list[i], q])

    # distance_summary.json
    stats = {
        "clusters": int(k),
        "vectors": int(n),
        "cluster_size_min": int(min(cluster_sizes)),
        "cluster_size_max": int(max(cluster_sizes)),
        "cluster_avg_l2_min": float(min(per_cluster_avg_l2)),
        "cluster_avg_l2_max": float(max(per_cluster_avg_l2)),
        "topic_based": True,
    }
    (OUTDIR / f"{PREFIX}.distance_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # distance_stats.json with per-cluster details
    per_cluster_details = []
    for c in range(k):
        member_idx = np.where(labels == c)[0]
        center_vec = vectors[center_ids[c]]
        members = vectors[member_idx]
        sims = np.clip(members @ center_vec, -1.0, 1.0)
        angles = np.degrees(np.arccos(sims))
        l2 = np.linalg.norm(members - center_vec, axis=1)
        cos_dist = 1.0 - sims
        per_cluster_details.append({
            "cluster": c,
            "center_idx": int(center_ids[c]),
            "cluster_size": int(member_idx.size),
            "other_count": int(member_idx.size - 1),
            "avg_l2": float(np.mean(l2)),
            "avg_cos_dist": float(np.mean(cos_dist)),
            "avg_angle_deg": float(np.mean(angles)),
        })

    weighted_l2 = sum(s * l for s, l in zip(cluster_sizes, per_cluster_avg_l2)) / n
    distance_stats = {
        "clusters": int(k),
        "vectors": int(n),
        "cluster_size_min": int(min(cluster_sizes)),
        "cluster_size_max": int(max(cluster_sizes)),
        "weighted_avg_l2": float(weighted_l2),
        "avg_of_cluster_avg_l2": float(np.mean(per_cluster_avg_l2)),
        "avg_cosine_distance": float(np.mean([d["avg_cos_dist"] for d in per_cluster_details])),
        "avg_angle_deg": float(np.mean([d["avg_angle_deg"] for d in per_cluster_details])),
        "cluster_avg_l2_min": float(min(per_cluster_avg_l2)),
        "cluster_avg_l2_max": float(max(per_cluster_avg_l2)),
        "per_cluster": per_cluster_details,
    }
    (OUTDIR / f"{PREFIX}.distance_stats.json").write_text(
        json.dumps(distance_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "output_dir": str(OUTDIR),
        "total_vectors": n,
        "clusters": k,
        "embedding_dim": int(vectors.shape[1]),
        "avg_queries_per_cluster": round(total_queries / k, 1),
        "device": device,
        "files": [
            f"{PREFIX}.xq.fbin",
            f"{PREFIX}.labels.ibin",
            f"{PREFIX}.centers.ibin",
            f"{PREFIX}.clusters.csv",
            f"{PREFIX}.cluster_topic_map.csv",
            f"{PREFIX}.queries.csv",
            f"{PREFIX}.distance_summary.json",
            f"{PREFIX}.distance_stats.json",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
