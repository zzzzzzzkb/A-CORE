import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import torch
import open_clip
from sklearn.cluster import KMeans


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


def load_topic_queries(jsonl_path: Path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    flat_queries = []
    meta = []
    for topic in records:
        topicid = str(topic.get("topicid", ""))
        topic_content = topic.get("topic_content", "")
        for q in topic.get("queries", []):
            q = (q or "").strip()
            if not q:
                continue
            flat_queries.append(q)
            meta.append({
                "topicid": topicid,
                "topic_content": topic_content,
                "query": q,
            })
    return records, flat_queries, meta


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

    x = np.vstack(vectors)
    return x


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


def main():
    parser = argparse.ArgumentParser(description="Encode topic queries via CLIP and export s1 files")
    parser.add_argument("--input", default="topic_queries.jsonl", help="Input JSONL topic-query file")
    parser.add_argument("--outdir", default="clip_topic_vectors", help="Output folder")
    parser.add_argument("--prefix", default="s1", help="Output file prefix")
    parser.add_argument("--clusters", type=int, default=50, help="Number of clusters")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for CLIP encoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    topics, queries, meta = load_topic_queries(input_path)
    if not queries:
        raise RuntimeError("No queries found in input JSONL")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vectors = encode_queries_clip(queries, batch_size=args.batch_size, device=device)
    vectors = l2_normalize(vectors)

    n = vectors.shape[0]
    k = max(1, min(args.clusters, n))

    km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    labels = km.fit_predict(vectors).astype(np.int32)

    center_ids = []
    for c in range(k):
        member_idx = np.where(labels == c)[0]
        if member_idx.size == 0:
            center_ids.append(0)
            continue
        centroid = km.cluster_centers_[c]
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = vectors[member_idx] @ centroid
        best_local = int(np.argmax(sims))
        center_ids.append(int(member_idx[best_local]))
    center_ids = np.asarray(center_ids, dtype=np.int32)

    write_fbin(outdir / f"{args.prefix}.xq.fbin", vectors)
    write_ibin(outdir / f"{args.prefix}.labels.ibin", labels.reshape(-1, 1))
    write_ibin(outdir / f"{args.prefix}.centers.ibin", center_ids.reshape(-1, 1))
    build_cluster_csv(outdir / f"{args.prefix}.clusters.csv", vectors, labels, center_ids)

    # Optional manifest for traceability between vector row and topic/query text.
    manifest_path = outdir / f"{args.prefix}.queries.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "topicid", "query"])
        for i, item in enumerate(meta):
            writer.writerow([i, item["topicid"], item["query"]])

    print(json.dumps({
        "input": str(input_path),
        "output_dir": str(outdir),
        "topics": len(topics),
        "queries": len(queries),
        "embedding_dim": int(vectors.shape[1]),
        "clusters": int(k),
        "device": device,
        "files": [
            f"{args.prefix}.xq.fbin",
            f"{args.prefix}.labels.ibin",
            f"{args.prefix}.centers.ibin",
            f"{args.prefix}.clusters.csv",
            f"{args.prefix}.queries.csv",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
