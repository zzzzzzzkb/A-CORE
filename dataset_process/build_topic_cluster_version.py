import argparse
import csv
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
    if data.size != n * d:
        raise ValueError(f"Invalid fbin size in {path}: expected {n*d}, got {data.size}")
    return data.reshape(n, d)


def write_ibin(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("ibin expects 1D/2D int32 array")
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<ii", n, d))
        f.write(arr.tobytes(order="C"))


def load_topic_order(topic_jsonl: Path):
    topic_ids = []
    with topic_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            topic_ids.append(str(obj["topicid"]))
    return topic_ids


def load_query_topic_rows(queries_csv: Path):
    rows = []
    with queries_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["idx"]), str(r["topicid"]), r["query"]))
    rows.sort(key=lambda x: x[0])
    return rows


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict topic-based cluster files")
    parser.add_argument("--topic-jsonl", default="topic_queries.jsonl")
    parser.add_argument("--xq", default="clip_topic_vectors/s1.xq.fbin")
    parser.add_argument("--queries-csv", default="clip_topic_vectors/s1.queries.csv")
    parser.add_argument("--outdir", default="clip_topic_vectors_topic")
    parser.add_argument("--prefix", default="s1")
    args = parser.parse_args()

    topic_order = load_topic_order(Path(args.topic_jsonl))
    topic_to_cluster = OrderedDict((tid, i) for i, tid in enumerate(topic_order))

    xq = read_fbin(Path(args.xq))
    rows = load_query_topic_rows(Path(args.queries_csv))
    if xq.shape[0] != len(rows):
        raise ValueError(f"xq rows ({xq.shape[0]}) != queries csv rows ({len(rows)})")

    labels = np.empty((xq.shape[0],), dtype=np.int32)
    for idx, topicid, _ in rows:
        if topicid not in topic_to_cluster:
            raise ValueError(f"Topic {topicid} in queries csv not found in topic jsonl")
        labels[idx] = topic_to_cluster[topicid]

    vecs_norm = l2_normalize(xq)
    k = len(topic_to_cluster)

    centers = np.empty((k,), dtype=np.int32)
    cluster_csv_rows = []
    cluster_sizes = []
    per_cluster_avg_l2 = []

    for topicid, c in topic_to_cluster.items():
        member_idx = np.where(labels == c)[0]
        if member_idx.size == 0:
            centers[c] = 0
            cluster_csv_rows.append((c, 0, 0, 0.0, 0.0))
            cluster_sizes.append(0)
            per_cluster_avg_l2.append(0.0)
            continue

        cluster_sizes.append(int(member_idx.size))
        members = vecs_norm[member_idx]
        centroid = np.mean(members, axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)

        sims_to_centroid = members @ centroid
        best_local = int(np.argmax(sims_to_centroid))
        center_idx = int(member_idx[best_local])
        centers[c] = center_idx

        center_vec = vecs_norm[center_idx]
        sims = np.clip(members @ center_vec, -1.0, 1.0)
        angles = np.degrees(np.arccos(sims))
        sigma = float(np.std(angles))
        theta_min = float(np.min(angles))

        l2 = np.linalg.norm(members - center_vec, axis=1)
        per_cluster_avg_l2.append(float(np.mean(l2)))
        cluster_csv_rows.append((c, center_idx, int(member_idx.size), sigma, theta_min))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Keep vectors unchanged; create a dedicated copy for this strict-topic split version.
    xq_out = outdir / f"{args.prefix}.xq.fbin"
    xq_out.write_bytes(Path(args.xq).read_bytes())

    write_ibin(outdir / f"{args.prefix}.labels.ibin", labels.reshape(-1, 1))
    write_ibin(outdir / f"{args.prefix}.centers.ibin", centers.reshape(-1, 1))

    with (outdir / f"{args.prefix}.clusters.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "center_idx", "qpc", "sigma_deg", "theta_min_deg"])
        for c, center_idx, qpc, sigma, theta_min in cluster_csv_rows:
            writer.writerow([c, center_idx, qpc, f"{sigma:.6f}", f"{theta_min:.6f}"])

    with (outdir / f"{args.prefix}.cluster_topic_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "topicid"])
        for tid, c in topic_to_cluster.items():
            writer.writerow([c, tid])

    stats = {
        "clusters": int(k),
        "vectors": int(xq.shape[0]),
        "cluster_size_min": int(min(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_max": int(max(cluster_sizes)) if cluster_sizes else 0,
        "cluster_avg_l2_min": float(min(per_cluster_avg_l2)) if per_cluster_avg_l2 else 0.0,
        "cluster_avg_l2_max": float(max(per_cluster_avg_l2)) if per_cluster_avg_l2 else 0.0,
        "topic_based": True,
    }
    (outdir / f"{args.prefix}.distance_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "output_dir": str(outdir),
        "clusters": k,
        "vectors": int(xq.shape[0]),
        "files": [
            f"{args.prefix}.xq.fbin",
            f"{args.prefix}.labels.ibin",
            f"{args.prefix}.centers.ibin",
            f"{args.prefix}.clusters.csv",
            f"{args.prefix}.cluster_topic_map.csv",
            f"{args.prefix}.distance_summary.json",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
