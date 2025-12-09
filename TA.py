#!/usr/bin/env python3

import re
import json
import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def split_blocks(text: str):
    # Split by blank line(s) between robot entries
    # Ensure consecutive blank lines treated as one split
    blocks = re.split(r"\n\s*\n+", text.strip())
    return [b.strip() for b in blocks if b.strip()]

def parse_block(block: str):
    # Returns dict with fields: id, name, type, size, capabilities(list), raw_text
    robot = {"raw_text": block, "capabilities": []}

    id_m = re.search(r"Robot ID:\s*(\d+)", block)
    robot["id"] = int(id_m.group(1)) if id_m else None

    name_m = re.search(r'Robot Name:\s*(.*)', block)
    robot["name"] = name_m.group(1).strip() if name_m else None

    type_m = re.search(r"Robot Type:\s*(.*)", block)
    robot["type"] = type_m.group(1).strip() if type_m else None

    size_m = re.search(r"Robot Size:\s*(.*)", block)
    robot["size"] = size_m.group(1).strip() if size_m else None

    caps = re.findall(r"-\s*(.*)", block)
    robot["capabilities"] = [c.strip() for c in caps]

    return robot

def load_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    return model

def build_input_text(robot: dict):
    # Create a single text blob for embedding; adjust as desired
    parts = [
        f"Robot Type: {robot.get('type')}",
        f"Robot Size: {robot.get('size')}",
        "Capabilities:",
    ]
    parts += [f"- {c}" for c in robot.get("capabilities", [])]
    return "\n".join(parts)

def encode(model, texts, batch_size=8, normalize_output=True):
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    emb = np.array(embeddings, dtype=np.float32)
    if normalize_output:
        emb = normalize(emb, norm='l2', axis=1)
    return emb

def save_json(robots, path: Path):
    # remove raw_text if you want smaller file; keep for traceability
    out = []
    for r in robots:
        out.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "type": r.get("type"),
            "size": r.get("size"),
            "capabilities": r.get("capabilities"),
            "vector": r.get("vector")
        })
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON to {path}")

def save_npy(emb: np.ndarray, path: Path):
    np.save(path, emb)
    print(f"Saved NumPy .npy to {path}")

def cosine_similarity_matrix(emb: np.ndarray):
    # embeddings expected normalized
    return np.dot(emb, emb.T)


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Multi Robot Task Allocation")
    parser.add_argument("--input", type=str, default="robots.txt", help="Input robots.txt path")
    parser.add_argument("--out-json", type=str, default="robot_vectors.json", help="Output JSON file")
    parser.add_argument("--out-npy", type=str, default="robot_vectors.npy", help="Output numpy .npy file")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="SentenceTransformer model")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for encoding") # 임베딩 시, 한 번에 처리하는 문장 수
    args = parser.parse_args()

    # robot
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    raw = input_path.read_text(encoding="utf-8")
    blocks = split_blocks(raw)
    robots = [parse_block(b) for b in blocks]
    
    texts = [build_input_text(r) for r in robots]    
    print(f"Encoding {len(texts)} robots in batches (batch_size={args.batch}) ...")
    
    # task
    task = input("Task : ")
    task = [task]
    
    print(task)
    

    model = load_model(args.model)
    emb_robot = encode(model, texts, batch_size=args.batch, normalize_output=True)
    emb_task = encode(model, task, batch_size=args.batch, normalize_output=True)
    
    # attach to robots
    for i, r in enumerate(robots):
        r["vector"] = emb_robot[i].tolist()

    # save_json(robots, Path(args.out_json))
    # save_npy(emb, Path(args.out_npy))

    # # Optional: print cosine similarity matrix
    # sim = cosine_similarity_matrix(emb)
    # print("\nCosine similarity matrix (rounded):")
    # with np.printoptions(precision=3, suppress=True):
    #     print(sim)
    
    # ----------------------------
    # Robot-task similarity
    # ----------------------------
    print("\nComputing similarities...")

    # emb_robot: (N, 768)
    # emb_task: (1, 768)
    task_vec = emb_task[0]                      # (768,)
    sims = np.dot(emb_robot, task_vec)          # (N,)

    # 가장 유사한 로봇 선택
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    best_robot = robots[best_idx]

    print("\n=== Best Robot for the Task ===")
    print(f"Robot ID   : {best_robot.get('id')}")
    print(f"Robot Name : {best_robot.get('name')}")
    print(f"Similarity : {best_score:.4f}")
    print("Capabilities:")
    for cap in best_robot.get("capabilities", []):
        print(f" - {cap}")


if __name__ == "__main__":
    main()
