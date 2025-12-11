import re
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = None

# HF fallback
from transformers import AutoTokenizer, AutoModel
import torch

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

def build_input_text(robot: dict, input_type: int):
    if input_type == 1:
        parts = [
            f"Robot Type: {robot.get('type')}",
            f"Robot Size: {robot.get('size')}",
            "Capabilities:",
        ]
        parts += [f"- {c}" for c in robot.get("capabilities", [])]
        return "\n".join(parts)
    
    elif input_type == 2:
        parts = [
            f"This robot is a {robot.get('type')}.",
            f"Its size is {robot.get('size')}."
        ]
        parts += [f"{c}" for c in robot.get("capabilities", [])]
        return " ".join(parts)

# def load_model(model_name: str):
#     print(f"Loading model: {model_name} ...")
#     model = SentenceTransformer(model_name)
#     return model

def load_model_auto(model_name: str):

    print(f"\033[38;5;208mLoading model: {model_name}\033[0m")

    # 1) SentenceTransformer 시도
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
            return ("st", model)
        except Exception as e:
            print(f"  ST load failed → HF fallback. Reason: {e}")

    # 2) HF fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return ("hf", (tokenizer, model))
    except Exception as e:
        raise RuntimeError(f"Model load failed for {model_name}\nReason: {e}")
    
def hf_mean_pooling(tokenizer, model, texts, batch_size=8, device="cpu"):
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            out = model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)

            sum_vec = (last * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            mean_vec = sum_vec / lengths

            all_embeddings.append(mean_vec.cpu().numpy())

    return np.vstack(all_embeddings)

# def encode(model, texts, batch_size=8, normalize_output=True):
#     embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
#     emb = np.array(embeddings, dtype=np.float32)
#     if normalize_output:
#         emb = normalize(emb, norm='l2', axis=1)
#     return emb

def encode(model_info, texts, batch_size=8, normalize_output=True):
    mode, model_obj = model_info

    if mode == "st":
        model = model_obj
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        emb = np.array(emb, dtype=np.float32)

    else:
        tokenizer, model = model_obj
        emb = hf_mean_pooling(tokenizer, model, texts, batch_size=batch_size)
        emb = np.array(emb, dtype=np.float32)

    if normalize_output:
        emb = normalize(emb, "l2", axis=1)

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
    parser.add_argument("--dir", type=str, default="robots", help="Directory containing robot .txt files")
    # parser.add_argument("--out-json", type=str, default="robot_vectors.json", help="Output JSON file")
    # parser.add_argument("--out-npy", type=str, default="robot_vectors.npy", help="Output numpy .npy file")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for encoding") # 임베딩 시, 한 번에 처리하는 문장 수
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise SystemExit(f"Directory not found: {dir_path}")

    robot_files = sorted(dir_path.glob("*.txt"))
    if not robot_files:
        raise SystemExit(f"No .txt robot files found in: {dir_path}")

    # robot
    robots = []
    for f in robot_files:
        text = f.read_text(encoding="utf-8")
        robot = parse_block(text)
        robots.append(robot)
    
    texts_1 = [build_input_text(r, 1) for r in robots]
    texts_2 = [build_input_text(r, 2) for r in robots]

    models = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",

        "intfloat/e5-base-v2",
        "intfloat/e5-large-v2",

        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",

        "sentence-transformers/gtr-t5-base",
        "sentence-transformers/gtr-t5-large",
        "sentence-transformers/gtr-t5-xl",

        "allenai/specter2_base",

        "sentence-transformers/all-roberta-large-v1",
        "sentence-transformers/sentence-t5-large"
    ]

    all_results = {}

    for model_name in models:
        # model = load_model(model_name)
        model = load_model_auto(model_name)
        emb_robot = encode(model, texts_1, batch_size=args.batch, normalize_output=True)

        # Optional: print cosine similarity matrix
        sim = cosine_similarity_matrix(emb_robot)
        print("Cosine similarity matrix (rounded):")
        with np.printoptions(precision=3, suppress=True):
            print(sim)
        print()



    for model_name in models:
        # model = load_model(model_name)
        model = load_model_auto(model_name)
        emb_robot = encode(model, texts_2, batch_size=args.batch, normalize_output=True)

        # Optional: print cosine similarity matrix
        sim = cosine_similarity_matrix(emb_robot)
        print("Cosine similarity matrix (rounded):")
        with np.printoptions(precision=3, suppress=True):
            print(sim)
        print()


if __name__ == "__main__":
    main()
