import os
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class DisjointSetUnion:
    """Disjoint Set Union (DSU) / Union-Find for bipartite graph connected component grouping."""
    def __init__(self):
        self.parent = {}

    def find(self, i: str) -> str:
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Helper to read JSONL files."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_scifact_binary_dataset(data_dir: str = "data/datasets/scifact/data") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads official SciFact dataset, reconstructs evidence text, performs connected-component document grouping,
    and returns 1,295 labeled SUPPORT (1) and CONTRADICT (0) instances.
    """
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    train_path = os.path.join(data_dir, "claims_train.jsonl")
    dev_path = os.path.join(data_dir, "claims_dev.jsonl")

    if not os.path.exists(corpus_path) or not os.path.exists(train_path) or not os.path.exists(dev_path):
        raise FileNotFoundError(f"SciFact dataset files not found in {data_dir}. Please verify download.")

    corpus = {doc["doc_id"]: doc for doc in load_jsonl(corpus_path)}
    train_claims = load_jsonl(train_path)
    dev_claims = load_jsonl(dev_path)
    all_claims = train_claims + dev_claims

    instances = []
    dsu = DisjointSetUnion()

    for c in all_claims:
        c_id = c["id"]
        c_text = c["claim"].strip()
        ev_dict = c.get("evidence", {})
        
        # Connect claim node and evidence doc nodes in graph
        c_node = f"C_{c_id}"
        dsu.find(c_node)

        for doc_id_str, ev_list in ev_dict.items():
            doc_id = int(doc_id_str)
            d_node = f"D_{doc_id}"
            dsu.find(d_node)
            dsu.union(c_node, d_node)

            doc = corpus.get(doc_id)
            if not doc:
                continue

            for ev in ev_list:
                label_str = ev["label"]
                if label_str not in ("SUPPORT", "CONTRADICT"):
                    continue
                label_num = 1 if label_str == "SUPPORT" else 0

                sentences = ev["sentences"]
                abstract = doc.get("abstract", [])
                rationale_text = " ".join([abstract[s] for s in sentences if s < len(abstract)]).strip()

                instances.append({
                    "claim_id": c_id,
                    "claim_text": c_text,
                    "doc_id": doc_id,
                    "doc_title": doc.get("title", ""),
                    "sentences": sentences,
                    "rationale_text": rationale_text,
                    "label": label_str,
                    "label_num": label_num,
                })

    # Assign connected component ID as group variable for each instance
    for inst in instances:
        c_node = f"C_{inst['claim_id']}"
        inst["group_id"] = dsu.find(c_node)

    unique_groups = set(inst["group_id"] for inst in instances)
    support_count = sum(1 for inst in instances if inst["label_num"] == 1)
    contradict_count = sum(1 for inst in instances if inst["label_num"] == 0)

    metadata = {
        "total_instances": len(instances),
        "support_count": support_count,
        "contradict_count": contradict_count,
        "support_pct": round(support_count / len(instances) * 100, 2) if instances else 0,
        "contradict_pct": round(contradict_count / len(instances) * 100, 2) if instances else 0,
        "unique_claims": len(set(inst["claim_id"] for inst in instances)),
        "unique_docs": len(set(inst["doc_id"] for inst in instances)),
        "connected_components": len(unique_groups),
    }

    logger.info(f"Loaded SciFact dataset: {metadata['total_instances']} instances across {metadata['connected_components']} components.")
    return instances, metadata
