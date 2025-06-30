# ------------------------------------------------------------
#  risk_ranking.py
# ------------------------------------------------------------
from __future__ import annotations
from typing import List, Sequence

import torch
import pandas as pd
from torch_geometric.data import Data
from torch import Tensor


def rank_users(data: Data,
               model,
               k: int = -1,
               node_types: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Return the top-k user nodes by model-predicted compromise probability.

    Parameters
    ----------
    data : PyG Data
    model : torch.nn.Module    # returns (N, C) logits
    k : int                    # how many top users to keep
    node_types : list[str]     # optional explicit list; otherwise assume
                               # every index < n_users is a user.

    Returns
    -------
    DataFrame with columns [node_id, p_compromised], sorted desc.
    """
    k = len(data.x) if k <= 0 else k
    model.eval()
    with torch.no_grad():
        logits: Tensor = model(data.x, data.edge_index)
        probs:  Tensor = logits.softmax(dim=-1)

    n = data.num_nodes
    if node_types is None:
        # fallback heuristic: first block of nodes are users
        n_users = len([i for i in range(n) if data.train_mask[i]])
        node_types = ['user'] * n_users + ['other'] * (n - n_users)

    df = pd.DataFrame({
        "node_id": range(n),
        "type":    node_types,
        "p_compromised": probs[:, 1].cpu().tolist()
    })
    users = (df["type"] == "user")
    df = df[users].nlargest(k, "p_compromised").reset_index(drop=True)
    return df


def broadcast_scores(data: Data,
                     explainer,
                     user_ids: Sequence[int]) -> pd.DataFrame:
    """
    Sum the explainer's edge-mask weights incident to each node in `user_ids`.
    Higher score ⇒ stronger 'broadcast' of malicious signal.
    """
    edge_src, edge_dst = data.edge_index
    rows = []
    for nid in user_ids:
        exp = explainer(data.x, data.edge_index, index=nid)
        incident = (edge_src == nid) | (edge_dst == nid)
        score = float(exp.edge_mask[incident].sum())
        rows.append({"node_id": nid, "broadcast_score": score})
    return pd.DataFrame(rows).sort_values("broadcast_score",
                                          ascending=False).reset_index(drop=True)


# ---------------- example usage ---------------------------------------
if __name__ == "__main__":
    from generate_access_graph import generate_access_graph
    from model_and_explainer import model, explainer  # wherever you defined them

    data, meta = generate_access_graph(seed=42)

    top_users = rank_users(data, model, k=3, node_types=
                           ['user']*len(meta['users']) +
                           ['system']*len(meta['systems']) +
                           ['resource']*len(meta['resources']))

    print("=== highest-risk users ===")
    print(top_users)

    bc = broadcast_scores(data, explainer, top_users["node_id"])
    print("\n=== broadcast score ===")
    print(bc)
