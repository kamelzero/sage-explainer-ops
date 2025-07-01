import torch
import torch.nn.functional as F
from torch_geometric.nn       import GCNConv
from torch_geometric.explain  import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode
from gnn_factory import build_gnn
from helper_tabular import TabularData, NodeCategories, get_node_types

# -------------------------------------------------------------------
# 1  build the toy "access" graph
# -------------------------------------------------------------------
from random_access_graph import generate_access_graph

# build a random graph
data, meta = generate_access_graph(
    n_users=8, n_systems=6, n_resources=10,
    p_login=0.3, p_lateral=0.04, p_sys_access=0.4,
    p_cross_user_cluster=0.15, n_user_clusters=3,
    seed=123
)

# -------------------------------------------------------------------
# 2  tiny GCN
# -------------------------------------------------------------------
MODEL_NAME = "sage"          # <── swap "gcn"  /  "sage"
model = build_gnn(MODEL_NAME,
                  in_dim=3,
                  hidden=32,
                  n_classes=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(200):
    opt.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()
model.eval()

# -------------------------------------------------------------------
# 3  new-API GNNExplainer
# -------------------------------------------------------------------
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=500, lr=0.005),        # more epochs → clearer masks
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=ModelConfig(
        mode=ModelMode.multiclass_classification,
        task_level='node',
        return_type='raw',                     # logits
    ),
)

#####################################################################
# risk rank users and select which ones to investigate

from rank_user_compromise import rank_users, broadcast_scores

top_users = rank_users(data, model, k=3, node_types=get_node_types(meta))

print("=== highest-risk users ===")
print(top_users)

bc = broadcast_scores(data, explainer, top_users["node_id"])
print("\n=== broadcast score ===")
print(bc)

# for now, only explain the node with highest broadcast score
node_ids = top_users["node_id"].tolist()
for node_id in top_users["node_id"]:
    explanation = explainer(data.x, data.edge_index, index=node_id)

# -------------------------------------------------------------------
# 4  inspect masks
# -------------------------------------------------------------------
feat_names = ["is_admin", "login_freq", "anomaly_score"]
row = explanation.node_mask[node_id]           # (3,) tensor
for w, n in sorted(zip(row.tolist(), feat_names), reverse=True):
    print(f"{n:<15} {w:6.3f}")

print("\nTop-5 influential edges for node {node_id}:")
edge_scores = explanation.edge_mask
edge_idx_T  = data.edge_index.t()
top = torch.topk(edge_scores, 5)
for s, idx in zip(top.values, top.indices):
    u, v = edge_idx_T[idx].tolist()
    print(f"{u:>2} → {v:<2}  score={s:.3f}")

# -------------------------------------------------------------------
# 5  visualise sub-graph
# -------------------------------------------------------------------

from visualize_graph import visualize_graph
visualize_graph(data, meta, top_users, bc)

############################################################

# LLM Explanation Summary

from helper_llm_explain import explain_edges_with_llm, build_edge_sentence_fn
import json

node_categories = NodeCategories(meta)
tabular_data = TabularData(data, meta, explanation, node_id)
topk_edges = tabular_data.get_topk_edges()

to_sentence = build_edge_sentence_fn(node_categories.users, node_categories.systems, node_categories.resources)
bullets = [ "• "+to_sentence(r) for _, r in topk_edges.iterrows() ]

bullets = "\n".join("• "+to_sentence(r) for _, r in tabular_data.get_topk_edges().iterrows())
print("=== Bullets ===")
print(bullets)

print("=== LLM Explanation Summary ===")
report = explain_edges_with_llm(bullets)
print(json.dumps(report, indent=2))
