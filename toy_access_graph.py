import torch
import torch.nn.functional as F
from torch_geometric.nn       import GCNConv
from torch_geometric.explain  import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode
from gnn_factory import build_gnn
from helper_tabular import TabularData, NodeCategories, get_node_types
import argparse

# -------------------------------------------------------------------
# build the toy "access" graph
# -------------------------------------------------------------------
from random_access_graph import generate_access_graph

# Set up argument parser
parser = argparse.ArgumentParser(description='Run toy access graph with a specified seed.')
parser.add_argument('--seed', type=int, default=99, help='Seed for random graph generation')
args = parser.parse_args()

# Use the seed from command line argument

# build a random graph
data, meta = generate_access_graph(
    n_users=8, n_systems=6, n_resources=10,
    p_login=0.3, p_lateral=0.04, p_sys_access=0.4,
    p_cross_user_cluster=0.15, n_user_clusters=3,
    high_value_ratio=0.15,
    seed=args.seed
)

# -------------------------------------------------------------------
# create a tiny GCN or GraphSAGE model
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
# run GNNExplainer
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

# -------------------------------------------------------------------
# calculate risk scores and braodcast scores
# -------------------------------------------------------------------

from rank_user_compromise import rank_users, broadcast_scores

top_users = rank_users(data, model, k=3, node_types=get_node_types(meta))

print("=== highest-risk users ===")
print(top_users)

bc = broadcast_scores(data, explainer, top_users["node_id"])
print("\n=== broadcast score ===")
print(bc)

# for now, only explain the node with highest risk score
node_ids = top_users["node_id"].tolist()
for node_id in top_users["node_id"]:
    explanation = explainer(data.x, data.edge_index, index=node_id)
    break # for now we only explain one node

# -------------------------------------------------------------------
# inspect masks
# -------------------------------------------------------------------
feat_names = meta["data_feature_names"]
row = explanation.node_mask[node_id]           # (3,) tensor
for w, n in sorted(zip(row.tolist(), feat_names), reverse=True):
    print(f"{n:<15} {w:6.3f}")

print(f"\nTop-5 influential edges for node {node_id}:")
edge_scores = explanation.edge_mask
edge_idx_T  = data.edge_index.t()
top = torch.topk(edge_scores, 5)
for s, idx in zip(top.values, top.indices):
    u, v = edge_idx_T[idx].tolist()
    print(f"{u:>2} → {v:<2}  score={s:.3f}")

# Get tabular data for easier, further inspection
node_categories = NodeCategories(meta)
tabular_data = TabularData(data, meta, explanation, node_id)
node_info = tabular_data.get_per_node_info()
edge_info = tabular_data.get_per_edge_info()
topk_edges = tabular_data.get_topk_edges()

print(node_info)
print(edge_info)

########################################
from helper_tabular import explain_resource_paths
from helper_llm_explain import build_edge_sentence_fn
import json

# 1) decide target(s)
targets = meta.get('high_value', []) or meta['resources']

# 2) for each crown–jewel resource, get the top K conduit edges
all_hv_reports = {}
for rid in targets:
    hv_df, hv_nodes = explain_resource_paths(data, meta, explainer, rid, top_k=10)
    all_hv_reports[rid] = hv_df        # store for dashboards / bullets
    print(f"\nTop paths to resource {rid}")
    print(hv_df[['src','dst','importance','kind']])

# 3) convert hv_df rows to human bullets → feed to LLM summariser
to_sentence = build_edge_sentence_fn(set(meta['users']),
                                     set(meta['systems']),
                                     set(meta['resources']))
resource_bullets = "\n".join("• "+to_sentence(r) for _, r in hv_df.iterrows())
user_bullets = "\n".join("• "+to_sentence(r) for _, r in topk_edges.iterrows())

# # -------------------------------------------------------------------
# # create an LLM explanation summary from top edges
# # -------------------------------------------------------------------
from helper_llm_explain import explain_edges_with_llm, build_edge_sentence_fn
import json

to_sentence = build_edge_sentence_fn(node_categories.users, node_categories.systems, node_categories.resources)
#bullets = [ "• "+to_sentence(r) for _, r in topk_edges.iterrows() ]

print("=== LLM Explanation Summary ===")
report_resource = explain_edges_with_llm(resource_bullets)
report_user = explain_edges_with_llm(user_bullets)

print("=== Resource Report ===")
print(resource_bullets)
print("-"*100)
print(json.dumps(report_resource, indent=2))

print("=== User Report ===")
print(user_bullets)
print("-"*100)
print(json.dumps(report_user, indent=2))

# # -------------------------------------------------------------------
# # visualise the graph
# # -------------------------------------------------------------------

from visualize_graph import visualize_graph
visualize_graph(data, meta, top_users, bc)
