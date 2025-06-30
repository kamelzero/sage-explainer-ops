######################################################################
# 0   install deps  (only once, from a shell)
#     ---------------------------------------------------------------
#     python -m venv venv && . venv/bin/activate
#     pip install torch==2.2.0 torch_geometric==2.6.1 networkx matplotlib
######################################################################

import torch, random, networkx as nx, matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data     import Data
from torch_geometric.nn       import GCNConv
from torch_geometric.explain  import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelMode
from torch_geometric.utils    import to_networkx


# -------------------------------------------------------------------
# 1  build the toy "access" graph
# -------------------------------------------------------------------
from random_access_graph import generate_access_graph

# build a random graph
data, meta = generate_access_graph(
    n_users=50,
    n_systems=20,
    n_resources=10,
    p_login=0.3,
    p_lateral=0.08,
    p_sys_access=0.5,
    p_compromised=0.12,
    seed=2025
)

# ❶ rebuild node_types from meta
node_types = ['user']     * len(meta['users']) \
           + ['system']   * len(meta['systems']) \
           + ['resource'] * len(meta['resources'])
# length check
assert len(node_types) == data.num_nodes

# -------------------------------------------------------------------
# 2  tiny GCN
# -------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 8)
        self.conv2 = GCNConv(8, 2)
    def forward(self, x, ei):
        return self.conv2(F.relu(self.conv1(x, ei)), ei)

model = GCN()
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

node_id = 3                                    # investigate user #3
explanation = explainer(data.x, data.edge_index, index=node_id)

# -------------------------------------------------------------------
# 4  inspect masks
# -------------------------------------------------------------------
feat_names = ["is_admin", "login_freq", "anomaly_score"]
row = explanation.node_mask[node_id]           # (3,) tensor
for w, n in sorted(zip(row.tolist(), feat_names), reverse=True):
    print(f"{n:<15} {w:6.3f}")

print("\nTop-5 influential edges for node 3:")
edge_scores = explanation.edge_mask
edge_idx_T  = data.edge_index.t()
top = torch.topk(edge_scores, 5)
for s, idx in zip(top.values, top.indices):
    u, v = edge_idx_T[idx].tolist()
    print(f"{u:>2} → {v:<2}  score={s:.3f}")

# -------------------------------------------------------------------
# 5  visualise sub-graph
# -------------------------------------------------------------------

############################################################################
# Visualise the sub-graph returned by the explainer (PyG 2.6.1)
############################################################################
from torch_geometric.utils import to_networkx
import networkx as nx, matplotlib.pyplot as plt

#sub_exp = explanation.get_explanation_subgraph()   # ← no args in 2.6.1
# G       = to_networkx(sub_exp,
#                       node_attrs=['node_mask'],
#                       edge_attrs=['edge_mask'],
#                       to_undirected=True)

# pos   = nx.spring_layout(G, seed=2)
# width = [G[u][v]['edge_mask']*4 for u,v in G.edges()]
# nx.draw(G, pos,
#         node_color=['tomato' if n == node_id else 'skyblue' for n in G.nodes()],
#         width=width, with_labels=True)
# plt.show()

#####################

th = 0.10
edge_keep = explanation.edge_mask > th
edge_idx  = data.edge_index[:, edge_keep]
edge_w    = explanation.edge_mask[edge_keep]

G = nx.Graph()
G.add_nodes_from(range(data.num_nodes))
for (u,v), w in zip(edge_idx.t().tolist(), edge_w.tolist()):
    G.add_edge(u, v, weight=w)

pos   = nx.spring_layout(G, seed=3)
width = [G[u][v]['weight']*4 for u,v in G.edges()]
nx.draw(G, pos,
        node_color=['tomato' if n == node_id else 'skyblue' for n in G.nodes()],
        width=width, with_labels=True)
plt.show()

############################################################

# Node-level summary → DataFrame

import pandas as pd, torch

# --- basic per-node info -------------------------------------------
df_nodes = pd.DataFrame({
    'node_id'   : range(data.num_nodes),
    'type'      : node_types,
    'is_admin'  : data.x[:,0].tolist(),
    'login_freq': data.x[:,1].tolist(),
    'anomaly'   : data.x[:,2].tolist(),
    'label'     : data.y.tolist(),              # 0 = safe, 1 = compromised
})

# --- feature importance for the *explained* node -------------------
feat_imp = explanation.node_mask[node_id].tolist()
df_nodes.loc[node_id, ['is_admin','login_freq','anomaly']] = feat_imp

print(df_nodes.head(15))

###############################################################
# Edge-importance DataFrame  —  works for any edge_index size
###############################################################
import pandas as pd, torch

edge_index_T = data.edge_index.t().tolist()    # 46 tuples in the same order
edge_mask    = explanation.edge_mask.detach().tolist()

df_edges = pd.DataFrame(edge_index_T, columns=['src', 'dst'])
df_edges['importance'] = edge_mask

# quick semantic tag
def tag(u, v):
    if u < 12 and v < 12:                        # both users
        return 'lateral'
    if (u < 12 <= v < 18) or (v < 12 <= u < 18): # user ↔ system
        return 'login'
    return 'sys→res'

df_edges['kind'] = [tag(u, v) for u, v in edge_index_T]

# sort & show the 10 most influential edges
topk_edges = df_edges.sort_values('importance', ascending=False).head(10)
print(topk_edges)

##############################
# Keep track of only one direction of edges (higher importance one)

# keep only one direction by dropping duplicates of {min(u,v), max(u,v)}
df_undirected = (df_edges
            .assign(pair=df_edges.apply(lambda r: tuple(sorted((r.src, r.dst))), axis=1))
            .sort_values('importance', ascending=False)
            .drop_duplicates('pair'))
df_undirected = df_undirected[['src','dst','importance','kind']]
print(df_undirected.head(10))

############################################################

def to_sentence(row):
    u, v, w, k = int(row.src), int(row.dst), row.importance, row.kind
    if k == 'login':
        if u < 12:   return f"User {u} logs into system {v}  (w={w:.2f})"
        else:        return f"User {v} logs into system {u}  (w={w:.2f})"
    if k == 'lateral':
        return f"User {u} shares creds with user {v}  (w={w:.2f})"
    return f"System {u} accesses resource {v}  (w={w:.2f})"

print("\n".join("• "+to_sentence(r) for _, r in topk_edges.iterrows()))

############################################################

# LLM Explanation Summary

from helper_llm_explain import explain_edges_with_llm
import json

bullets = """• User 3 shares creds with user 2  (w=0.96)
• User 6 shares creds with user 2  (w=0.94)
• User 3 logs into system 14       (w=0.94)
• User 3 shares creds with user 9  (w=0.94)
• User 8 logs into system 14       (w=0.94)
"""

report = explain_edges_with_llm(bullets)
print(json.dumps(report, indent=2))
