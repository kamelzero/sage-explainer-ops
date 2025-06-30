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
    n_users=8, n_systems=6, n_resources=10,
    p_login=0.3, p_lateral=0.04, p_sys_access=0.4,
    p_cross_user_cluster=0.15, n_user_clusters=3,
    seed=123
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

# th = 0.10
# edge_keep = explanation.edge_mask > th
# edge_idx  = data.edge_index[:, edge_keep]
# edge_w    = explanation.edge_mask[edge_keep]

# G = nx.Graph()
# G.add_nodes_from(range(data.num_nodes))
# for (u,v), w in zip(edge_idx.t().tolist(), edge_w.tolist()):
#     G.add_edge(u, v, weight=w)

# pos   = nx.spring_layout(G, seed=3)
# width = [G[u][v]['weight']*4 for u,v in G.edges()]
# nx.draw(G, pos,
#         node_color=['tomato' if n == node_id else 'skyblue' for n in G.nodes()],
#         width=width, with_labels=True)
# plt.show()

# #########################
# import networkx as nx
# def sbm_wrap(n_users=60, n_clusters=4, p_in=0.2, p_out=0.02, seed=42):
#     sizes = [n_users // n_clusters]*n_clusters
#     p = [[p_in if i==j else p_out
#           for j in range(n_clusters)] for i in range(n_clusters)]
#     G = nx.stochastic_block_model(sizes, p, seed=seed)
#     return G

# G = sbm_wrap(n_users=60, n_clusters=4, p_in=0.2, p_out=0.02, seed=42)

# #pos = nx.kamada_kawai_layout(G)        # energy-based, no ring bias#
# pos = nx.spring_layout(G, seed=random.randint(0, 9999), k=0.3)  # new seed & k
# # k controls edge length; lower k → tighter clusters, higher → spread
# nx.draw(G, pos, with_labels=True, node_color='skyblue')
# plt.show()

###############################################################
# Nicely-styled plot of the synthetic enterprise graph
###############################################################
import networkx as nx, matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib.patches as mpatches

# 1) convert to networkx (undirected for layout simplicity)
G = to_networkx(data, to_undirected=True)

# 2) node categories
users       = set(meta['users'])
systems     = set(meta['systems'])
resources   = set(meta['resources'])
compromised = set(meta['compromised'])
admins      = set(meta['admins'])

# 3) layout - multipartite keeps layers separate
for n in G.nodes():
    if n in users:       G.nodes[n]['layer'] = 0
    elif n in systems:   G.nodes[n]['layer'] = 1
    else:                G.nodes[n]['layer'] = 2
pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")

# 4) draw each class with distinct style
fig, ax = plt.subplots(figsize=(10, 4))

nx.draw_networkx_nodes(G, pos,
    nodelist=[n for n in users if n not in compromised],
    node_color="#66a3ff", node_size=350, edgecolors='k', linewidths=0.4, label="User")
nx.draw_networkx_nodes(G, pos,
    nodelist=list(compromised),
    node_color="#ff6666", node_size=400, edgecolors='k', linewidths=1.0, label="Compromised")
nx.draw_networkx_nodes(G, pos,
    nodelist=list(systems),
    node_color="#ffbe5c", node_shape='s', node_size=450, edgecolors='k', linewidths=0.4, label="System")
nx.draw_networkx_nodes(G, pos,
    nodelist=list(resources),
    node_color="#7ec87e", node_shape='D', node_size=450, edgecolors='k', linewidths=0.4, label="Resource")

# highlight admins with a thick ring
nx.draw_networkx_nodes(G, pos,
    nodelist=list(admins),
    node_color="none", node_size=500, edgecolors='navy', linewidths=2.0)

# 5) edge styling – WITHOUT the 'dashes=' kwarg
# (keep the same pos and node-drawing code)

# ---------- EDGE DRAWING WITH VISUALLY DISTINCT STYLES ----------------
for (u, v) in G.edges():
    if u in users and v in users:
        # lateral or credential reuse
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color="dimgray", alpha=0.7, width=1.2, style="dashdot"
        )
    elif (u in users and v in systems) or (v in users and u in systems):
        # user → system login
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color="black", width=1.8, style="solid"
        )
    else:
        # system → resource
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color="#228B22",  # forest-green
            width=1.2, style="dotted"
        )

# 6) labels only on small graphs (<80 nodes) to avoid clutter
if G.number_of_nodes() <= 80:
    nx.draw_networkx_labels(G, pos, font_size=8)

# 7) legend
handles = [
    mpatches.Patch(color="#66a3ff", label="User"),
    mpatches.Patch(color="#ff6666", label="Compromised user"),
    mpatches.Patch(color="#ffbe5c", label="System",  ec='k'),
    mpatches.Patch(color="#7ec87e", label="Resource", ec='k'),
    mpatches.Patch(facecolor='none', edgecolor='navy', label="Admin ring", linewidth=2)
]

from matplotlib.lines import Line2D
node_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#66a3ff', label='User'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff6666', label='Compromised user'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#ffbe5c', label='System'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#7ec87e', label='Resource'),
    Line2D([0],[0], marker='o', markersize=15, markerfacecolor='none',
           markeredgecolor='navy', label='Admin ring', linewidth=2)
]

edge_handles = [
    Line2D([0],[0], color='black',   lw=1.8, linestyle='solid',  label='User → System login'),
    Line2D([0],[0], color='dimgray', lw=1.2, linestyle='dashdot',label='User ↔ User lateral'),
    Line2D([0],[0], color='#228B22', lw=1.2, linestyle='dotted', label='System → Resource')
]

# ax.legend(handles=node_handles+edge_handles, frameon=False, loc='upper right',
#           ncol=3, fontsize=9)
leg = ax.legend(handles=node_handles + edge_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),      # just outside axes
                frameon=False)
# plt.tight_layout()
# ax.set_axis_off()
plt.tight_layout()
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

bullets = "\n".join("• "+to_sentence(r) for _, r in topk_edges.iterrows())
print(bullets)

############################################################

# LLM Explanation Summary

from helper_llm_explain import explain_edges_with_llm
import json

report = explain_edges_with_llm(bullets)
print(json.dumps(report, indent=2))
