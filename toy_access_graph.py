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
torch.manual_seed(42)
node_types = ['user']*12 + ['system']*6 + ['resource']*4
N = len(node_types)

x = torch.zeros((N, 3))
for i in range(12):
    x[i,0] = 1. if i in (0,1) else 0.       # admin bit
    x[i,1] = random.randint(2,20)/20.        # login freq
    x[i,2] = random.random()*0.5             # anomaly score

edges = [(0,12),(0,13),(1,13),(2,12),(3,14),(4,15),(5,15),(6,16),(7,17),
         (8,14),(9,16),(10,17),(11,13),
         (3,2),(5,4),(6,2),(9,3),
         (12,18),(13,19),(14,20),(15,21),(16,19),(17,20)]
edge_index = torch.tensor(edges + [(j,i) for i,j in edges]).t()

y = torch.zeros(N, dtype=torch.long); y[[3,5,6]] = 1   # compromised users
mask = torch.zeros(N, dtype=torch.bool); mask[:12] = True
data = Data(x=x, edge_index=edge_index, y=y, train_mask=mask)

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
    loss = F.cross_entropy(out[mask], y[mask])
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
