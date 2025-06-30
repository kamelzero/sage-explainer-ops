# random_access_graph.py
import torch, random
from torch_geometric.data import Data

def generate_access_graph(n_users: int = 30,
                          n_systems: int = 12,
                          n_resources: int = 6,
                          p_login: float = 0.25,
                          p_lateral: float = 0.05,
                          p_sys_access: float = 0.45,
                          p_compromised: float = 0.15,
                          seed: int | None = None):
    """
    Returns
    -------
    data      : torch_geometric.data.Data
        data.x               – (num_nodes, 3)  float features
        data.edge_index      – (2, num_edges)  long
        data.y               – (num_nodes,)    long  (0 = safe, 1 = compromised user)
        data.train_mask      – bool mask over user nodes (optional)
    meta      : dict
        { 'users': [ids], 'systems': [ids], 'resources': [ids] }
    """
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed)

    # ----- assign node id ranges ------------------------------------------------
    users      = list(range(n_users))
    systems    = list(range(n_users, n_users + n_systems))
    resources  = list(range(n_users + n_systems,
                            n_users + n_systems + n_resources))
    num_nodes  = n_users + n_systems + n_resources

    # ----- node features --------------------------------------------------------
    x = torch.zeros((num_nodes, 3))
    # feature[0] = admin flag (pick two random admins)
    admins = random.sample(users, k=max(1, n_users // 10))
    x[admins, 0] = 1.
    # feature[1] = login frequency (0.05–1 on users)
    x[users, 1] = torch.rand(len(users)) * 0.95 + 0.05
    # feature[2] = anomaly score baseline
    x[users, 2] = torch.rand(len(users)) * 0.3

    # ----- edges ----------------------------------------------------------------
    edges = []

    # user logins to systems
    for u in users:
        for s in systems:
            if random.random() < p_login:
                edges.append((u, s))

    # user ↔ user credential sharing / lateral movement (undirected semantic)
    for i, u in enumerate(users):
        for v in users[i+1:]:
            if random.random() < p_lateral:
                edges.append((u, v))

    # system accesses to resources
    for s in systems:
        for r in resources:
            if random.random() < p_sys_access:
                edges.append((s, r))

    # make graph undirected
    edges += [(v, u) for (u, v) in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ----- labels ----------------------------------------------------------------
    y = torch.zeros(num_nodes, dtype=torch.long)
    k_comp = max(1, int(p_compromised * n_users))
    compromised = random.sample(users, k=k_comp)
    y[compromised] = 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[users] = True      # usually you train only on user nodes

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

    meta = dict(users=users, systems=systems, resources=resources,
                compromised=compromised, admins=admins)
    return data, meta


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    g, m = generate_access_graph(seed=42)
    print(g)
    print(m)
