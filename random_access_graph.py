# random_access_graph.py
import torch, random
from torch_geometric.data import Data

def generate_access_graph(n_users: int = 30,
                          n_user_clusters: int = 3,
                          n_systems: int = 12,
                          n_resources: int = 6,
                          p_cross_user_cluster: float = 0.02,
                          p_login: float = 0.25,
                          p_lateral: float = 0.05,
                          p_sys_access: float = 0.45,
                          p_compromised: float = 0.15,
                          high_value_ratio: float = 0.15,   # % of resources
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
    data_feature_names = ["is_admin", "login_freq", "anomaly_score"]

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

    # split users into clusters, then add extra intra-cluster edges
    clusters   = torch.tensor_split(torch.tensor(users),
                                    n_user_clusters, dim=0)
    for cl in clusters:
        for i, u in enumerate(cl):
            for v in cl[i+1:]:
                if random.random() < p_cross_user_cluster:
                    edges.append((int(u), int(v)))

    # make systems talk to other systems
    for i, s in enumerate(systems):
        for t in systems[i+1:]:
            if random.random() < 0.10:          # tune
                edges.append((s, t))

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

    # pick high-value resources
    n_high = max(1, int(high_value_ratio * n_resources))
    high_value = random.sample(resources, k=n_high)
    # tag in feature[2] for non-high (optional visual difference)
    for r in resources:
        if r not in high_value:
            x[r, 2] = 0.0     # no "anomaly" at start

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
    meta = dict(users=users, systems=systems, resources=resources,
                high_value=high_value,
                compromised=compromised, admins=admins,
                data_feature_names=data_feature_names)
    return data, meta


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    g, m = generate_access_graph(seed=42)
    print(g)
    print(m)
