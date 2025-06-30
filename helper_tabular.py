import pandas as pd

class NodeCategories:
    def __init__(self, meta):
        self.users       = set(meta['users'])
        self.systems     = set(meta['systems'])
        self.resources   = set(meta['resources'])
        self.compromised = set(meta['compromised'])
        self.admins      = set(meta['admins'])

def get_node_types(meta):
    return ['user']     * len(meta['users']) \
           + ['system']   * len(meta['systems']) \
           + ['resource'] * len(meta['resources'])

class TabularData:
    def __init__(self, data, meta, explanation, node_id):
        """
        data: torch_geometric.data.Data
        meta: dict
        explanation: torch_geometric.explain.Explainer
        node_id: int (the node to be explained)
        """
        self.data = data
        self.meta = meta
        self.explanation = explanation
        self.df_nodes = self._setup_node_info(node_id)
        self.df_edges = self._setup_edge_info()

    def _setup_node_info(self, node_id):
        """
        node_id: int (the node to be explained)
        returns: pd.DataFrame (nodes)
        """
        # --- basic per-node info -------------------------------------------
        df_nodes = pd.DataFrame({
            'node_id'   : range(self.data.num_nodes),
            'type'      : get_node_types(self.meta),
            'is_admin'  : self.data.x[:,0].tolist(),
            'login_freq': self.data.x[:,1].tolist(),
            'anomaly'   : self.data.x[:,2].tolist(),
            'label'     : self.data.y.tolist(),              # 0 = safe, 1 = compromised
        })

        # --- feature importance for the *explained* node -------------------
        feat_imp = self.explanation.node_mask[node_id].tolist()
        df_nodes.loc[node_id, ['is_admin','login_freq','anomaly']] = feat_imp
        return df_nodes

    def _setup_edge_info(self):
        """
        returns: pd.DataFrame (edges)
        """
        edge_index_T = self.data.edge_index.t().tolist()    # 46 tuples in the same order
        edge_mask    = self.explanation.edge_mask.detach().tolist()
        df_edges = pd.DataFrame(edge_index_T, columns=['src', 'dst'])
        df_edges['importance'] = edge_mask
        # quick semantic tag
        nc = NodeCategories(self.meta)
        def tag(u, v):
            if u in nc.users and v in nc.users:
                return "lateral"
            if (u in nc.users and v in nc.systems) or (v in nc.users and u in nc.systems):
                return "login"
            if (u in nc.systems and v in nc.resources) or (v in nc.systems and u in nc.resources):
                return "sys→res"
            return "other"
        edge_index_T = self.data.edge_index.t().tolist()    # 46 tuples in the same order
        df_edges['kind'] = [tag(u, v) for u, v in edge_index_T]
        return df_edges
    
    def get_per_node_info(self):
        return self.df_nodes

    def get_per_edge_info(self):
        return self.df_edges
    
    def get_topk_edges(self, k=10):
        # sort & return the k most influential edges
        topk_edges = self.df_edges.sort_values('importance', ascending=False).head(k)
        return topk_edges

    def get_undirected_edges(self):
        # keep only one direction by dropping duplicates of {min(u,v), max(u,v)}
        df_undirected = (self.df_edges
                    .assign(pair=self.df_edges.apply(lambda r: tuple(sorted((r.src, r.dst))), axis=1))
                    .sort_values('importance', ascending=False)
                    .drop_duplicates('pair'))
        df_undirected = df_undirected[['src','dst','importance','kind']]
        return df_undirected
    