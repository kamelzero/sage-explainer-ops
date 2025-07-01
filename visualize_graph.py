#####################################################################
# Visualise the graph
#####################################################################
from torch_geometric.utils import to_networkx
import networkx as nx, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def visualize_graph(data, meta,
                    risk_df: pd.DataFrame | None = None,
                    bc_df:   pd.DataFrame | None = None,
                    k: int = 3,
                    figsize: tuple[int, int] = (10, 4)):
    # 1) convert to networkx (undirected for layout simplicity)
    G = to_networkx(data, to_undirected=True)

    # 2) node categories
    users       = set(meta['users'])
    systems     = set(meta['systems'])
    resources   = set(meta['resources'])
    compromised = set(meta['compromised'])
    admins      = set(meta['admins'])

    # high-value resources may or may not be present
    high_value  = set(meta.get('high_value', []))        # empty set if key absent
    hv_resources = resources & high_value                # subset of green diamonds
    std_resources = resources - high_value               # remaining resources

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
        nodelist=list(std_resources),                    # draw normal resources first
        node_color="#7ec87e", node_shape='D',
        node_size=450, edgecolors='k', linewidths=0.4, label="Resource")
    # draw high-value resources as red diamonds
    nx.draw_networkx_nodes(G, pos,
        nodelist=list(hv_resources),
        node_color="#ff4d4d", node_shape='D',
        node_size=520, edgecolors='k', linewidths=1.2, label="High-value resource")

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
    from matplotlib.lines import Line2D
    node_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#66a3ff', label='User'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff6666', label='Compromised user'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#ffbe5c', label='System'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#7ec87e', label='Resource'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#ff4d4d', label='High-value resource'),  # ← NEW
        Line2D([0],[0], marker='o', markersize=15, markerfacecolor='none',
               markeredgecolor='navy', linewidth=2, label='Admin ring'),
    ]

    edge_handles = [
        Line2D([0],[0], color='black',   lw=1.8, linestyle='solid',  label='User → System login'),
        Line2D([0],[0], color='dimgray', lw=1.2, linestyle='dashdot',label='User ↔ User lateral'),
        Line2D([0],[0], color='#228B22', lw=1.2, linestyle='dotted', label='System → Resource')
    ]

    leg = ax.legend(handles=node_handles + edge_handles,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),      # just outside axes
                    fontsize=8,
                    frameon=False)

    # ----------  score panels  -----------------------------------------
    def panel_lines(df, title, col, k=3):
        lines = [title] + [
            f"{int(nid):>3} : {score:.3f}"
            for nid, score in zip(df['node_id'].head(k), df[col].head(k))
        ]
        return "\n".join(lines)

    # Use the same anchor as the legend, but shift downward.
    x_anchor, y_anchor = 1.02, 0.26   # tweak second number to control vertical gap

    if risk_df is not None:
        txt_risk = panel_lines(risk_df.sort_values('p_compromised', ascending=False),
                            "Top-k Risk (p)", 'p_compromised', k=3)
        ax.text(x_anchor, y_anchor, txt_risk,
                transform=ax.transAxes, fontsize=8,
                fontfamily="monospace", va='top', ha='left')

    if bc_df is not None:
        txt_bc = panel_lines(bc_df.sort_values('broadcast_score', ascending=False),
                            "Top-k Broadcast", 'broadcast_score', k=3)
        ax.text(x_anchor, y_anchor - 0.14, txt_bc,    # 0.14 ≈ line-height gap
                transform=ax.transAxes, fontsize=8,
                fontfamily="monospace", va='top', ha='left')
    # ----------  score panels  -----------------------------------------

    plt.tight_layout()
    plt.show()
