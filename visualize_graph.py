#####################################################################
# Visualise the sub-graph returned by the explainer (PyG 2.6.1)
#####################################################################
from torch_geometric.utils import to_networkx
import networkx as nx, matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_graph(data, meta):
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
