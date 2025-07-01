# Structural Vulnerability Discovery with GNN Explainers

Tiny, reproducible sandbox that shows how **Graph Neural Networks + GNNExplainer**
can flip a classic blue-team model (“find compromised hosts”) into a
red-team recon tool (“which nodes/edges give me the shortest path to the crown-jewel?”).

---

## Quick-start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # PyG, etc.
export OPENAI_API_KEY="sk-..."         # for LLM summaries
```

Run the notebook `toy_access_graph.ipynb`

# Key concepts

| Term                          | What it means                                                                                                                                                       | Why you care                                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Influence mask**            | GNNExplainer optimises soft gates on every edge and feature; any gate that cannot be closed without changing the model’s prediction is flagged **important (≈ 1)**. | Turns an opaque GNN into a minimal, human-readable sub-graph.                                                               |
| **Broadcast score**           | For node *v*:  `Σ edge_mask[e]`  over all incident edges.                                                                                                           | Quantifies *how much* a node’s connections propagate risk or privilege—great for spotting credential hubs and choke-points. |
| **High-value nodes**          | Mark any asset—user, system, or resource—as “HV”; run the explainer on that node instead of on a label.                                                             | Reveals the **shortest-influence path** an attacker would follow to reach the crown jewel.                                  |
| **Inductive vs transductive** | Swap `"gcn"` (single-graph) for `"sage"` (many-graph) in one line.                                                                                                  | Lets you test whether the influence patterns generalise to unseen topologies.                                               |
| **LLM post-processor**        | Converts the top-weight edges into JSON red/blue-team playbooks.                                                                                                    | Bridges raw masks → actionable guidance without manual wording.                                                             |
