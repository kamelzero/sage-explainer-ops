# Overview

This project considers discovery of structural vulnerabilities via node influence, using Graph Neural Netowrks (GNNs). This aims to flip a blue-team strategy (network hardening) into a target discovery tool for offensive ops.

The idea is to treat a network (logical, social, communication, or access) as a graph, and use GNN explainability tools to identify critical nodes that:

    * exert high influence on other nodes' embeddings
    * are bottlenecks in information flow
    * determine classification outcomes (e.g., "malicious" vs. "benign")

Once identified, these nodes become high-value targets for red teamers.


# Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set `OPENAI_API_KEY` is set in your environment variables, e.g., set it in your `~/.bashrc`.

This is used for LLM sentence-level explanation of GNNExplainer output.
