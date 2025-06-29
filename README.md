# Goals

1) Start with GCN vs. GraphSAGE for static node classification and inductive generalization.
2) Then expand to building a system with multiple components (e.g., detection, attribution, trend monitoring), e.g.
    - GAT for explainability,
    - Temporal GNN for evolving threat modeling


## Phase 1: Baseline + Inductive Generalization

### Objective

Compare GCN (transductive) and GraphSAGE (inductive) on node classification:

- Classify whether each IP address is malicious or benign.

### Steps

1. Load + preprocess CIC IoT 2023 (e.g., 1-day slice)
2. Build graph:
   - Nodes = IPs
   - Edges = flows (src → dst)
   - Node features = aggregated flow stats
   - Labels = derived from flow labels (if any IP has malicious flows → label as malicious)
3. Train/Test split:
   - By IP addresses (so GraphSAGE can generalize to unseen nodes)
4. Train GCN and GraphSAGE
5. Compare:
   - Accuracy, F1, ROC-AUC
   - Generalization to unseen IPs

## Phase 2: Multi-Component System

### Component 1: Detection

Use GraphSAGE for inductive threat detection on incoming IPs or flows.

### Component 2: Attribution

Use GAT:

- To see which neighbors most influence a node's classification
- Helps highlight which connections are likely attack-related

### Component 3: Trend Monitoring

Use Temporal GNN (e.g., TGAT, TGN):

- Model how threats emerge or escalate over time
- Temporal edge features (e.g., timestamped flows)
- Output: time series of risk scores or activity heatmaps


# Install

```
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

# References

* CIC IIoT Dataset (2024)
    * https://www.unb.ca/cic/datasets/iiot-dataset-2024.html

* CIC Tabular IoT Attack (2024)
    * https://www.unb.ca/cic/datasets/tabular-iot-attack-2024.html

* CIC IoT Dataset (2023)
    * https://www.unb.ca/cic/datasets/iotdataset-2023.html
        * http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/example/ - training example
    * https://www.kaggle.com/datasets/akashdogra/cic-iot-2023?resource=download - 2.8 GB

* Evaluating deep learning variants for cyber-attacks detection and multi-class classification in IoT networks (2024)
    * https://pmc.ncbi.nlm.nih.gov/articles/PMC10803060/#_ad93_
    * peerj-cs-10-1793-s001.ipynb - training example

* XG-NID: Dual-Modality Network Intrusion Detection using a Heterogeneous Graph Neural Network and Large Language Model
    * https://arxiv.org/html/2408.16021v1 (2024)
    * https://github.com/Yasir-ali-farrukh/GNN4ID/blob/main/GNN4ID.ipynb - downloading
    * https://github.com/Yasir-ali-farrukh/GNN4ID/blob/main/Data_preprocessing_CIC-IoT2023.ipynb - preprocessing
    * https://github.com/Yasir-ali-farrukh/GNN4ID/blob/main/GNN4ID_Model.ipynb - training

* Exploring Graph Neural Networks for Robust Network Intrusion Detection (2025)
    * https://www.sciencedirect.com/science/article/pii/S1877050925017223


