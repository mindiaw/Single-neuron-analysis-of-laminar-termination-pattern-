# Single-neuron analysis of laminar termination patterns

This repository contains the analysis code, curated input tables, and generated outputs for my master's thesis on laminar projection motifs in cortical feedforward (FF) and feedback (FB) pathways.

The workflow maps single-neuron soma coordinates to Allen CCFv3, builds layer-resolved terminal features, clusters neurons by laminar projection profile, and compares motif structure between FF and FB projections.

For the full methods description, see [methods_section.md](methods_section.md).

## Data and outputs included

This repository currently includes:

- input data in `data/` (soma coordinates, terminal-point tables, config files)
- generated analysis outputs in `output/thesis/`

## Repository structure

- `run_thesis_pipeline.py`: end-to-end thesis pipeline orchestrator
- `mapping_to_ccf.py`: CCF mapping + per-neuron feature construction
- `evaluate_optimal_k.py`: optimal cluster-number analysis
- `cluster_termination_patterns.py`: hierarchical clustering
- `visualize_clustering.py`: motif-centric clustering visualizations
- `analyze_motif_core.py`: source-layer, enrichment, and stability analyses
- `analyze_cluster_overlap.py`: FF/FB overlap and cross-assignment analyses
- `visualize_raw_data.py`: optional descriptive/raw visualizations

## Environment setup

Tested environment:

- Python `3.11`
- dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the thesis pipeline

Run the full pipeline:

```bash
python run_thesis_pipeline.py
```

Example configuration aligned with the methods summary (terminal-based clustering, FF/FB-specific cluster counts, QC mismatch exclusion):

```bash
python run_thesis_pipeline.py \
  --mode terminals \
  --scope direction \
  --n_clusters_ff 6 \
  --n_clusters_fb 6 \
  --min_target_terminals 1 \
  --exclude_qc_mismatches
```

Optional: run only selected steps (1-6) or skip specific steps:

```bash
python run_thesis_pipeline.py --only 1 2 3
python run_thesis_pipeline.py --skip 4
```

## Main outputs

Key outputs are written under `output/thesis/`:

- `ALL_CONNECTIONS__mapped_and_terminals.csv`: mapped table with per-neuron features
- `clustering/`: cluster assignments, centroids, and heatmaps
- `optimal_k/`: k-selection tables and curves
- `source_layers/`: enrichment, source-layer, and stability analyses
- `overlap/`: FF/FB motif overlap analyses
- `raw_viz/`: optional raw descriptive visualizations

## Methods snapshot

Core analysis decisions:

- CCF mapping to assign source area/layer from soma coordinates
- canonical cortical layers: `L1`, `L2/3`, `L4`, `L5`, `L6`
- terminal-fraction feature representation per neuron
- hierarchical clustering (Ward) evaluated over `k=2..20`
- final thesis setting: `k_FF = 6`, `k_FB = 6`
- source-layer and motif analyses with multiple-testing correction (BH-FDR)

Detailed definitions, QC logic, and equations are in [methods_section.md](methods_section.md).
