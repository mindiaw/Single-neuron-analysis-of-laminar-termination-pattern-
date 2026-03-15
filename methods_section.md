# Methods

## Data preprocessing and anatomical mapping
Single-neuron soma coordinates and target-area projection data were downloaded from https://mouse.digital-brain.cn/projectome and processed using a custom Python pipeline implemented in a Python 3.11 environment with AllenSDK. Soma coordinates (`x`, `y`, `z`, in um) were mapped to the Allen Common Coordinate Framework version 3 (CCFv3; 10 um voxel resolution) to assign each neuron to a source structure acronym, source area, and source layer. Cortical source layers were reduced to five canonical categories (L1, L2/3, L4, L5, and L6), with layers 6a and 6b merged into L6. Soma coordinates falling outside the annotation volume were clipped to the nearest valid voxel and flagged.

For each source-target connection, wide-format terminal-point tables were parsed and restricted to rows corresponding to the expected target cortical area. When the Allen structure tree was available, target-area matching included the target area and all of its annotated descendants; otherwise, matching relied on exact or hyphen-delimited acronym matching. Terminal counts were summed within each canonical cortical layer to obtain per-neuron layer-specific terminal counts (`tL1`, `tL23`, `tL4`, `tL5`, `tL6`) and total target-area terminals (`tTotal`).

## Feature construction
Laminar projection profiles were represented using terminal fractions. The fraction of a neuron's target-area terminals located in layer `l` was defined as

```text
fT_l = tL_l / tTotal
```

and the resulting vector was then normalized to sum to 1 across layers for each neuron.

## Quality control and inclusion criteria
Neurons with non-numeric soma coordinates were removed before mapping. Duplicate neuron IDs within a connection were removed, retaining the first occurrence. During preprocessing, neurons with at least 1 target-area terminals were included: 

```text
tTotal >= 1
```

A quality-control (QC) mismatch was defined as a neuron whose mapped source area did not match the expected source area for that connection. QC mismatch rows were written to a separate file and were excluded from downstream analyses. When excluded, rows were removed using the pair (`neuron_id`, `connection_name`) to avoid erroneous filtering across repeated neuron IDs in different connections.

## Determination of cluster number
The optimal number of clusters was evaluated separately for feedforward (FF) and feedback (FB) projections using hierarchical clustering over the range `k = 2` to `k = 20`. Two complementary criteria were computed for each `k`: silhouette score and size-weighted within-cluster variance. 

Final cluster numbers were set to `k_FF = 6` and `k_FB = 6`.

## Hierarchical clustering
Hierarchical agglomerative clustering was performed using the Ward linkage method and the correlation distance geometry. FF and FB neurons were clustered separately. 

For Ward clustering with correlation distance, each neuron's feature vector was mean-centered and L2-normalized before clustering, so that Euclidean distances in the transformed space were proportional to correlation distance. 

## Source-layer and motif analyses
Several downstream analyses were performed on the clustered data. First, source-to-target profile analysis quantified the relationship between soma depth and target-layer projection pattern by computing the mean laminar profile for each source layer. Differences across source layers were tested separately for each target layer using Kruskal-Wallis tests with Benjamini-Hochberg false-discovery-rate (BH-FDR) correction. For target layers showing a significant omnibus effect, pairwise source-layer comparisons were performed using two-sided Mann-Whitney `U` tests with BH-FDR correction.

Second, cluster stability was assessed by bootstrap resampling within each direction. In each bootstrap iteration, neurons were resampled with replacement, reclustered using the same parameters as in the main analysis, and used to update a co-clustering matrix. The final stability matrix reported, for each neuron pair, the proportion of bootstrap iterations in which both neurons were present and assigned to the same cluster.

Third, source-layer enrichment and connection enrichment were quantified for each cluster using Fisher's exact tests. For a given cluster and category (source layer or connection), enrichment was expressed as

```text
log2((observed fraction + delta) / (expected fraction + delta))
```

where `delta = 1 / (2N)` was a small pseudocount and `N` was the number of neurons in the relevant direction-specific dataset. Resulting `p`-values were adjusted using BH-FDR.

## Cross-direction motif overlap
To assess the extent to which FF and FB clusters represented shared versus direction-specific motifs, two complementary analyses were performed. First, pairwise distances between FF and FB cluster centroids were computed. Second, FF and FB neurons were pooled and jointly clustered into `k_combined = k_FF + k_FB` groups. Direction purity for each combined cluster was defined as

```text
purity = max(n_FF / n, n_FB / n)
```

where `n` is the cluster size. Clusters with lower purity were interpreted as candidate shared motifs. Third, each FF neuron was assigned to its nearest FB centroid and each FB neuron to its nearest FF centroid, yielding cross-assignment matrices and nearest-centroid distance distributions.

