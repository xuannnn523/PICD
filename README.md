## Prototype-based interpretable community detection

## Introduction:
We propose a novel interpretable community detection algorithm that represents each community via a prototype structure—a central node (the most representative member) and its coverage radius (defining community size). This formulation enables intuitive and explainable graph partitioning.
To optimize community assignments, we frame the problem as an objective optimization task, simultaneously minimizing:
1. Uncovered nodes (to ensure broad community membership)
2. Overlapping nodes (to maintain partition clarity)
Our solution employs an efficient heuristic algorithm that:
* Dynamically selects optimal central nodes based on representativeness metrics
* Adaptively adjusts coverage radii for balanced community sizes

## Features
- ​**Interpretable Community Detection**: Uses center nodes and radii to define communities 
- ​**Optimization Process**: 
  - Coordinated radius search
  - Center node selection based on centrality measures
  - Iterative improvement of community boundaries
- ​**Metrics Calculation**:
  - Overlap ratio
  - Uncovered ratio
- ​**Performance Tracking**: Measures and records processing times

## Requirements

- Python 3.7+
- Required packages:
  - networkx
  - numpy
  - scipy
  - matplotlib
  - scikit-learn
  - tqdm
  - pandas


# In PICD initialization:
a = 0.5          # Weight for overlap ratio in objective function
alpha = 0.8      # Weight for degree centrality in center selection
max_iter = 100   # Maximum iterations for optimization

# Dataset-specific k values:
k_values = {
    'karate': 2,
    'football': 12,
    'personal': 8,
    'polblogs': 2,
    'railways': 21,
    'email': 42,
    'polbooks': 3
}

# Input Data
Place your graph data in data/real/graph/ directory as .txt files with edge lists:
markdown

node1 node2
node1 node3
...
# Output
Results are including:
* Each community has a corresponding central node and radius, and the overall uncovered rate and repeated coverage rate of the graph. - Interpretable metrics
* {dataset_name}_communities.txt - Detected communities
* processing_times.csv - Execution times for each dataset

# example datasets:
* karate
* football
* personal
* polblogs
* railways
* email
* polbooks

# Citation
If you use this code in your research, please cite:


