# GEM: Graph Embedding Methods
GEM is a Python package which offers a general framework for graph embedding methods.

Please refer https://palash1992.github.io/GEM/ to access the readme as a webpage.


## Implemented Methods
* node2vec

* Higher-Order Proximity preserved Embedding (HOPE)

## Graph Format
We used undirected graph as protein-protein interaction data.

## Data Format
### Edge List
-Read and write NetworkX graphs as edge lists.

-With the edgelist format simple edge data can be stored

*Example:

Node pairs 

1 2

1 3

2 3

Please refer https://github.com/serbulent/HOPER/blob/main/Reproduction/ppi_representations/intact_data_preprocess.py 

#### How to run Methods

* Create edgelist (input data)

* İdentify the protein names corresponding to the nodes

* Set parameters

* Determine directed or undirected graph

* Create embeddings



