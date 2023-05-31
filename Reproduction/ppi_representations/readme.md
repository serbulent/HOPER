# Protein-Protein Interaction (PPI) Representation

Protein-Protein Interaction (PPI) representation refers to the various ways in which the interactions between proteins can be represented or encoded. PPI representations aim to capture the structural, functional, and relational aspects of protein interactions and are used in various computational methods and analyses.
Methods used for representation:

* [Node2vec](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

* [Higher-Order Proximity preserved Embedding (HOPE)](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)

Node2Vec and HOPE are a popular algorithm used for generating node embeddings in network analysis, including graph-based protein representations. It is a representation learning method that learns low-dimensional vector representations, or embeddings, for nodes in a graph.

Please refer https://palash1992.github.io/GEM/ to access the readme as a webpage.

## Node2vec parameters
| Parameter  |Description|
| ------------| ------------|
|       d     |  embedding dimension   |
|     ret_p   |        return parameter    |
|    inout_p  |       In-out parameter    |
|   max_iter  |        maximum iterations    |
|   walk_len  |        random walk length    |
|   con_size  |        context size    |
|   num_walks |         number of random walks    |


d = [10,50,100,200,500,1000],
p = [0.25,0.5,1,2],
q = [0.25,0.5,1,2],
max_iter=1,
walk_len=80,
num_walks=10,
con_size=10

## HOPE parameters

| Parameter  |Description|
| ------------| ------------|
|       d     |  embedding dimension   |
|      beta     |  decay factor  |

d = [10,50,100,200,500,1000],
beta = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5]

## Graph Format
We used undirected graph as protein-protein interaction data.

## Data Format
### Edge List
-Read and write NetworkX graphs as edge lists.

-With the edgelist format simple edge data can be stored

*Example:
 
|Interaction A|Interaction B|
| ------------| ------------|
|       1     |        2    |
|       1     |        3    |
|       2     |        3    |


Please refer https://github.com/serbulent/HOPER/blob/main/Reproduction/ppi_representations/intact_data_preprocess.py 

#### How to run Methods

* Create edgelist (input data)

* İdentify the protein names corresponding to the nodes

* Set parameters

* Determine directed or undirected graph

* Create representations

It can be run  as python Node2vec.py and HOPE.py



