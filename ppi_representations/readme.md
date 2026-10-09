# Protein-Protein Interaction (PPI) Representation

Protein-Protein Interaction (PPI) representation refers to the various ways in which the interactions between proteins can be represented or encoded. PPI representations aim to capture the structural, functional, and relational aspects of protein interactions and are used in various computational methods and analyses.
Methods used for representation:

* [Node2vec](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

* [Higher-Order Proximity preserved Embedding (HOPE)](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)

Node2Vec and HOPE are a popular algorithm used for generating node embeddings in network analysis, including graph-based protein representations. It is a representation learning method that learns low-dimensional vector representations, or embeddings, for nodes in a graph.

Please refer https://palash1992.github.io/GEM/ to access the readme as a webpage.

## Dependencies

`bash create_env.sh` (repository root) creates the `hoper_PPI` environment from **hoper_PPI.yml**, installs
[GEM](https://github.com/palash1992/GEM) at commit `213189b` and builds the SNAP `node2vec` binary into
`ppi_representations/bin/`. No manual installation is needed.

## Node2vec parameters
| Parameter  |Description|  Value |
| ------------| ------------| ------------|
|       d     |  embedding dimension   | 10, 50, 100, 200, 500, 1000  |
|       p     |        return parameter(Parameter p controls the likelihood of immediately revisiting a node in the walk) |  0.25, 0.5, 1, 2 |
|       q     |       In-out parameter(Parameter q allows the search to differentiate between “inward” and “outward” nodes) | 0.25, 0.5, 1, 2|
|   max_iter  |        maximum iterations    | 1  |
|   walk_len  |        random walk length    |  80 |
|   con_size  |        context size    |  10 |
|   num_walks |        number of random walks    |10 |




## HOPE parameters

| Parameter  |Description |  Value   | 
| ------------| ------------|------------|
|       d     |  embedding dimension   |10, 50, 100, 200, 500, 1000 |
|      beta     |  decay factor  | 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5 |



## Data Format
### Edge List
-Read and write NetworkX graphs as edge lists.

-With the edgelist format simple edge data can be stored

*Example:
 
 <table>
<tr><th> Interaction data </th><th></th><th></th><th> Edgelist Data </th></tr>
<tr><td>
 
|Interaction A|Interaction B|                
| ------------| ------------|
|  P05089     |   P05362    |
|  P05362	    |   P14902    |
|  P14902     |   P16410    |
|  P15692     |   P14902    |
|  P16070     |   P14902    |
|  P16410     |   P05362    |

</td><td></th><th></th><th>
 
|Interaction A|Interaction B|
| ------------| ------------|
|  0    |   1    |
|  1    |   2    |
|  2    |   5    |
|  3    |   2    |
|  4    |   2    |
|  5    |   1    |

</td></tr> </table>


#### How to run methods

Inputs: an edge list (`.edgelist`, node indices) and a CSV that maps node indices to protein ids (column `0`).
The example data (`bash download_data.sh`) contains a small network in `data/hoper_PPI/PPI_example_data/`
(`example.edgelist`, `proteins_id.csv`).

Run from the repository root with the launcher (`choice_of_module: [PPI]` in `Hoper_representation_generetor.yaml`,
parameters as in the tables above):

```shell
conda activate hoper
python Hoper_representation_generetor_main.py
```

or directly, in the `hoper_PPI` environment (list arguments are JSON/Python lists):

```shell
conda activate hoper_PPI
python ppi_representations/Node2vec.py data/hoper_PPI/PPI_example_data/example.edgelist data/hoper_PPI/PPI_example_data/proteins_id.csv False "[10]" "[0.25]" "[0.25]"
python ppi_representations/HOPE.py data/hoper_PPI/PPI_example_data/example.edgelist data/hoper_PPI/PPI_example_data/proteins_id.csv False "[5]" "[0.00390625]"
```

The third argument is `is_directed` (`False` for an undirected interaction network).
Outputs are written to `data/Node2vec_d_<d>_p_<p>_q_<q>.pkl` and `data/HOPE_d_<d>_beta_<beta>.pkl`
(pandas DataFrames with columns `Entry` and `Vector`).

`edgelist_code.py` and `data_preprocess.py` (IntAct preprocessing) are the scripts used to build the paper's
network; they still contain the authors' local paths and are not part of the tested workflow.



