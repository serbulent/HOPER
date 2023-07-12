# Protein-Protein Interaction (PPI) Representation

Protein-Protein Interaction (PPI) representation refers to the various ways in which the interactions between proteins can be represented or encoded. PPI representations aim to capture the structural, functional, and relational aspects of protein interactions and are used in various computational methods and analyses.
Methods used for representation:

* [Node2vec](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

* [Higher-Order Proximity preserved Embedding (HOPE)](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)

Node2Vec and HOPE are a popular algorithm used for generating node embeddings in network analysis, including graph-based protein representations. It is a representation learning method that learns low-dimensional vector representations, or embeddings, for nodes in a graph.

Please refer https://palash1992.github.io/GEM/ to access the readme as a webpage.

## Dependencies

Related dependencies are available in the **ppi_environment.yml** file. Related dependencies can be installed by importing **ppi_environment.yml** file.

## Node2vec parameters
| Parameter  |Description|  Value |
| ------------| ------------| ------------|
|       d     |  embedding dimension   | 10, 50, 100, 200, 500, 1000  |
|     ret_p   |        return parameter    |  0.25, 0.5, 1, 2 |
|    inout_p  |       In-out parameter    |   0.25, 0.5, 1, 2  |
|   max_iter  |        maximum iterations    | 1  |
|   walk_len  |        random walk length    |  80 |
|   con_size  |        context size    |  10 |
|   num_walks |         number of random walks    |10 |




## HOPE parameters

| Parameter  |Description |  Value   | 
| ------------| ------------|------------|
|       d     |  embedding dimension   |10, 50, 100, 200, 500, 1000 |
|      beta     |  decay factor  | 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5 |



## Graph Format
We used undirected graph as protein-protein interaction data.

## Data Format
### Edge List
-Read and write NetworkX graphs as edge lists.

-With the edgelist format simple edge data can be stored

*Example:

 Interaction data
 
 <table>
<tr><th>Table 1 Heading 1 </th><th>Table 1 Heading 2</th></tr>
<tr><td>
 
|Interaction A|Interaction B|                
| ------------| ------------|
|  P05089     |   P05362    |
|  P05362	    |   P14902    |
|  P14902     |   P16410    |
|  P15692     |   P14902    |
|  P16070     |   P14902    |
|  P16410     |   P05362    |

</td><td>
 
|Interaction A|Interaction B|
| ------------| ------------|
|  0    |   1    |
|  1    |   2    |
|  2    |   5    |
|  3    |   2    |
|  4    |   2    |
|  5    |   1    |

</td></tr> </table>

Please refer https://github.com/serbulent/HOPER/blob/main/Reproduction/ppi_representations/intact_data_preprocess.py 

#### How to run Methods

* Create edgelist (input data)

* To install packages to use for Node2vec and HOPE in your home directory, use:

  pip install git+https://github.com/palash1992/GEM.git

  * GEM version 213189b; use for old version:
  
    git clone [https://github.com/palash1992/GEM.git]
    
    git checkout  [213189b]

*To make Node2vec executable; Clone repository   https://github.com/snap-stanford/snap
*Compiles SNAP

    cd snap-master/
       rm -rf examples/Release
          make all
              cd examples/node2vec
                   chmod +x node2vec
                        ls -alh node2vec

* Make node2vec executable and add to system PATH or move it to the location you run.

* Identify the protein names corresponding to the nodes(.edgelist)

* You can use small sample for application (The sample interaction is randomly generated) (data konum) 

* Preprocessing is required for the IntAct database. The relevant code for this  ()
 
* Set parameters

* Create representations



It can be run  as python Node2vec.py and HOPE.py



