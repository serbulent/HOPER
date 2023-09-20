

import matplotlib.pyplot as plt
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time
import pandas as pd
import pickle
from gem.embedding.node2vec import node2vec
import os


# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
#edge_f="/media/DATA2/isik/HOPER/Reproduction/ppi_representations/GEM/examples/data/deneme.edgelist"
#protein_id=pd.read_csv("/media/DATA2/isik/HOPER/Reproduction/ppi_representations/proteins_id_deneme.csv")
# Specify whether the edges are not directed
#isDirected =False
#d = [2,20]
#p = [0.25,1]
#q = [0.25]

def node2vec_repesentation_call(edge_f,protein_id,isDirected,d,p,q):
# Load graph
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed =isDirected )
    G = G.to_directed()
    for i in d:
    
      for j in p:
      
        for k in q:
        
          models = []

        
        #node2vec takes embedding dimension (d),  maximum iterations (max_iter), random walk length (walk_len), number of random walks (num_walks), context size (con_size), return             weight (re#t_p), inout weight (inout_p) as inputs
          models.append(node2vec(d=i,max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=j, inout_p=k))
          for embedding in models:
            print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
            t1 = time()
            # Learn embedding - accepts a networkx graph or file with edge list
            Y, t = embedding.learn_embedding(graph=G, is_weighted=False, no_python=True)
            print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
            # Evaluate on graph reconstruction
            MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
            #---------------------------------------------------------------------------------
            print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
            emb= embedding.get_embedding()
            print(emb)
            emb_list=list(emb)

          
            protein_id_list=protein_id['0'].tolist()
   
            ent_vec = {'Entry':protein_id_list,'Vector':emb_list}
            
            ent_vec_data_frame = pd.DataFrame(ent_vec)
            path=os.getcwd()
            
            output = open(os.path.join(path,"ppi_representations/data",Node2vec_'+'d_'+str(i) + '_' +'p_'+str(j) +  '_' +'q_'+str(k) +'.pkl'), 'wb')
            pickle.dump(ent_vec_data_frame, output)
            output.close()   
            
#node2vec_repesentation_call(edge_f,protein_id,isDirected,d,p,q)           


