
import matplotlib.pyplot as plt
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time
import pandas as pd
import pickle
from gem.embedding.hope     import HOPE

# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
edge_f="/media/DATA/home/isik/hope_pkls/intactdata_dataframe_filter_human_proteins.edgelist"
# Specify whether the edges are not directed
isDirected =False
d = [10,50,100,200,500,1000]
beta = [0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5]
def hope_repesentation_call(edge_f,isDirected,d,beta):
# Load graph

    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed =isDirected )
    G = G.to_directed()



    for x in d:

        for y in beta:
      
            models = []

        #HOPE takes embedding dimension (d) and decay factor (beta) as inputs
            models.append(HOPE(d=x, beta=y))
        
            for embedding in models:
                print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
                t1 = time()
            # Learn embedding - accepts a networkx graph or file with edge list
                Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=False, no_python=True)
                print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
            # Evaluate on graph reconstruction
                MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
            #---------------------------------------------------------------------------------
                print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
                emb= embedding.get_embedding()
    return emb
            
emb=hope_repesentation_call(edge_f,isDirected,d,beta)            
emb_list=list(emb)
protein_id=pd.read_csv("/media/DATA/home/isik/hope_pkls/proteins_id.csv")
          
protein_id_list=protein_id['0'].tolist()
   
ent_vec = {'Entry':protein_id_list,'Vector':emb_list}
            
ent_vec_data_frame = pd.DataFrame(ent_vec)
    
output = open('HOPE_'+'d_'+str(x) + '_' +'beta_'+str(y) +'.pkl', 'wb')
pickle.dump(ent_vec_data_frame, output)
output.close()
    
#viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
#plt.show()