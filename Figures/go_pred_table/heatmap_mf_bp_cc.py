import pandas as pd
import glob
import os
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import pixiedust
pd.set_option('display.max_colwidth', None)
pd.options.display.max_rows = 4000

go_pred_tableF1_MF_filter_dropna_node2vec = pd.read_csv("/home/isik/signifi_new/go_pred_tableF1_MF_filter_dropna_node2vec.csv")
go_pred_tableF1_MF_filter_dropna_node2vec_index=go_pred_tableF1_MF_filter_dropna_node2vec.set_index('index_col')
#print(go_pred_tableF1_MF_filter_dropna_node2vec_index)

#print(go_pred_tableF1_MF_filter_dropna_node2vec_index)
go_pred_tableF1_MF_filter_dropna_HOPE = pd.read_csv("/home/isik/signifi_new/go_pred_tableF1_MF_filter_dropna_HOPE.csv")
go_pred_tableF1_MF_filter_dropna_HOPE_index=go_pred_tableF1_MF_filter_dropna_HOPE.set_index('index_col')

#print(go_pred_tableF1_MF_filter_dropna_node2vec)
#print(go_pred_tableF1_MF_filter_dropna_HOPE)


go_pred_tableF1_CC_filter_node2vec = pd.read_csv("/home/isik/signifi_new/go_pred_tableF1_CC_filter_node2vec.csv")
go_pred_tableF1_CC_filter_node2vec_index=go_pred_tableF1_CC_filter_node2vec.set_index('index_col')
go_pred_tableF1_CC_filter_HOPE = pd.read_csv("/home/isik/signifi_new/go_pred_tableF1_CC_filter_HOPE.csv")
go_pred_tableF1_CC_filter_HOPE_index=go_pred_tableF1_CC_filter_HOPE.set_index('index_col')
'''
go_pred_tableF1_BP_filter_node2vec = pd.read_csv("/media/DATA2/isik/gcloud_isik/isik/makale_heatmap_deneme/go_pred_tableF1_BP_filter_node2vec.csv")
go_pred_tableF1_BP_filter_node2vec_index=go_pred_tableF1_BP_filter_node2vec.set_index('index_col')
go_pred_tableF1_BP_filter_HOPE = pd.read_csv("/media/DATA2/isik/gcloud_isik/isik/makale_heatmap_deneme/go_pred_tableF1_BP_filter_HOPE.csv")
go_pred_tableF1_BP_filter_HOPE_index=go_pred_tableF1_BP_filter_HOPE.set_index('index_col')

'''
result_go_pred_tableF1_MF_filter_dropna_node2vec_HOPE_concat = pd.concat([go_pred_tableF1_MF_filter_dropna_node2vec_index, go_pred_tableF1_MF_filter_dropna_HOPE_index], axis=1)
#print(result_go_pred_tableF1_MF_filter_dropna_node2vec_HOPE_concat.columns)

MF_column_rename=result_go_pred_tableF1_MF_filter_dropna_node2vec_HOPE_concat.rename(columns={"node_d_1000_p_2_q_0.25": "NODE2VEC_d_1000_p_2_q_0.25", "node_d_1000_p_2_q_0.5": "NODE2VEC_d_1000_p_2_q_0.5","node_d_1000_p_2_q_1": "NODE2VEC_d_1000_p_2_q_1","node_d_100_p_0.25_q_0.5":"NODE2VEC_d_100_p_0.25_q_0.5","node_d_100_p_0.5_q_1": "NODE2VEC_d_100_p_0.5_q_1","node_d_100_p_2_q_0.5": "NODE2VEC_d_100_p_2_q_0.5","node_d_200_p_2_q_0.25": "NODE2VEC_d_200_p_2_q_0.25","node_d_500_p_0.5_q_0.25": "NODE2VEC_d_500_p_0.5_q_0.25","node_d_500_p_1_q_2": "NODE2VEC_d_500_p_1_q_2","node_d_500_p_2_q_0.25": "NODE2VEC_d_500_p_2_q_0.25"})

result_go_pred_tableF1_CC_filter_node2vec_HOPE_concat = pd.concat([go_pred_tableF1_CC_filter_node2vec_index, go_pred_tableF1_CC_filter_HOPE_index], axis=1)

#print(result_go_pred_tableF1_CC_filter_node2vec_HOPE_concat.columns)

CC_column_rename=result_go_pred_tableF1_CC_filter_node2vec_HOPE_concat.rename(columns={"node_d_1000_p_0.25_q_0.5": "NODE2VEC_d_1000_p_0.25_q_0.5", "node_d_1000_p_0.25_q_1": "NODE2VEC_d_1000_p_0.25_q_1","node_d_1000_p_0.5_q_0.5": "NODE2VEC_d_1000_p_0.5_q_0.5","node_d_1000_p_0.5_q_1": "NODE2VEC_d_1000_p_0.5_q_1","node_d_1000_p_1_q_2": "NODE2VEC_d_1000_p_1_q_2","node_d_1000_p_2_q_0.5": "NODE2VEC_d_1000_p_2_q_0.5","node_d_1000_p_2_q_1": "NODE2VEC_d_1000_p_2_q_1","node_d_200_p_2_q_1": "NODE2VEC_d_200_p_2_q_1","node_d_500_p_2_q_1": "NODE2VEC_d_500_p_2_q_1","node_d_500_p_2_q_2": "NODE2VEC_d_500_p_2_q_2"})
'''

result_go_pred_tableF1_BP_filter_node2vec_HOPE_concat = pd.concat([go_pred_tableF1_BP_filter_node2vec_index, go_pred_tableF1_BP_filter_HOPE_index], axis=1)
#print(result_go_pred_tableF1_BP_filter_node2vec_HOPE_concat)

'''
'''
BP_column_rename=result_go_pred_tableF1_BP_filter_node2vec_HOPE_concat.rename(columns={"node_d_1000_p_2_q_1": "NODE2VEC_d_1000_p_2_q_1", "node_d_100_p_0.25_q_0.25": "NODE2VEC_d_100_p_0.25_q_0.25","node_d_200_p_1_q_2": "NODE2VEC_d_200_p_1_q_2","node_d_500_p_0.5_q_0.25": "NODE2VEC_d_500_p_0.5_q_0.25","node_d_500_p_0.5_q_2": "NODE2VEC_d_500_p_0.5_q_2","node_d_500_p_1_q_1": "NODE2VEC_d_500_p_1_q_1","node_d_500_p_1_q_2": "NODE2VEC_d_500_p_1_q_2","node_d_500_p_2_q_1": "NODE2VEC_d_500_p_2_q_1","node_d_500_p_2_q_2": "NODE2VEC_d_500_p_2_q_2","node_d_50_p_1_q_2": "NODE2VEC_d_50_p_1_q_2"})
'''
'''
'''
#dict=CC
group_color_dict= {'NODE2VEC_d_1000_p_1_q_2':'orange','NODE2VEC_d_1000_p_0.25_q_0.5':'orange','NODE2VEC_d_200_p_2_q_1':'orange','NODE2VEC_d_1000_p_2_q_0.5':'orange','NODE2VEC_d_1000_p_2_q_1':'orange','NODE2VEC_d_500_p_2_q_2':'orange','NODE2VEC_d_1000_p_0.5_q_1':'orange','NODE2VEC_d_500_p_2_q_1':'orange','NODE2VEC_d_1000_p_0.25_q_1':'orange','NODE2VEC_d_1000_p_0.5_q_0.5':'orange','HOPE_d_500_beta_0.0625':'blue','HOPE_d_1000_beta_0.0625':'blue','HOPE_d_500_beta_0.125':'blue','HOPE_d_500_beta_0.25':'blue','HOPE_d_100_beta_0.125':'blue','HOPE_d_1000_beta_0.25':'blue','HOPE_d_200_beta_0.25':'blue','HOPE_d_1000_beta_0.125':'blue','HOPE_d_200_beta_0.125':'blue','HOPE_d_200_beta_0.0625':'blue'}
#print(group_color_dict)
'''
#dict=MF
'''
group_color_dict={'NODE2VEC_d_200_p_2_q_0.25':'orange','NODE2VEC_d_500_p_0.5_q_0.25':'orange','NODE2VEC_d_1000_p_2_q_0.25':'orange','NODE2VEC_d_100_p_0.5_q_1':'orange','NODE2VEC_d_500_p_2_q_0.25':'orange','NODE2VEC_d_100_p_0.25_q_0.5':'orange','NODE2VEC_d_1000_p_2_q_0.5':'orange','NODE2VEC_d_1000_p_2_q_1':'orange','NODE2VEC_d_100_p_2_q_0.5':'orange','NODE2VEC_d_500_p_1_q_2':'orange','HOPE_d_1000_beta_0.25':'blue','HOPE_d_1000_beta_0.125':'blue','HOPE_d_500_beta_0.25':'blue','HOPE_d_200_beta_0.25':'blue','HOPE_d_200_beta_0.125':'blue','HOPE_d_500_beta_0.0625':'blue','HOPE_d_200_beta_0.0625':'blue','HOPE_d_100_beta_0.25':'blue','HOPE_d_500_beta_0.125':'blue','HOPE_d_1000_beta_0.0625':'blue'}

'''
#dict=BP
group_color_dict={'NODE2VEC_d_1000_p_2_q_1':'orange','NODE2VEC_d_200_p_1_q_2':'orange', 'NODE2VEC_d_500_p_2_q_2':'orange', 'NODE2VEC_d_500_p_1_q_2':'orange','NODE2VEC_d_100_p_0.25_q_0.25':'orange','NODE2VEC_d_500_p_0.5_q_2':'orange','NODE2VEC_d_500_p_1_q_1':'orange','NODE2VEC_d_500_p_2_q_1':'orange','NODE2VEC_d_50_p_1_q_2':'orange','NODE2VEC_d_500_p_0.5_q_0.25':'orange','HOPE_d_1000_beta_0.25':'blue','HOPE_d_500_beta_0.25':'blue','HOPE_d_1000_beta_0.5':'blue','HOPE_d_200_beta_0.25':'blue','HOPE_d_1000_beta_0.125':'blue','HOPE_d_100_beta_0.25':'blue','HOPE_d_500_beta_0.125':'blue','HOPE_d_500_beta_0.0625':'blue','HOPE_d_500_beta_0.5':'blue','HOPE_d_1000_beta_0.0625':'blue'}



def set_colors_and_marks_for_representation_groups(ax):
    for label in ax.get_xticklabels():
        label.set_color(group_color_dict[label.get_text()])
        lt = label.get_text()
        #if  lt == 'node_d_10_p_0.5_q_0.25' or lt =='node_d_10_p_2_q_1' or lt =='HOPE_d_500_beta_0.015625' or lt =='HOPE_d_1000_beta_0.015625':
           # signed_text = "^" + label.get_text()
            #label.set_text(signed_text)
    fontproperties = {'weight' : 'bold'}
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties)


g = sns.clustermap(MF_column_rename , annot=True, cmap="YlGnBu", row_cluster=False,figsize=(15, 15))
ax = g.ax_heatmap
ax.set_xlabel("")
ax.set_ylabel("")
set=set_colors_and_marks_for_representation_groups(ax)

g.savefig('/media/DATA2/isik/gcloud_isik/isik/result_go_pred_tableF1_MF_filter_heat.png')
