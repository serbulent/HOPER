# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

"""from zipfile import ZipFile
with ZipFile('intact.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()"""

import pandas as pd
intact_df=pd.read_csv("intact.txt", sep='\t')

intact_df_filter=intact_df.filter(['#ID(s) interactor A' , 'ID(s) interactor B'])

#a=intact_df_filter['#ID(s) interactor A',] = intact_df_filter['#ID(s) interactor A'].str.replace('uniprotkb:','')
#intact_df_filter.replace({'#ID(s) interactor A':'uniprotkb:','ID(s) interactor B':'uniprotkb:'}, {'#ID(s) interactor A':'','ID(s) interactor B':''}, regex=True)
intact_df_filter_Nan=intact_df_filter.replace(regex=r'uniprotkb:', value='')
#intact_df_filterintact_df_filter=intact_df.filter(['#ID(s) interactor A' , 'ID(s) interactor B'])

drop_values = [':','_', '-',',']
y=intact_df_filter_Nan[~intact_df_filter_Nan['#ID(s) interactor A'].str.contains('|'.join(drop_values))]
z=y[~y['ID(s) interactor B'].str.contains('|'.join(drop_values))]
z

intact_df_dub=z.drop_duplicates()


intact_df_reset_index=intact_df_dub.reset_index()

intact_df_raw_sorted=intact_df_reset_index.sort_values(by=['#ID(s) interactor A'])
intact_df_raw_sorted_reset=intact_df_raw_sorted.reset_index()
intact_df_raw_sorted_reset

intact_df_raw_sorted_reset_filter=intact_df_raw_sorted_reset.filter(['#ID(s) interactor A' , 'ID(s) interactor B'])
intact_df_raw_sorted_reset_filter

intact_df_raw_sorted_reset_filter=intact_df_raw_sorted_reset.filter(['#ID(s) interactor A' , 'ID(s) interactor B'])
intact_df_raw_sorted_reset_filter

#intact_df_raw_sorted_reset_filter.to_excel("intact_raw_data_preprocessing_new.xlsx")

protein_df=pd.read_excel("/media/DATA2/testuser2/HOPER/datam/hoper_PPI/human_proteins.xlsx")
protein_df

intact_data=intact_df_raw_sorted_reset_filter#pd.read_excel("intact_raw_data_preprocessing_new.xlsx")
intact_data

intact_data_frame=intact_data.filter(['#ID(s) interactor A','ID(s) interactor B'])
intact_data_frame

human_protein=pd.DataFrame()
human_protein=protein_df.filter(items=['A'])
human_protein

intact_data_frame.columns

filtered_intact_df = intact_data_frame[
    intact_data_frame["#ID(s) interactor A"].isin(human_protein["A"]) &
    intact_data_frame["ID(s) interactor B"].isin(human_protein["A"])
]

filtered_intact_df
intact_data_frame_filter_sorted=filtered_intact_df.filter(['#ID(s) interactor A','ID(s) interactor B'])

intactdata_dataframe=intact_data_frame_filter_sorted.reset_index().drop(["index"],axis=1)

import pandas as pd
#intactdata_dataframe=pd.read_excel("/content/intact_data_frame_filter_sorted.xlsx")
intactdata_dataframe=intact_data_frame_filter_sorted
print(intactdata_dataframe)
protein=[]
protein=intactdata_dataframe['#ID(s) interactor A']
interactionprotein=intactdata_dataframe['ID(s) interactor B']

list =[]
for col in intactdata_dataframe.columns:
    val = intactdata_dataframe[col]
    for v in val:
        list.append(v)

lengt=intactdata_dataframe.shape[0]
from collections import defaultdict
temp = defaultdict(lambda: len(temp))
res = [temp[ele] for ele in list]
P=res[0:lengt]
i=res[lengt:len(res)]
intactdata_dataframe= pd.DataFrame()
intactdata_dataframe['#ID(s) interactor A']=P
intactdata_dataframe['ID(s) interactor B']=i
print(intactdata_dataframe)

intactdata_dataframe.to_csv("intactdata_dataframe_filter_human_proteins_new.edgelist",header=False, sep=' ', index=False)

res = [temp[ele] for ele in list]
P=res[0:lengt]
i=res[lengt:len(res)]
intactdata_dataframe= pd.DataFrame()
intactdata_dataframe['#ID(s) interactor A']=P
intactdata_dataframe['ID(s) interactor B']=i
print(intactdata_dataframe)

print(temp.items())
print(len(temp.values()))
print(temp.values())
print(temp.keys())
keys=temp.keys()
type(keys)

newlist =[]
for i in temp.keys():
    newlist.append(i)

protein_id_df=pd.DataFrame(newlist)

protein_id_df.to_csv("proteinliste_id_new.csv")

protein_id_df

