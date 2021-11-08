#!/usr/bin/env python
import numpy as np
import pandas as pd
import glob2,os,json

"""
KMC data gathering
Here are the format for each results file
self.results = {'D_J':[],'D_tracer':[],'conductivity':[],'f':[],'H_R':[],
        'occupation':[],'ekra':[],'displacement':[],'hop_count':[]}

shape of matrices
hop_count[n_pass,n_na]
displacement[n_pass,n_na,3]
ekra[n_pass]
"""
def search_string_list(fname,word,location):
    res = []
    with open(fname) as f:
        for l in f.readlines():
            if word in l:
                res.append(float(l.strip().split()[location]))
    return res

def gather_data(path):
    locations = glob2.glob(path)
    data = []
    print(locations)
    for location in locations:
        x = float(location.split('.csv.gz')[0].split('_')[1])
        structure_index = int(location.split('.csv.gz')[0].split('_')[2])
        print(x,structure_index)
        df = pd.read_csv(location,compression='gzip')
        df['comp']=x
        df['structure_index']=structure_index
        
        data.append(df)

    df_merged = pd.concat(data,axis=0)
    df_merged.index.rename('n_pass',inplace=True)
    df_merged.reset_index(inplace=True)
    # df_matrices = df_merged[['comp','structure_index','time','hop_count','displacement','ekra','occupation']]
#    df_merged.drop(['hop_count','displacement','ekra','occupation'],axis=1,inplace=True)
    return df_merged


df = gather_data('results*.csv.gz')
print(df.columns)
print(df)
df.to_csv('gathered_results.csv.gz',compression='gzip')
# print(df.D_J)

