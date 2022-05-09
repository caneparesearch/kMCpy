#!/usr/bin/env python
import numpy as np
import pandas as pd
import glob2,os,json

def gather_data(path,shape):
    locations = glob2.glob(path)
    data = []
    print(locations)
    for location in locations:
        x = float(location.split('_')[1])
        structure_index = int(location.split('_')[2])
        occ = load_occ(location,shape)
        data.append([x,structure_index,occ])
    return pd.DataFrame(data,columns=['comp','structure_index','occ']).sort_values(by=['comp','structure_index'])

def load_occ(mc_result,shape,select_sites=[0,1,2,3,4,5,6,7,12,13,14,15,16,17]):
    print(mc_result)
    with open(mc_result+'/conditions.0/final_state.json','r') as f:
        occupation = (np.array(json.load(f)['occupation']).reshape((42,)+shape)[select_sites].flatten('C')) # the global occupation array in the format of (site,x,y,z)
    occupation_chebyshev = np.where(occupation==0, -1, occupation)  # replace 0 with -1 for Chebyshev basis
    return occupation_chebyshev
if __name__=="__main__":
    df = gather_data('comp*',(2,1,1))
    print(df)
    df.to_hdf('mc_results.h5',key='df',complevel=9,mode='w')
