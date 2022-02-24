"""
IO takes dictionary like object and convert them into json writable string

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import numpy as np
import json
# class IO:
#     def __init__(self):
#         pass

#     def to_json(self,fname):
#         print('Saving:',fname)
#         with open(fname,'w') as fhandle:
#             d = self.as_dict()
#             jsonStr = json.dumps(d,indent=4,default=convert) # to get rid of errors of int64
#             fhandle.write(jsonStr)
    
def convert(o):
    if isinstance(o, np.int64): return int(o)
    elif isinstance(o, np.int32): return int(o)  
    raise TypeError

def load_occ(fname,shape,select_sites=[0,1,2,3,4,5,6,7,12,13,14,15,16,17]):
    with open(fname,'r') as f:
        occupation = (np.array(json.load(f)['occupation']).reshape((42,)+shape)[select_sites].flatten('C')) # the global occupation array in the format of (site,x,y,z)
    occupation_chebyshev = np.where(occupation==0, -1, occupation)  # replace 0 with -1 for Chebyshev basis
    return occupation_chebyshev

# to be developed
def input_reader():
    """
    input_reader takes input (a json file with all parameters as shown in run_kmc.py in examples folder)

    return will be a dictionary with all input parameters
    """
    
    return 0