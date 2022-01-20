"""
This file contains auxillary functions

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import numpy as np

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError