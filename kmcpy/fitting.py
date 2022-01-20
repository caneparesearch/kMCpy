"""
Fitting is a class to store the fitted results

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import json
from kmcpy.kmc_tools import convert

class Fitting:
    def __init__(self):
        pass

    def add_data(self,time_stamp,keci,empty_cluster,weight,alpha,rmse,loocv):
        self.time_stamp = time_stamp
        self.weight = weight
        self.alpha = alpha
        self.keci = keci
        self.empty_cluster = empty_cluster
        self.rmse = rmse
        self.loocv =loocv
        
    def as_dict(self):
        d = {"@module":self.__class__.__module__,
        "@class": self.__class__.__name__,
        "time_stamp":self.time_stamp,
        "weight":self.weight,
        "alpha":self.alpha,
        "keci":self.keci,
        "empty_cluster":self.empty_cluster,
        "rmse":self.rmse,
        "loocv":self.loocv}
        return d

    def to_json(self,fname):
        print('Saving:',fname)
        with open(fname,'w') as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(d,indent=4,default=convert) # to get rid of errors of int64
            fhandle.write(jsonStr)
    
    @classmethod
    def from_json(self,fname):
        print('Loading:',fname)
        with open(fname,'rb') as fhandle:
            objDict = json.load(fhandle)
        obj = Fitting()
        obj.__dict__ = objDict
        return obj