"""
Fitting is a class to generate and store the fitted results. This is the class for generating fitting_results.json

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
import json
from kmcpy.io import convert

class Fitting:
    """
    main function of fitting the NEB result to the kinetic monte carlo input
    """
    def __init__(self):
        pass

    def add_data(self,time_stamp,time,keci,empty_cluster,weight,alpha,rmse,loocv):
        self.time_stamp = time_stamp
        self.time = time
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

    """
    Fitting

    E_KRA[m x 1] =  diagonal(Weight)[m x m] * Corr[m x n] * ECI[n x 1] + V_0[m x 1]

    E_KRA is a m x 1 vector
    ECI is a n x 1 vector
    Corr is a m x n matrix
    V_0 is a m x 1 vector
    Weight is a n x n diagonal matrix
    m is the number of E_KRA
    n is the number of clusers

    Lasso estimator is used

    correlation_matrix is stored in correlation_matrix.txt
    e_kra is stored in e_kra.txt
    """

    def fit(self,alpha,max_iter=1000000,ekra_fname='e_kra.txt',keci_fname='keci.txt',
    weight_fname='weight.txt',corr_fname='correlation_matrix.txt',
    fit_results_fname='fitting_results.json'):
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_squared_error

        from copy import copy
        from datetime import datetime
        import numpy as np
        import pandas as pd

        print('Loading E_KRA from',ekra_fname, '...')
        e_kra = np.loadtxt(ekra_fname)
        weight = np.loadtxt(weight_fname)
        weight_copy = copy(weight)
        correlation_matrix = np.loadtxt(corr_fname)

        alpha = alpha
        estimator = Lasso(alpha=alpha, max_iter=max_iter,
                           fit_intercept=True)
        estimator.fit(correlation_matrix, e_kra, sample_weight=weight)
        keci = estimator.coef_
        empty_cluster = estimator.intercept_
        print('Lasso Results:')
        print('KECI = ')
        print(np.round(keci, 2))
        print('There are ', np.count_nonzero(abs(keci) > 1e-2), 'Non Zero KECI')
        print('Empty Cluster = ')
        print(empty_cluster)

        y_true = e_kra
        y_pred = estimator.predict(correlation_matrix)
        print('index\tNEB\tLCE\tNEB-LCE')
        index = np.linspace(1, len(y_true), num=len(y_true), dtype='int')
        print(
            np.round(np.array([index, y_true, y_pred, y_true-y_pred]).T, decimals=2))

        # cv = sqrt(mean(scores)) + N_nonzero_eci*penalty, penalty = 0 here
        scores = -1 * cross_val_score(estimator=estimator, X=correlation_matrix,
                                      y=e_kra, scoring='neg_mean_squared_error', cv=LeaveOneOut(), n_jobs=-1)
        loocv = np.sqrt(np.mean(scores))
        print('LOOCV = ', np.round(loocv, 2), 'meV')
        # compute RMS error
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print('RMSE = ', np.round(rmse, 2), 'meV')
        np.savetxt(fname= keci_fname ,X=keci,fmt='%.8f')
        time_stamp = datetime.now().timestamp()
        time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        try:
            print('Try loading ',fit_results_fname,' ...')
            df = pd.read_json(fit_results_fname,orient='index')
            new_data = pd.DataFrame([[time_stamp,time,keci,empty_cluster,weight_copy,alpha,rmse,loocv]],columns=df.columns)
            df2 = df.append(new_data,ignore_index=True)
            print('Updated latest results: ')
            df2.to_json(fit_results_fname,orient='index',indent=4)
            print(df2.iloc[-1])
        except:
            print(fit_results_fname,'is not found, create a new file...')
            print(weight_copy)
            df = pd.DataFrame(data =np.array([[time_stamp,time,keci,empty_cluster,weight_copy,alpha,rmse,loocv]]),
            columns=['time_stamp','time','keci','empty_cluster','weight','alpha','rmse','loocv'])
            df.to_json(fit_results_fname,orient='index',indent=4)
            print('Updated latest results: ')
            print(df.iloc[-1])

        return y_pred, y_true
