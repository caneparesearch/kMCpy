#!/usr/bin/env python
"""
This module provides the Fitting class for generating and storing fitting results.
"""
from abc import ABC, abstractmethod
import json
from kmcpy.io import convert
import logging
from kmcpy.model.model_parameters import LCEModelParameters, LCEModelParamHistory
import os

logger = logging.getLogger(__name__) 

class BaseFitter(ABC):
    """Main class for model fitting"""

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def as_dict(self):
        """Convert the fitting object to a dictionary"""
        raise NotImplementedError(
            "This method should be implemented in the subclass to convert the fitting object to a dictionary."
        )
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the model to the data"""
        raise NotImplementedError(
            "This method should be implemented in the subclass to fit the model to the data."
        )
    
class LCEFitter(BaseFitter):
    """Fitting class for Local Cluster Expansion (LCE) model"""

    def __init__(self) -> None:
        """
        Initialize the LCEFitter object.
        """
        super().__init__()
        self.model_parameters = LCEModelParameters(
            keci=[],
            empty_cluster=0.0,
            sublattice_indices=[],
            weight=[],
            alpha=0.0,
            time_stamp="",
            time="",
            rmse=0.0,
            loocv=0.0)

    def as_dict(self):
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "time_stamp": self.model_parameters.time_stamp,
            "weight": self.model_parameters.weight,
            "alpha": self.model_parameters.alpha,
            "keci": self.model_parameters.keci,
            "empty_cluster": self.model_parameters.empty_cluster,
            "rmse": self.model_parameters.rmse,
            "loocv": self.model_parameters.loocv,
        }
        return d

    def to_json(self, fname):
        logger.info(f"Saving: {fname}")
        with open(fname, "w") as fhandle:
            d = self.as_dict()
            jsonStr = json.dumps(
                d, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(jsonStr)

    @classmethod
    def from_json(self, fname):
        logger = logging.getLogger(__name__)
        logger.info(f"Loading: {fname}")
        with open(fname, "rb") as fhandle:
            objDict = json.load(fhandle)
        obj = LCEFitter()
        obj.__dict__ = objDict
        return obj

    def fit(
        self,
        alpha,
        max_iter=1000000,
        ekra_fname="e_kra.txt",
        keci_fname="keci.txt",
        weight_fname="weight.txt",
        corr_fname="correlation_matrix.txt",
        lce_params_fname="lce_params.json",
        lce_params_history_fname="lce_params_history.json",
    ) -> 'LCEModelParameters':
        """Main fitting function

        Args:
            alpha (float): Alpha value for Lasso regression
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000000.
            ekra_fname (str, optional): File name for E_KRA storage. Defaults to 'e_kra.txt'.
            keci_fname (str, optional): File name for KECI storage. Defaults to 'keci.txt'.
            weight_fname (str, optional): File name for weight storage. Defaults to 'weight.txt'.
            corr_fname (str, optional): File name for correlation matrix storage. Defaults to 'correlation_matrix.txt'.
            lce_params_fname (str, optional): File name for LCE parameters storage. Defaults to 'lce_params.json'.
            lce_params_history_fname (str, optional): File name for LCE parameters history storage. Defaults to 'lce_params_history.json'.

        Returns:
            y_pred (numpy.ndarray(float)),y_true (numpy.ndarray(float)): Predicted E_KRA; DFT Computed E_KRA
        """

        """
        E_KRA[m x 1] =  diagonal(Weight)[m x m] * Corr[m x n] * ECI[n x 1] + V_0[m x 1]
        E_KRA is a m x 1 vector
        ECI is a n x 1 vector
        Corr is a m x n matrix
        V_0 is a m x 1 vector
        Weight is a n x n diagonal matrix
        m is the number of E_KRA
        n is the number of clusers
        """
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import root_mean_squared_error

        from copy import copy
        from datetime import datetime
        import numpy as np

        logger.info(f"Loading E_KRA from {ekra_fname} ...")
        e_kra = np.loadtxt(ekra_fname)
        weight = np.loadtxt(weight_fname)
        weight_copy = copy(weight)
        correlation_matrix = np.loadtxt(corr_fname)

        estimator = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        estimator.fit(correlation_matrix, e_kra, sample_weight=weight)
        keci = estimator.coef_
        empty_cluster = estimator.intercept_
        logger.info("Lasso Results:")
        logger.info(f"KECI = \n{np.round(keci, 2)}")
        logger.info(f"There are {np.count_nonzero(abs(keci) > 1e-2)} Non Zero KECI")
        logger.info(f"Empty Cluster = {empty_cluster}")

        y_true = e_kra
        y_pred = estimator.predict(correlation_matrix)
        logger.info("index\tNEB\tLCE\tNEB-LCE")
        index = np.linspace(1, len(y_true), num=len(y_true), dtype="int")
        logger.info(
            f"\n{np.round(np.array([index, y_true, y_pred, y_true - y_pred]).T, decimals=2)}"
        )

        # cv = sqrt(mean(scores)) + N_nonzero_eci*penalty, penalty = 0 here
        scores = -1 * cross_val_score(
            estimator=estimator,
            X=correlation_matrix,
            y=e_kra,
            scoring="neg_mean_squared_error",
            cv=LeaveOneOut(),
            n_jobs=-1,
        )
        loocv = np.sqrt(np.mean(scores))
        logger.info(f"LOOCV = {np.round(loocv, 2)} meV")
        # compute RMS error
        rmse = root_mean_squared_error(y_true, y_pred)
        logger.info(f"RMSE = {np.round(rmse, 2)} meV")
        np.savetxt(fname=keci_fname, X=keci, fmt="%.8f")
        time_stamp = datetime.now().timestamp()
        time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        lce_model_params = LCEModelParameters(
            keci=keci.tolist(),
            empty_cluster=empty_cluster,
            sublattice_indices=[],
            weight=weight_copy.tolist(),
            alpha=alpha,
            time_stamp=time_stamp,
            time=time,
            rmse=rmse,
            loocv=loocv,
        )
        logger.info(f"Saving LCE model parameters to {lce_params_fname} ...")
        lce_model_params.to(lce_params_fname)

        if lce_params_history_fname:
            try:
                logger.info(f"Saving LCE model parameters history to {lce_params_history_fname} ...")
                if os.path.exists(lce_params_history_fname):
                    logger.info("Loading LCE model parameters history ...")
                    lce_model_param_history = LCEModelParamHistory.from_file(
                        lce_params_history_fname
                    )
                    lce_model_param_history.append(lce_model_params)
                    lce_model_param_history.to(lce_params_history_fname)
                else:
                    logger.info("Creating new LCE model parameters history ...")
                    lce_model_param_history = LCEModelParamHistory()
                    lce_model_param_history.append(lce_model_params)
                    lce_model_param_history.to(lce_params_history_fname)
            except Exception as e:
                logger.error(f"Error saving LCE model parameters history: {e}")
                raise e
        return lce_model_params, y_pred, y_true
