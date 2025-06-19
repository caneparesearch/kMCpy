"""
Fitting is a class to generate and store the fitted results. This is the class for generating fitting_results.json

Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""

import json
from kmcpy.io import convert
import logging

logger = logging.getLogger(__name__) 

class Fitting:
    """Main class for model fitting"""

    def __init__(self) -> None:
        pass

    def add_data(
        self, time_stamp, time, keci, empty_cluster, weight, alpha, rmse, loocv
    ) -> None:
        """
        Add data to the Fitting object

        Args:
            time_stamp (float): Time stamp string of the fitting
            time (string): Human redable date time of the fitting
            weight ([float]): Weights of each NEB data point
            alpha (float): Alpha value for Lasso regression
            keci ([float]): Kinetic effective cluster interactions
            empty_cluster (float): Empty cluster
            rmse (float): Root mean square error
            loocv (float): Leave-one-out cross validation error
        """
        self.time_stamp = time_stamp
        self.time = time
        self.weight = weight
        self.alpha = alpha
        self.keci = keci
        self.empty_cluster = empty_cluster
        self.rmse = rmse
        self.loocv = loocv

    def as_dict(self):
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "time_stamp": self.time_stamp,
            "weight": self.weight,
            "alpha": self.alpha,
            "keci": self.keci,
            "empty_cluster": self.empty_cluster,
            "rmse": self.rmse,
            "loocv": self.loocv,
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
        obj = Fitting()
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
        fit_results_fname="fitting_results.json",
    ) -> tuple:
        """Main fitting function

        Args:
            alpha (float): Alpha value for Lasso regression
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000000.
            ekra_fname (str, optional): File name for E_KRA storage. Defaults to 'e_kra.txt'.
            keci_fname (str, optional): File name for KECI storage. Defaults to 'keci.txt'.
            weight_fname (str, optional): File name for weight storage. Defaults to 'weight.txt'.
            corr_fname (str, optional): File name for correlation matrix storage. Defaults to 'correlation_matrix.txt'.
            fit_results_fname (str, optional): File name for fitting results storage. Defaults to 'fitting_results.json'.

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
        import pandas as pd

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

        try:
            logger.info(f"Try loading {fit_results_fname} ...")
            df = pd.read_json(fit_results_fname, orient="index")
            new_data = pd.DataFrame(
                [
                    [
                        time_stamp,
                        time,
                        keci,
                        empty_cluster,
                        weight_copy,
                        alpha,
                        rmse,
                        loocv,
                    ]
                ],
                columns=df.columns,
            )
            df2 = pd.concat([df, new_data])
            df2.to_json(fit_results_fname, orient="index", indent=4)
            logger.info("Updated latest results: ")
            logger.info(f"\n{df2.iloc[-1]}")
        except Exception as e:
            logger.info(f"{fit_results_fname} is not found, create a new file...")
            logger.info(weight_copy)
            df = pd.DataFrame(
                data=[
                    [
                        time_stamp,
                        time,
                        keci,
                        empty_cluster,
                        weight_copy,
                        alpha,
                        rmse,
                        loocv,
                    ]
                ],
                columns=[
                    "time_stamp",
                    "time",
                    "keci",
                    "empty_cluster",
                    "weight",
                    "alpha",
                    "rmse",
                    "loocv",
                ],
            )
            df.to_json(fit_results_fname, orient="index", indent=4)
            logger.info("Updated latest results: ")
            logger.info(f"\n{df.iloc[-1]}")

        return y_pred, y_true
