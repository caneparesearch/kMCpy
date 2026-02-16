#!/usr/bin/env python
"""Model fitting implementations."""

from abc import ABC, abstractmethod
import json
import logging
import os

from kmcpy.io import convert
from kmcpy.models.parameters import LCEModelParamHistory, LCEModelParameters

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
            cluster_site_indices=[],
            weight=[],
            alpha=0.0,
            time_stamp="",
            time="",
            rmse=0.0,
            loocv=0.0,
        )

    @staticmethod
    def _model_parameters_from_dict(payload: dict) -> LCEModelParameters:
        return LCEModelParameters(
            keci=payload.get("keci", []),
            empty_cluster=payload.get("empty_cluster", 0.0),
            cluster_site_indices=payload.get("cluster_site_indices", []),
            weight=payload.get("weight", []),
            alpha=payload.get("alpha", 0.0),
            time_stamp=payload.get("time_stamp", ""),
            time=payload.get("time", ""),
            rmse=payload.get("rmse", 0.0),
            loocv=payload.get("loocv", 0.0),
        )

    @staticmethod
    def _extract_serialized_record(payload: dict) -> dict:
        """Extract one fitting record from supported serialized payloads."""
        if "keci" in payload:
            return payload

        # Legacy "orient=index" payload used by historical fitting outputs.
        numeric_records = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            try:
                numeric_records.append((int(key), value))
            except (TypeError, ValueError):
                continue
        if numeric_records:
            return max(numeric_records, key=lambda item: item[0])[1]

        raise ValueError("Unsupported serialized fitter schema.")

    @staticmethod
    def _save_fit_results_history(
        model_parameters: LCEModelParameters, fit_results_fname: str
    ) -> None:
        import pandas as pd

        columns = [
            "time_stamp",
            "time",
            "keci",
            "empty_cluster",
            "weight",
            "alpha",
            "rmse",
            "loocv",
        ]
        row = [
            model_parameters.time_stamp,
            model_parameters.time,
            model_parameters.keci,
            model_parameters.empty_cluster,
            model_parameters.weight,
            model_parameters.alpha,
            model_parameters.rmse,
            model_parameters.loocv,
        ]
        try:
            logger.info("Try loading %s ...", fit_results_fname)
            df = pd.read_json(fit_results_fname, orient="index")
            if not set(columns).issubset(df.columns):
                raise ValueError("Unexpected file schema")
            new_data = pd.DataFrame([row], columns=columns)
            df2 = pd.concat([df[columns], new_data], ignore_index=True)
            df2.to_json(fit_results_fname, orient="index", indent=4)
            logger.info("Updated latest results:\n%s", df2.iloc[-1])
        except Exception:
            logger.info(
                "%s is not found or incompatible, create a new file...",
                fit_results_fname,
            )
            df = pd.DataFrame([row], columns=columns)
            df.to_json(fit_results_fname, orient="index", indent=4)
            logger.info("Updated latest results:\n%s", df.iloc[-1])

    def as_dict(self):
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "time_stamp": self.model_parameters.time_stamp,
            "time": self.model_parameters.time,
            "weight": self.model_parameters.weight,
            "alpha": self.model_parameters.alpha,
            "keci": self.model_parameters.keci,
            "empty_cluster": self.model_parameters.empty_cluster,
            "rmse": self.model_parameters.rmse,
            "loocv": self.model_parameters.loocv,
        }
        return d

    def to_json(self, fname):
        logger.info("Saving: %s", fname)
        with open(fname, "w") as fhandle:
            payload = self.as_dict()
            json_str = json.dumps(
                payload, indent=4, default=convert
            )  # to get rid of errors of int64
            fhandle.write(json_str)

    @classmethod
    def from_json(cls, fname):
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            payload = json.load(fhandle)
        if not isinstance(payload, dict):
            raise ValueError("Serialized fitter payload must be a JSON object.")

        record = cls._extract_serialized_record(payload)
        obj = cls()
        obj.model_parameters = cls._model_parameters_from_dict(record)
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
        fit_results_fname=None,
    ) -> tuple[LCEModelParameters, object, object]:
        """Main fitting function

        Args:
            alpha (float): Alpha value for Lasso regression
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000000.
            ekra_fname (str, optional): File name for E_KRA storage. Defaults to 'e_kra.txt'.
            keci_fname (str, optional): File name for KECI storage. Defaults to 'keci.txt'.
            weight_fname (str, optional): File name for weight storage. Defaults to 'weight.txt'.
            corr_fname (str, optional): File name for correlation matrix storage. Defaults to 'correlation_matrix.txt'.
            lce_params_fname (str, optional): File name for LCE parameters storage. If None, skip saving.
            lce_params_history_fname (str, optional): File name for LCE parameters history storage. Defaults to 'lce_params_history.json'.
            fit_results_fname (str | None, optional): Legacy fitting history file
                in orient=index JSON format. If None, skip writing.

        Returns:
            tuple[LCEModelParameters, numpy.ndarray, numpy.ndarray]:
                Fitted model parameters, predicted E_KRA, and DFT-computed E_KRA.
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

        logger.info("Loading E_KRA from %s ...", ekra_fname)
        e_kra = np.loadtxt(ekra_fname)
        weight = np.loadtxt(weight_fname)
        weight_copy = copy(weight)
        correlation_matrix = np.loadtxt(corr_fname)

        estimator = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        estimator.fit(correlation_matrix, e_kra, sample_weight=weight)
        keci = estimator.coef_
        empty_cluster = estimator.intercept_
        logger.info("Lasso Results:")
        logger.info("KECI = \n%s", np.round(keci, 2))
        logger.info(
            "There are %s Non Zero KECI", np.count_nonzero(abs(keci) > 1e-2)
        )
        logger.info("Empty Cluster = %s", empty_cluster)

        y_true = e_kra
        y_pred = estimator.predict(correlation_matrix)
        logger.info("index\tNEB\tLCE\tNEB-LCE")
        index = np.linspace(1, len(y_true), num=len(y_true), dtype="int")
        logger.info(
            "\n%s",
            np.round(np.array([index, y_true, y_pred, y_true - y_pred]).T, decimals=2),
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
        logger.info("LOOCV = %s meV", np.round(loocv, 2))
        # compute RMS error
        rmse = root_mean_squared_error(y_true, y_pred)
        logger.info("RMSE = %s meV", np.round(rmse, 2))
        np.savetxt(fname=keci_fname, X=keci, fmt="%.8f")
        now = datetime.now()
        time_stamp = now.timestamp()
        time = now.strftime("%m/%d/%Y, %H:%M:%S")
        lce_model_params = LCEModelParameters(
            keci=keci.tolist(),
            empty_cluster=empty_cluster,
            cluster_site_indices=[],
            weight=weight_copy.tolist(),
            alpha=alpha,
            time_stamp=time_stamp,
            time=time,
            rmse=rmse,
            loocv=loocv,
        )
        self.model_parameters = lce_model_params

        if fit_results_fname:
            self._save_fit_results_history(lce_model_params, fit_results_fname)

        if lce_params_fname:
            logger.info("Saving LCE model parameters to %s ...", lce_params_fname)
            lce_model_params.to(lce_params_fname)

        if lce_params_history_fname:
            try:
                logger.info(
                    "Saving LCE model parameters history to %s ...",
                    lce_params_history_fname,
                )
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
                logger.error("Error saving LCE model parameters history: %s", e)
                raise e
        return lce_model_params, y_pred, y_true
