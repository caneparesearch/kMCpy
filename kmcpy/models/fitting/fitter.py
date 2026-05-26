#!/usr/bin/env python
"""Model fitting implementations."""

from abc import ABC, abstractmethod
import logging
import os

from monty.serialization import dumpfn, loadfn
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
            normalize=True,
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
            normalize=payload.get("normalize", True),
            orbit_fingerprints=payload.get("orbit_fingerprints"),
            local_environment_hash=payload.get("local_environment_hash"),
            # Legacy payloads may contain this provenance field. New fitted
            # parameter files rely on local_environment_hash for validation.
            ordering_convention=payload.get("ordering_convention"),
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

        required_columns = [
            "time_stamp",
            "time",
            "keci",
            "empty_cluster",
            "weight",
            "alpha",
            "rmse",
            "loocv",
        ]
        columns = required_columns + [
            "normalize",
            "orbit_fingerprints",
            "local_environment_hash",
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
            model_parameters.normalize,
            model_parameters.orbit_fingerprints,
            model_parameters.local_environment_hash,
        ]
        try:
            logger.info("Try loading %s ...", fit_results_fname)
            df = pd.read_json(fit_results_fname, orient="index")
            if not set(required_columns).issubset(df.columns):
                raise ValueError("Unexpected file schema")
            if "normalize" not in df.columns:
                df["normalize"] = True
            if "orbit_fingerprints" not in df.columns:
                df["orbit_fingerprints"] = None
            if "local_environment_hash" not in df.columns:
                df["local_environment_hash"] = None
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
            "normalize": self.model_parameters.normalize,
        }
        if self.model_parameters.orbit_fingerprints is not None:
            d["orbit_fingerprints"] = self.model_parameters.orbit_fingerprints
        if self.model_parameters.local_environment_hash is not None:
            d["local_environment_hash"] = self.model_parameters.local_environment_hash
        return d

    def to_json(self, fname):
        logger.info("Saving: %s", fname)
        dumpfn(self.as_dict(), fname, indent=4)

    @classmethod
    def from_json(cls, fname):
        logger.info("Loading: %s", fname)
        payload = loadfn(fname, cls=None)
        if not isinstance(payload, dict):
            raise ValueError("Serialized fitter payload must be a JSON object.")

        record = cls._extract_serialized_record(payload)
        obj = cls()
        obj.model_parameters = cls._model_parameters_from_dict(record)
        return obj

    @staticmethod
    def _fit_lasso_values(
        correlation_matrix,
        e_kra,
        weight,
        alpha,
        max_iter,
        normalize,
    ):
        """Fit Lasso and return unscaled coefficients, intercept, and predictions."""
        from sklearn.linear_model import Lasso
        import numpy as np

        x = np.asarray(correlation_matrix, dtype=float)
        y = np.asarray(e_kra, dtype=float)
        sample_weight = None if weight is None else np.asarray(weight, dtype=float)

        if not normalize:
            estimator = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True)
            fit_kwargs = (
                {} if sample_weight is None else {"sample_weight": sample_weight}
            )
            estimator.fit(x, y, **fit_kwargs)
            keci = estimator.coef_
            empty_cluster = estimator.intercept_
            return keci, empty_cluster, estimator.predict(x)

        if sample_weight is None:
            x_offset = np.mean(x, axis=0)
            y_offset = float(np.mean(y))
        else:
            x_offset = np.average(x, axis=0, weights=sample_weight)
            y_offset = float(np.average(y, weights=sample_weight))
        x_centered = x - x_offset
        y_centered = y - y_offset

        # Mirror the deprecated sklearn normalize=True path used by the
        # historical fitting workflow: center first, then divide each feature
        # by its unweighted L2 norm.
        x_scale = np.sqrt(np.sum(x_centered ** 2, axis=0))
        x_scale[x_scale == 0.0] = 1.0
        x_normalized = x_centered / x_scale

        estimator = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
        fit_kwargs = (
            {} if sample_weight is None else {"sample_weight": sample_weight}
        )
        estimator.fit(x_normalized, y_centered, **fit_kwargs)
        keci = estimator.coef_ / x_scale
        empty_cluster = y_offset - float(np.dot(x_offset, keci))
        y_pred = x @ keci + empty_cluster
        return keci, empty_cluster, y_pred

    @classmethod
    def _leave_one_out_rmse(
        cls,
        correlation_matrix,
        e_kra,
        alpha,
        max_iter,
        normalize,
    ) -> float:
        """Compute the historical unweighted leave-one-out RMSE."""
        import numpy as np

        x = np.asarray(correlation_matrix, dtype=float)
        y = np.asarray(e_kra, dtype=float)
        squared_errors = []
        for excluded_index in range(len(y)):
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[excluded_index] = False
            keci, empty_cluster, _ = cls._fit_lasso_values(
                x[train_mask],
                y[train_mask],
                weight=None,
                alpha=alpha,
                max_iter=max_iter,
                normalize=normalize,
            )
            y_pred = float(x[excluded_index] @ keci + empty_cluster)
            squared_errors.append((float(y[excluded_index]) - y_pred) ** 2)
        return float(np.sqrt(np.mean(squared_errors)))

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
        normalize=True,
        orbit_fingerprints=None,
        local_environment_hash=None,
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
            orbit_fingerprints (list[str] | None, optional): Orbit fingerprints
                associated with the fitted coefficient order.
            local_environment_hash (str | None, optional): Hash of the ordered
                local environment used to validate fitted parameters later.

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
        from sklearn.metrics import root_mean_squared_error

        from copy import copy
        from datetime import datetime
        import numpy as np

        logger.info("Loading E_KRA from %s ...", ekra_fname)
        e_kra = np.loadtxt(ekra_fname)
        weight = np.loadtxt(weight_fname)
        weight_copy = copy(weight)
        correlation_matrix = np.loadtxt(corr_fname)

        keci, empty_cluster, y_pred = self._fit_lasso_values(
            correlation_matrix=correlation_matrix,
            e_kra=e_kra,
            weight=weight,
            alpha=alpha,
            max_iter=max_iter,
            normalize=normalize,
        )
        if orbit_fingerprints is not None and len(keci) != len(orbit_fingerprints):
            raise ValueError(
                "orbit_fingerprints length does not match fitted keci length: "
                f"{len(orbit_fingerprints)} != {len(keci)}"
            )
        logger.info("Lasso Results:")
        logger.info("KECI = \n%s", np.round(keci, 2))
        logger.info(
            "There are %s Non Zero KECI", np.count_nonzero(abs(keci) > 1e-2)
        )
        logger.info("Empty Cluster = %s", empty_cluster)

        y_true = e_kra
        logger.info("index\tNEB\tLCE\tNEB-LCE")
        index = np.linspace(1, len(y_true), num=len(y_true), dtype="int")
        logger.info(
            "\n%s",
            np.round(np.array([index, y_true, y_pred, y_true - y_pred]).T, decimals=2),
        )

        # cv = sqrt(mean(scores)) + N_nonzero_eci*penalty, penalty = 0 here
        loocv = self._leave_one_out_rmse(
            correlation_matrix=correlation_matrix,
            e_kra=e_kra,
            alpha=alpha,
            max_iter=max_iter,
            normalize=normalize,
        )
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
            normalize=normalize,
            orbit_fingerprints=orbit_fingerprints,
            local_environment_hash=local_environment_hash,
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
