#!/usr/bin/env python
"""
Legacy fitting compatibility layer.

This module keeps the historical ``kmcpy.fitting.Fitting`` API while delegating
the core fitting implementation to ``kmcpy.models.fitting.fitter.LCEFitter``.
"""

import json
import logging

import pandas as pd

from kmcpy.io.io import convert
from kmcpy.models.fitting.fitter import LCEFitter

logger = logging.getLogger(__name__)


class Fitting:
    """Backward-compatible wrapper for the legacy fitting API."""

    def __init__(self) -> None:
        self._fitter = LCEFitter()

    def add_data(
        self, time_stamp, time, keci, empty_cluster, weight, alpha, rmse, loocv
    ) -> None:
        self.time_stamp = time_stamp
        self.time = time
        self.weight = weight
        self.alpha = alpha
        self.keci = keci
        self.empty_cluster = empty_cluster
        self.rmse = rmse
        self.loocv = loocv

    def as_dict(self):
        return {
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

    def to_json(self, fname):
        logger.info("Saving: %s", fname)
        with open(fname, "w") as fhandle:
            json_str = json.dumps(self.as_dict(), indent=4, default=convert)
            fhandle.write(json_str)

    @classmethod
    def from_json(cls, fname):
        logger.info("Loading: %s", fname)
        with open(fname, "rb") as fhandle:
            obj_dict = json.load(fhandle)
        obj = cls()
        obj.__dict__.update(obj_dict)
        return obj

    def _save_legacy_fit_results(self, fit_results_fname: str) -> None:
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
            self.time_stamp,
            self.time,
            self.keci,
            self.empty_cluster,
            self.weight,
            self.alpha,
            self.rmse,
            self.loocv,
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
            logger.info("%s is not found or incompatible, create a new file...", fit_results_fname)
            df = pd.DataFrame([row], columns=columns)
            df.to_json(fit_results_fname, orient="index", indent=4)
            logger.info("Updated latest results:\n%s", df.iloc[-1])

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
        """Backward-compatible fitting entrypoint."""
        lce_model_params, y_pred, y_true = self._fitter.fit(
            alpha=alpha,
            max_iter=max_iter,
            ekra_fname=ekra_fname,
            keci_fname=keci_fname,
            weight_fname=weight_fname,
            corr_fname=corr_fname,
            lce_params_fname=None,
            lce_params_history_fname=None,
        )

        self.add_data(
            time_stamp=lce_model_params.time_stamp,
            time=lce_model_params.time,
            keci=lce_model_params.keci,
            empty_cluster=lce_model_params.empty_cluster,
            weight=lce_model_params.weight,
            alpha=lce_model_params.alpha,
            rmse=lce_model_params.rmse,
            loocv=lce_model_params.loocv,
        )
        self._save_legacy_fit_results(fit_results_fname)
        return y_pred, y_true
