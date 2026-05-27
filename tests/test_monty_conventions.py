from monty.json import MSONable
from monty.serialization import dumpfn, loadfn

from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.local_barrier_model import LocalBarrierModel
from kmcpy.models.parameters import LCEModelParamHistory, LCEModelParameters
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.state import State
from kmcpy.structure.basis import ChebyshevBasis
from kmcpy.structure.local_site_order import LocalSiteOrder


def test_core_serializable_objects_are_msonable(tmp_path):
    parameters = LCEModelParameters(
        keci=[1.0],
        empty_cluster=0.0,
        cluster_site_indices=[],
        weight=[],
        alpha=0.0,
        time_stamp=0.0,
        time="",
        rmse=0.0,
        loocv=0.0,
    )
    objects = [
        Configuration(temperature=300.0, kmc_passes=10),
        State([0, 1], time=1.2, step=3),
        LocalBarrierModel.constant_barrier(250.0),
        parameters,
        LCEModelParamHistory([parameters]),
        LCEFitter(),
        ChebyshevBasis(max_states=3),
        LocalSiteOrder.from_name("kmcpy_default"),
    ]

    for index, obj in enumerate(objects):
        assert isinstance(obj, MSONable)
        payload = obj.as_dict()
        assert payload["@module"] == obj.__class__.__module__
        assert payload["@class"] == obj.__class__.__name__

        filename = tmp_path / f"{index}_{obj.__class__.__name__}.json"
        dumpfn(obj, filename)
        loaded = loadfn(filename)
        assert isinstance(loaded, obj.__class__)
