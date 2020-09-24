# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild
from .naic import NAIC
from .reid2019 import REID2019
from .mot17 import MOT17
from .mot20 import MOT20
from .product10k import Product10k

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
