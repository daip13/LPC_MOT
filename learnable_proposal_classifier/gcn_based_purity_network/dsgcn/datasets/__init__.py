from .cluster_dataset import ClusterDataset_pipeline
from .cluster_det_processor import ClusterDetProcessor_pipeline
from .build_dataloader import build_dataloader


__factory__ = {
     'pipeline': ClusterDetProcessor_pipeline
}


def build_dataset(cfg):
    return ClusterDataset(cfg)

def build_dataset_pipeline(cfg):
    return ClusterDataset_pipeline(cfg)


def build_processor(name):
    if name not in __factory__:
        raise KeyError("Unknown processor:", name)
    return __factory__[name]
