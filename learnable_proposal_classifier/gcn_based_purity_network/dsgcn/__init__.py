#from .test_cluster_det import test_cluster_det
#from .train_cluster_det import train_cluster_det
#from .debug_cluster_det import debug_cluster_det
#from .get_cluster_det import get_cluster_det


__factory__ = { }

def build_handler(phase, stage):
    key_handler = '{}_{}'.format(phase, stage)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
