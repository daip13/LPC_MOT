import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from core.affinity import *

from algorithms.simple_online_method import OnlineHungarianMethod

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def create_affinities(configs, opt):
    affinities = []
    for config in configs:
        if config['type'] == "SumAffinities":
            config.setdefault("threshold", MAX_AFFINITY_VAL)
            affinity_list = create_affinities(config['list'], opt)
            affinities.append(SumAffinities(affinity_list,
                abs_threshold=config['threshold']))
        elif config['type'] == "NoneCompeteDirectedAffinity":
            base_affinity = create_affinities([ config['base'] ], opt)[0]
            affinities.append(NoneCompeteDirectedAffinity(
                base_affinity=base_affinity,
                abs_threshold=config['threshold'], 
                sim_threshold=config['sim_threshold']))
        elif config['type'] == "IoUAffinity":
            affinities.append(IoUAffinity(
                abs_threshold=config['threshold'],
                weight=config['weight']))
        elif config['type'] == "XDistAffinity":
            affinities.append(XDistAffinity(
                abs_threshold=config['threshold'],
                weight=config['weight']))
        elif config['type'] == "YDistAffinity":
            affinities.append(YDistAffinity(
                abs_threshold=config['threshold'],
                weight=config['weight']))
        elif config['type'] == "SingleViewAppearance":
            config.setdefault("budget", 200)
            config.setdefault("metric_type", "median")
            affinities.append(SingleViewAppearanceAffinity(
                metric_type=config['metric_type'],
                budget=config['budget'],
                abs_threshold=config['threshold'],
                weight=config['weight']))
        elif config['type'] == "NFrameDistance":
            config.setdefault("threshold", 0)
            config.setdefault("weight", 0)
            config.setdefault("allow_overlap", False)
            affinities.append(FastTimeDiffAffinity(
                abs_threshold=config['threshold'],
                weight=config['weight'],
                allow_overlap=config['allow_overlap']))
        else:
            mlog.info("{} is not supported affinity type.".format(config['type']))
            raise NotImplementedError
    return affinities


def create_algorithm(config, opts=None):
    if config['type'] == "OnlineHungarian":
        # default parameters
        config.setdefault("verbose", 1)
        config.setdefault("keep_alive", 10)
        config.setdefault("relative_threshold", 0.0)
        config.setdefault("cv_cap", None)
        config.setdefault("dbg_dir", None)
        config.setdefault("regular_batch_config", "")
        
        batch_opt = None
        if config["regular_batch_config"] != "":
            mlog.info("Loading regular interval batching config at {}".format(config["regular_batch_config"]))
            with open(config["regular_batch_config"], 'r') as f:
                batch_configs = json.loads(f.read())
            batch_affinities = create_affinities(batch_configs["affinity"], {})
            batch_engine = create_algorithm(batch_configs["algorithm"])
            batch_opt = (batch_affinities, batch_engine, batch_configs)

        ret = OnlineHungarianMethod(
                config['verbose'], 
                config['keep_alive'], 
                config['relative_threshold'],
                config['cv_cap'],
                config['dbg_dir'],
                batch_opt)
    else:
        mlog.info("{} is unknown algorithm".format(config['type']))
        raise NotImplementedError
    return ret
