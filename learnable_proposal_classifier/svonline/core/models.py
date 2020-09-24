import numpy as np
from data_types import *
from abc import ABCMeta, abstractmethod
from utils import *
import sys
import os
from bisect import bisect_left

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import copy
import scipy

class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._type = ModelTypes.Unknown
    
    @property
    def type(self):
        return self._type

    @abstractmethod
    def get(self, timestamp):
        '''
        Provide model output for the timestamp
        '''
        raise NotImplementedError

    @abstractmethod
    def update(self, timestamps, features, refresh):
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError
        
class MedianAppearanceModel(BaseModel):
    def __init__(self, budget = 100, sample_gap = 2):
        self._type = ModelTypes.MedianAppearance
        self._budget = budget
        self._sample_gap = sample_gap
        self._median = { 'front': None,
                         'rear': None }

    def clone(self):
        self_copy = MedianAppearanceModel(self._budget, self._sample_gap)
        self_copy._median['front'] = self._median['front']
        self_copy._median['rear'] = self._median['rear']
        return self_copy

    def clear(self):
        self._median = { 'front': None,
                         'rear': None }

    def update(self, timestamps, features, refresh):
        # This is slow!
        if FeatureTypes.ReID in features:
            feats = [ f for f in features[FeatureTypes.ReID] if f is not None ]
            if len(feats) != 0:
                if len(feats) > self._budget:
                    feats = feats[:self._budget:self._sample_gap]
                self._median['front'] = np.median(np.array(feats), axis=0)
            feats = [ f for f in features[FeatureTypes.ReID] if f is not None ]
            if len(feats) != 0:
                if len(feats) > self._budget:
                    feats = feats[-self._budget::self._sample_gap]
                self._median['rear'] = np.median(np.array(feats), axis=0)

    def get(self, position):
        if (position in self._median) and self._median[position] is not None:
            return self._median[position].flatten()
        else:
            print("ERROR: the position:" + position + " is None!")
            return None

