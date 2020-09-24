from enum import Enum
import numpy as np

class NodeTyep(Enum):
    Detection = 0
    Tracklet2D = 1

class FeatureTypes(Enum):
    Box2D = "box2D"
    OrigBox2D = "OrigBox2D"
    DetectionScore = "score"
    ReID = "ReID"
    ID = "DetectionIDs"  # it is not really a feature.. but convenient to use existing framework..

class AffinityTypes(Enum):
    DirectedAffinity = "directed"
    UndirectedAffinity = "undirected"
    Unknown = "unknown"

class AffinityOutputType(Enum):
    Dense = 0
    Sparse = 1  # very fast when it filter out a lot of trivial cases (e.g., temporal gap, camera)

class ModelTypes(Enum):
    MedianAppearance = "MedianAppearance"
    MovingAverage3D = "MovingAverage3D"
    Unknown = "unknown"

class NodeAttributeTypes(Enum):
    VideoFileName = "VideoFileName"
    TrackletType = "TrackletType"
    SVID = "SVID"
    DeteectedID = "DetectedID"
    Unknown = "unknown"

MAX_AFFINITY_VAL = 1e10
EPS = 1e-10

class IDContainer(object):
    '''
    Object containing all the IDs over the pipeline steps
    '''
    single_view_id = -1             # assigend after any single_view_pipeline
    labeled_pid = -1                # GT labeled PID
    frame_index = -1                # frame index in original video
