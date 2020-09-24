import sys
import os
sys.path.append(os.path.dirname(__file__))
from models import *
import copy

class ViewData(object):
    '''
    Store all the features specific to one camera view or 3D states
    the heavy fields are: __timestamps, __features, __camera_parameters, and __models
    '''
    def __init__(
            self, name, state_estimation_engine=None, timestamp=None, features=None, models=None, camera_params=None, is_head=0):
        self.__name = name 
        self.__timestamps = np.array([], dtype=np.int)
        self.__features = {}
        self.__camera_params = camera_params
        self.__is_head = is_head

        assert state_estimation_engine is None, \
                "Do you ever use this? If so, why?"

        self.__2d_state_estimation_engine = state_estimation_engine
        # unlike features, model may not have one-to-one correspondence
        # to the timestamps. e.g., kalman filter model, median appearance model, etc.
        self.__models = {}
        if models is not None:
            self.__models = models

        if timestamp is not None:
            self.__features = { key : [] for key in features.keys() }
            self.append(timestamp, features)
            self.update_models(True)

    def resample(self, every, update_models = False):
        if self.size == 0:
            return
        idx = [0, len(self.__timestamps) - 1] + \
                [ i for i in range(len(self.__timestamps)) if self.__timestamps[i] % every == 0 ]
        idx = sorted(list(set(idx)))
        self.__timestamps = np.array([ self.__timestamps[i] for i in idx ])
        for key in self.__features:
            self.__features[key] = [ self.__features[key][i] for i in idx ]
        if update_models:
            self.update_models(True)
    
    def delete_feature(self, feat_type):
        self.__features.pop(feat_type, None)
    
    def split(self, split_timestamps):
        ret = []

        timestamps = []
        features = { key: [] for key in self.__features.keys() }

        split_timestamps = sorted(list(split_timestamps))
        last_idx = 0
        for i in range(self.size):
            if self.__timestamps[i] >= split_timestamps[last_idx]:
                # new one
                ret.append(ViewData(self.__name, self.__2d_state_estimation_engine,
                    timestamps, features,
                    { key: model.clone() for key, model in self.__models.iteritems()},
                    self.__camera_params))
                timestamps = []
                features = { key: [] for key in self.__features.keys() }
                last_idx += 1
            timestamps.append(self.__timestamps[i])
            for key in features:
                features[key].append(self.__features[key][i])

        if timestamps:
            ret.append(ViewData(self.__name, self.__2d_state_estimation_engine,
                timestamps, features,
                { key: model.clone() for key, model in self.__models.iteritems()},
                self.__camera_params))
            timestamps = []
            features = { key: [] for key in self.__features.keys() }

        return ret

    def clone(self):
        models = {}
        for key in self.__models:
            models[key] = self.__models[key].clone()

        self_copy = ViewData(
            self.__name, self.__2d_state_estimation_engine)
        self_copy.__features = copy.deepcopy(self.__features)
        self_copy.__models = models
        self_copy.__timestamps = self.__timestamps[:]
        self_copy.__is_head = self.__is_head
        self_copy.__camera_params = self.__camera_params
        return self_copy
    
    def clear(self):
        self.__timestamps = np.array([], dtype=np.int)
        self.__features = { key : [] for key in self.__features.keys() }
        for key in self.__models:
            self.__models[key].clear()

    def update_models(self, refresh):
        for key in self.__models:
            self.__models[key].update(
                    self.__timestamps,
                    self.__features,
                    refresh)

    def set(self, timestamps, features, update_model=True):
        self.__timestamps = timestamps
        self.__features = features
        if update_model:
            self.update_models(True)

    def model(self, model_type):
        if model_type in self.__models:
            return self.__models[model_type]
        else:
            return None
    
    def set_model(self, model_type, model):
        self.__models[model_type] = model
        self.__models[model_type].update(
                self.__timestamps,
                self.__features,
                True)

    def set_models(self, models):
        self.__models = models
        self.update_models(True)

    def synchronize_timestamps(self, tsf_data, doupdate = True, discretize_ms = 40):
        # update all the timestamps to be globally consistent!
        synced_timestamps = [ np.round(float(tsf_data[t]) / discretize_ms).astype(int) 
                                for t in self.__timestamps if t in tsf_data ]
        self.__timestamps = synced_timestamps
        # we may drop some frames due to tsf_data missing
        for key in self.__features:
            if len(self.__features[key]) > self.size:
                self.__features[key] = self.__features[key][:self.size]
        if doupdate:
            self.update_models(True)

    @property
    def name(self):
        return self.__name

    @property
    def camera_params(self):
        return self.__camera_params

    @property
    def is_head(self):
        return self.__is_head
    
    @property
    def size(self):
        return len(self.__timestamps)

    @property
    def timestamps(self):
        return self.__timestamps

    @property
    def features(self):
        return self.__features

    @property
    def start(self):
        if self.size == 0:
            return None
        return self.__timestamps[0]

    @property
    def end(self):
        if self.size == 0:
            return None
        return self.__timestamps[-1]
    
    @property
    def feature_types(self):
        return self.__features.keys()
    
    @property
    def ids(self):
        assert FeatureTypes.ID in self.__features
        return self.__features[FeatureTypes.ID]

    @property
    def models(self):
        return self.__models

    def feature(self, key):
        if key not in self.__features:
            return None
        return self.__features[key]

    def __getitem__(self, timestamp):
        idx = np.where(self.__timestamps == timestamp)[0]
        if idx.size == 0:
            return None
        idx = idx[0]
        return { key : self.__features[key][idx] \
                    for key in self.__features.keys() }

    # add initial feature
    def add_feature(self, key, init_value, update_models=True):
        # add a new feature type
        assert key not in self.__features, \
                "{} already exists in the feature set". format(key)
        if self.size == 1 and not isinstance(init_value, list):
            self.__features[key] = [ init_value ]
        else:
            assert len(init_value) == self.size
            self.__features[key] = init_value
        self.update_models(True)

    # set one feature type
    def set_feature(self, key, values, update_models=False):
        assert key not in self.__features, \
                "{} already exists in the feature set". format(key)
        self.__features[key] = values
        self.update_models(update_models)

    def append(self, timestamp, feats):
        # need to make sure they are all sorted
        if self.size > 0:
            assert timestamp > self.end,\
                    "data should be appended in ordered way"

        self.__timestamps = np.append(self.__timestamps, timestamp).astype(np.int)
        assert len(self.__features.keys()) == 0 or \
                len(feats.keys()) == len(self.__features.keys()), \
                "The number of feature types should be consistent"

        for key in feats.keys():
            if key not in self.__features:
                self.__features[key] = []
            self.__features[key].append(feats[key])

        self.update_models(False)
    
    def extend(self, other, merge_overlap = False, update_models = True):
        '''
        extend self with the other. should be faster than merge
        '''
        ts = [ t for i, t in enumerate(other.timestamps) if t > self.end ]
        for key in self.feature_types:
            self.__features[key] += [ other.feature(key)[i] \
                for i, t in enumerate(other.timestamps) if t > self.end ]
        self.__timestamps = np.append(self.__timestamps, ts).astype(np.int)
        
        if update_models:
            self.update_models(False)

    def merge_with_dict(self, other, merge_overlap = False, update_models = True):
        """
        Could be a more efficient alternative to merge?
        """
        assert self.__name == other.__name, "cannot merge two data from other views"
        assert len(self.feature_types) == len(other.feature_types)

        import time
        tic = time.time()

        buf = {}
        for i, t in enumerate(other.timestamps):
            buf[t] = { key: other.feature(key)[i] for key in other.feature_types }
        for i, t in enumerate(self.timestamps):
            buf[t] = { key: self.feature(key)[i] for key in self.feature_types }
        
        timestamps = sorted(buf.keys())
        features = { key : [ None for _ in timestamps ] for key in self.feature_types}
        for key in self.feature_types:
            for i, t in enumerate(timestamps):
                features[key][i] = buf[t][key]

        print("merge:", time.time() - tic)
        tic = time.time()
        self.__timestamps = np.array(timestamps)
        self.__features = features
        if update_models:
            self.update_models(True)
        print("update:", time.time() - tic)
        

    # simple to skip the overlapping area
    def merge_with_overlap(self, other, update_models = True):
        timestamps = []
        feature_types = self.feature_types
        features = {key: [] for key in feature_types}

        t1 = 0
        t2 = 0
        while t1 < self.size and t2 < other.size:
            if self.timestamps[t1] <= other.timestamps[t2]:
                if not timestamps and self.timestamps[t1] > timestamps[-1]:  # only if there is no overlap
                        timestamps.append(self.timestamps[t1])
                        for key in feature_types:
                            features[key].append(self.feature(key)[t1])
                t1 += 1
            else:
                if not timestamps and other.timestamps[t2] > timestamps[-1]:  # only if there is no overlap
                    timestamps.append(other.timestamps[t2])
                    for key in feature_types:
                        features[key].append(other.feature(key)[t2])
                t2 += 1

        # append remaining data
        while t1 < self.size:
            if not timestamps and self.timestamps[t1] > timestamps[-1]:  # only if there is no overlap
                timestamps.append(self.timestamps[t1])
                for key in feature_types:
                    features[key].append(self.feature(key)[t1])
            t1 += 1
        while t2 < other.size:
            if not timestamps and other.timestamps[t2] > timestamps[-1]:  # only if there is no overlap
                timestamps.append(other.timestamps[t2])
                for key in feature_types:
                    features[key].append(other.feature(key)[t2])
            t2 += 1

        self.__timestamps = np.array(timestamps)
        self.__features = features
        if update_models:
            self.update_models(True)


    def merge(self, other, merge_overlap = False, update_models = True):
        '''
        Merge with the other Camera Data
        '''
        # return self.merge2(other, merge_overlap, update_models)

        assert self.__name == other.__name, "cannot merge two data from other views"

        # implement merging two sorted data
        timestamps = []
        feature_types = self.feature_types
        features = { key : [] for key in feature_types}

        if merge_overlap:
            self.merge_with_overlap(other, update_models)
            return

        t1 = 0
        t2 = 0
        while t1 < self.size and t2 < other.size:
            if self.timestamps[t1] <= other.timestamps[t2]: 
                if not timestamps or self.timestamps[t1] > timestamps[-1]: # only if there is no overlap
                    timestamps.append(self.timestamps[t1])
                    for key in feature_types:
                        features[key].append(self.feature(key)[t1])
                t1 += 1
            else:
                if not timestamps or other.timestamps[t2] > timestamps[-1]: # only if there is no overlap
                    timestamps.append(other.timestamps[t2])
                    for key in feature_types:
                        features[key].append(other.feature(key)[t2])
                t2 += 1

        # append remaining data
        while t1 < self.size:
            if not timestamps or self.timestamps[t1] > timestamps[-1]: # only if there is no overlap
                timestamps.append(self.timestamps[t1])
                for key in feature_types:
                    features[key].append(self.feature(key)[t1])
            t1 += 1
        while t2 < other.size:
            if not timestamps or other.timestamps[t2] > timestamps[-1]: # only if there is no overlap
                timestamps.append(other.timestamps[t2])
                for key in feature_types:
                    features[key].append(other.feature(key)[t2])
            t2 += 1
        
        self.__timestamps = np.array(timestamps)
        self.__features = features
        if update_models:
            self.update_models(True)
    
class Node(object):
    '''
    Define the structure of one node.
    A node could represent either a detection, a tracklet, or a trajectory.
    '''
    def __init__(self, tid, state_estimation_engine = None):
        self._tid = tid # track-id
        self._pid = -1  # globally unique ID, which is usually obtained by face or staff recognition
        self._label_result = None
        self._mv_ignore = 0
        self._pid_suggestions = []
        self._camera_data = {} # a track could be defined on mult-cameras
        self._state_estimation_engine = state_estimation_engine
        if state_estimation_engine is not None:
            self._states = ViewData("3D", models={
                ModelTypes.MovingAverage3D: MovingAverage3D()
                })
        else:
            self._states = None

    # calibration not needed for SV tracking
    def init(self, camera_name, timestamp, features, camera_params=None, models=None, is_head=0):
        if models is None:
            models = {
                ModelTypes.MedianAppearance: MedianAppearanceModel(),
                }
        self._attribute = {}

        self._camera_data[camera_name] = ViewData(camera_name, timestamp=timestamp, features=features,
                models=models, camera_params=camera_params, is_head=is_head)

    def clone(self):
        # TODO implement a clone method
        return self
    
    def clear(self):
        for key in self._camera_data:
            self._camera_data[key].clear()
        if self._states is not None:
            self._states.clear()
    
    def attribute(self, key):
        if key in self._attribute:
            return self._attribute[key]
        return None

    def set_attribute(self, key, data):
        self._attribute[key] = data

    @property
    def size(self):
        count = 0
        for key in self._camera_data:
            count += len(self._camera_data[key].timestamps)
        return count

    @property
    def isempty(self):
        return len([ key for key in self._camera_data \
                if self._camera_data[key].timestamps.size > 0 ]) == 0

    @property
    def isignore(self):
        return self._mv_ignore

    def ignore(self):
        self._mv_ignore = 1

    @property
    def tid(self):
        return int(self._tid)
    
    def changetid(self, new_tid):
        self._tid = new_tid
    
    @property
    def mv_start(self):
        '''
        :return: the earliest time that any camera starts working
        '''
        ret = None
        for key in self._camera_data.keys():
            camera_start = self._camera_data[key].start
            if ret is None:
                ret = camera_start
            elif camera_start < ret:
                ret = camera_start
        return ret
    
    @property
    def mv_end(self):
        '''
        :return: the latest time that any camera ends
        '''
        ret = None
        for key in self._camera_data.keys():
            camera_end = self._camera_data[key].end
            if ret is None:
                ret = camera_end
            elif camera_end > ret:
                ret = camera_end
        return ret

    def start(self, camera_view):
        if camera_view is None:
            return self.mv_start
        else:
            return self[camera_view].start
    
    def end(self, camera_view):
        if camera_view is None:
            return self.mv_end
        else:
            return self[camera_view].end
    
    @property
    def cameras(self):
        return self._camera_data.keys()

    def __getitem__(self, key):
        if key == "state":
            return self._states
        elif key in self._camera_data:
            return self._camera_data[key]
        return None
    
    def update_states(self):
        # update the 3D estimation
        if self._state_estimation_engine is not None:
            self._state_estimation_engine(self._camera_data, self._states)

    def merge_states(self, other):
        if self._state_estimation_engine is not None\
                and other._state_estimation_engine is not None:
            self._states.merge(other._states, update_models=True)

    def extend_states(self, other):
        if self._state_estimation_engine is not None:
            self._states.extend(other._states, update_models=True)

    def merge(self, other, update_state = True):
        # TODO: actually it will be useful to keep the original information
        #   by using tree structure instead of merging all..
        for key in other._camera_data.keys():
            if key not in self._camera_data:
                self._camera_data[key] = other._camera_data[key].clone()
            else:
                self._camera_data[key].merge(
                        other._camera_data[key],
                        update_models=update_state)

        # merge attributes
        self.merge_label_result(other)

    def extend(self, other, update_state = True):
        for key in other._camera_data.keys():
            if key not in self._camera_data:
                self._camera_data[key] = other._camera_data[key]
            else:
                self._camera_data[key].extend(
                        other._camera_data[key],
                        update_models=update_state)
        # update the 3D estimation
        if update_state:
            self.extend_states(other)

    def update_models(self):
        # update all the models across views
        for key in self._camera_data:
            self._camera_data[key].update_models(True)

    def update_detection_info(self, id_type="single_view"):
        # get the pid confidence
        assigned_pid_confidence = 0.0
        pid_classification = self.attribute(NodeAttributeTypes.PIDClass)
        if pid_classification is not None:
            p = pid_classification.probability(self.pid)
            if p >= 0:
                assigned_pid_confidence = p

        # update all ID related information per detection
        for name in self.cameras:
            id_container = self._camera_data[name].ids
            if id_container is None:
                continue
            # loop over all the individual detections
            # to assign the id information
            for i in range(self._camera_data[name].size):
                if id_type == "single_view":
                    id_container[i].single_view_id = self.tid
                elif id_type == "multi_view":
                    id_container[i].multi_view_id = self.tid
                elif id_type != "skip":
                    raise NotImplementedError
                # assign PID information
                if self.pid > 0:
                    id_container[i].pid_classification_id = self.pid
                    id_container[i].pid_classification_confidence = \
                            assigned_pid_confidence
                else:
                    id_container[i].pid_classification_id = 0
                    id_container[i].pid_classification_confidence = 0.0

def nodes_in_time(nodes, start, end, camera_view = None):
    # seems like this is the major bottleneck!
    if camera_view is None:
        idx = [ i for i, node in enumerate(nodes) \
                if node.end >= start and node.start <= end ] 
    else:
        idx = [ i for i, node in enumerate(nodes) \
                    if camera_view in node.cameras and \
                    node.end(camera_view) >= start and \
                    node.start(camera_view) <= end ]
    return idx, [nodes[i] for i in idx]

def split_nodes(nodes, interval, camera_view = None):
    assert interval == 1, "Currently supporting one frame interval only."
    ret = []
    for i, node in enumerate(nodes):
        # based on the start frame
        node_frame = node.start(camera_view)
        assert node_frame == node.end(camera_view), \
                "multiple frame nodes are not supported yet."
        while len(ret) <= node_frame:
            ret.append([])
        ret[node_frame].append(node)
    return ret
