import numpy as np
from data_types import *
from node import *
from utils import *
from enum import Enum
from abc import ABCMeta, abstractmethod

def sparse2dense(sparse, num):
    ret = np.zeros((num,), dtype=np.float32)
    ret[:] = MAX_AFFINITY_VAL
    for i, v in zip(sparse[0], sparse[1]):
        ret[i] = v
    return ret


def sparsemat2dense(sparse, dims):
    ret = np.zeros(dims, dtype=np.float32)
    ret[...] = MAX_AFFINITY_VAL
    for i, row in enumerate(sparse):
        for j, v in zip(row[0], row[1]):
            ret[i, j] = v
    return ret


def compute_matches_dense(nodes_from, nodes_to, affinity_list, camera_view, valid_idx = None):
    # start processing. loop over affinities
    matches = np.zeros((len(nodes_from), len(nodes_to)), dtype=np.float32)
    if valid_idx is not None:
        matches[...] = MAX_AFFINITY_VAL + 1
        for i, vidx in enumerate(valid_idx):
            if vidx.size > 0:
                matches[i, vidx] = 0
    for affinity in affinity_list:
        valid = matches < MAX_AFFINITY_VAL
        m, atype = affinity.compute(nodes_from, nodes_to, camera_view, valid=valid)
        if atype == AffinityOutputType.Dense:
            matches += m
        else:
            matches += sparsemat2dense(m, (len(nodes_from), len(nodes_to)))

    matches[matches >= MAX_AFFINITY_VAL] = MAX_AFFINITY_VAL + 1
    return matches



def dense_matches(sparse_matches):
    shape = sparse_matches[0]
    ret = np.zeros(shape, dtype=np.float32)
    ret[:] = MAX_AFFINITY_VAL
    for ijv in sparse_matches[1]:
        i, j, v = ijv
        ret[i, j] = v
    return ret


class BaseAffinityMetric(object):
    __metaclass__ = ABCMeta

    def __init__(self, abs_threshold, return_cost=True, weight=1.0, bias=0.0):
        self._abs_threshold = abs_threshold
        self._return_cost = return_cost
        self._weight = weight
        self._bias = bias
        self._type = AffinityTypes.Unknown

    @property
    def type(self):
        return self._type

    def compute(self, node_from, node_to, camera_view = None, valid = None):
        ret = []
        prev_atype = None
        for i, node in enumerate(node_from):
            if valid is None:
                m, atype = self(node, node_to, camera_view)
            else:
                m = [ MAX_AFFINITY_VAL for j in range(valid.shape[1]) ]
                atype = AffinityOutputType.Dense
                # compute matches for only valid ones
                idx = np.where(valid[i])[0]
                if idx.size == 0:
                    # no valid ones
                    pass
                else:
                    temp, temp_atype = self(node, [ node_to[j] for j in idx ], camera_view)
                    if temp_atype == AffinityOutputType.Sparse:
                        temp = sparse2dense(temp, valid.shape[1])
                    for j, jj in enumerate(idx):
                        m[jj] = temp[j]

            ret.append(m)
            if prev_atype is not None:
                assert prev_atype == atype
            prev_atype = atype

        if prev_atype is None:
            prev_atype = AffinityOutputType.Dense
            ret= np.zeros((0,0))
        elif prev_atype == AffinityOutputType.Dense:
            ret = np.array(ret)

        return ret, prev_atype

    @abstractmethod
    def __call__(self, node_from, node_to, camera_view=None):
        '''
        Compute the affinity between two nodes
        '''
        raise NotImplementedError

    def refresh(self):
        '''
        Refresh cached information
        '''
        pass

    def set_context(self, nodes_from, nodes_to):
        '''
        Set any contextual information
        '''
        pass


######################################################################
## Affinity Definition
######################################################################
class SumAffinities(BaseAffinityMetric):
    '''
    Sum the two metric
    It allow to bipass threhold of each if any of them are satisfied.
    '''

    def __init__(self, affinity_list, abs_threshold=np.Inf):
        super(SumAffinities, self).__init__(abs_threshold)
        self._affinity_list = affinity_list
        assert len(self._affinity_list) > 1

    def refresh(self):
        for a in self._affinity_list:
            a.refresh()

    def set_context(self, nodes_from, nodes_to):
        for a in self._affinity_list:
            a.set_context(nodes_from, nodes_to)

    def compute(self, node_from, node_to, camera_view = None, valid = None):
        data, atype = self._affinity_list[0].compute(node_from, node_to, camera_view, valid=valid)
        if atype == AffinityOutputType.Sparse:
            ret = sparsemat2dense(data, (len(node_from), len(node_to)))
        else:
            ret = data

        for i in range(1, len(self._affinity_list)):
            data, atype = self._affinity_list[i].compute(node_from, node_to, camera_view, valid=valid)
            if atype == AffinityOutputType.Sparse:
                mat = sparse2dense(data, len(node_to))
            else:
                mat = data
            ret += mat

        ret[ret > self._abs_threshold] = np.Inf
        return ret, AffinityOutputType.Dense

    def __call__(self, node_from, node_to, camera_view=None):
        ret = np.zeros((len(node_to),), dtype=np.float32)
        candidate_idx = np.arange(len(node_to))
        node_to = np.array(node_to)
        for affinity in self._affinity_list:
            if candidate_idx.size == 0:  # empty
                break
            m, atype = affinity(node_from, list(node_to[candidate_idx]), camera_view)
            if atype == AffinityOutputType.Dense:
                m = np.array(m)
                ret[candidate_idx] += m
                candidate_idx = candidate_idx[m < MAX_AFFINITY_VAL]
            else:
                candidate_idx = candidate_idx[np.array(m[0], dtype=int)]
                assert len(candidate_idx) == len(m[1])
                ret[candidate_idx] += np.array(m[1])

        idx = [i for i in candidate_idx if ret[i] < self._abs_threshold]
        values = [ret[i] for i in idx]

        return (idx, values), AffinityOutputType.Sparse


class FastTimeDiffAffinity(BaseAffinityMetric):
    '''
    Compute the temporal distance and return potential matches within abs_threhold distance.
    node_from must precede the node_to to be a valid pair.
    '''

    def __init__(self, abs_threshold, return_cost=True, weight=1.0, bias=0.0, allow_overlap=False):
        super(FastTimeDiffAffinity, self).__init__(abs_threshold, return_cost, weight, bias)
        self._allow_overlap = allow_overlap

    def __call__(self, node_from, node_to, camera_view=None):
        '''
        TODO: assume node_to are sorted -> use binary search instead of linear sweep
        '''
        assert camera_view is not None
        from_end = node_from[camera_view].end

        if self._allow_overlap:
            idx = [i for i, node in enumerate(node_to) \
                   if node[camera_view].end > from_end and \
                   node[camera_view].start <= from_end + 1 + self._abs_threshold]
        else:
            # 0 ~ threhold temporal distance only
            idx = [i for i, node in enumerate(node_to) \
                   if node[camera_view].start > from_end and \
                   node[camera_view].start <= from_end + 1 + self._abs_threshold]
        values = [node_to[i][camera_view].start - from_end - 1 for i in idx]

        if self._return_cost:
            values = [self._weight * v for v in values]
        else:
            raise NotImplementedError

        return (idx, values), AffinityOutputType.Sparse



class XDistAffinity(BaseAffinityMetric):
    def __init__(self, abs_threshold, return_cost=True, weight=1.0, bias=0.0):
        super(XDistAffinity, self).__init__(abs_threshold, return_cost, weight, bias)
        self._type = AffinityTypes.DirectedAffinity

    def __call__(self, node_from, node_to, camera_view=None):
        '''
        Compute the x dist between temporally nearest boxes
          node_from: one node item
          node_to: list of nodes
        '''
        assert isinstance(node_to, list)
        assert camera_view is not None
        # assert camera_view in node_from.cameras
        # get the adjacent boxes
        # last frame box
        box1 = np.array(node_from[camera_view].feature(FeatureTypes.Box2D)[-1], dtype=np.float32)
        # assume every one has the camera_view
        # to assure that use SameCameraAffinity beforehand
        box2 = np.array([node[camera_view].feature(FeatureTypes.Box2D)[0] for node in node_to], dtype=np.float32)
        # compute the L2 Dist in Y space

        ret = np.abs((box2[:, 0] + box2[:, 2]) / 2.0
                     - (box1[0] + box1[2]) / 2.0)

        # apply thresholding
        ret[ret > self._abs_threshold] = np.Inf
        if self._return_cost:
            return self._weight * ret, AffinityOutputType.Dense
        else:
            raise NotImplementedError


class YDistAffinity(BaseAffinityMetric):
    def __init__(self, abs_threshold, return_cost=True, weight=1.0, bias=0.0):
        super(YDistAffinity, self).__init__(abs_threshold, return_cost, weight, bias)
        self._type = AffinityTypes.DirectedAffinity

    def __call__(self, node_from, node_to, camera_view=None):
        '''
        Compute the y dist between temporally nearest boxes
          node_from: one node item
          node_to: list of nodes
        '''
        assert isinstance(node_to, list)
        assert camera_view is not None
        # assert camera_view in node_from.cameras
        # get the adjacent boxes
        # last frame box
        box1 = np.array(node_from[camera_view].feature(FeatureTypes.Box2D)[-1], dtype=np.float32)
        # assume every one has the camera_view
        # to assure that use SameCameraAffinity beforehand
        box2 = np.array([node[camera_view].feature(FeatureTypes.Box2D)[0] for node in node_to], dtype=np.float32)
        # compute the L2 Dist in Y space

        ret = np.abs(box2[:, 1] - box1[1])
        ret += np.abs((box2[:, 1] + box2[:, 3]) - (box1[1] + box1[3]))
        ret /= 2.0

        # apply thresholding
        ret[ret > self._abs_threshold] = np.Inf
        if self._return_cost:
            return self._weight * ret, AffinityOutputType.Dense
        else:
            raise NotImplementedError


class IoUAffinity(BaseAffinityMetric):
    def __init__(self, abs_threshold, return_cost=True, weight=1.0, bias=0.0):
        super(IoUAffinity, self).__init__(abs_threshold, return_cost, weight, bias)
        self._type = AffinityTypes.DirectedAffinity

    def __call__(self, node_from, node_to, camera_view=None):
        '''
        Compute the IoU between temporally nearest boxes
          node_from: one node item
          node_to: list of nodes
        '''
        assert isinstance(node_to, list)
        assert camera_view is not None
        # assert camera_view in node_from.cameras
        # get the adjacent boxes
        # last frame box
        box1 = np.array(node_from[camera_view].feature(FeatureTypes.Box2D)[-1], dtype=np.float32)
        # assume every one has the camera_view
        # to assure that use SameCameraAffinity beforehand
        box2 = np.array([node[camera_view].feature(FeatureTypes.Box2D)[0] for node in node_to], dtype=np.float32)
        # compute the IoU
        ret = iou(box1, box2)
        # apply thresholding
        ret[ret < self._abs_threshold] = -np.Inf
        if self._return_cost:
            # change to cost format else we can do -np.log(v + 1e-10)
            return self._weight * (1.0 - ret), AffinityOutputType.Dense
            # return self._weight * ( - np.log(ret + 1e-30) ), AffinityOutputType.Dense
        else:
            return self._weight * ret, AffinityOutputType.Dense

class SingleViewAppearanceAffinity(BaseAffinityMetric):
    def __init__(self, metric_type, abs_threshold, return_cost = True, weight = 1.0, bias = 0.0, budget = 200):
        super(SingleViewAppearanceAffinity, self).__init__(abs_threshold, return_cost, weight, bias)
        self._type = AffinityTypes.UndirectedAffinity
        self._budget = budget

        def median_distance(x, y):
            xm = np.median(x, axis=1)
            xy = np.median(y, axis=1)
            distance = cosine_distance(xm, xy)
            return distance

        if metric_type == 'median':
            self._metric = median_distance
        else:
            raise ValueError("Invalid metric; must be 'median'")


    def __call__(self, node_from, node_to, camera_view = None):
        '''
        Compute the appearance difference between two nodes
          node_from: one node item
          node_to: list of nodes
        '''
        assert isinstance(node_to, list)
        assert len(node_to) > 0
        assert camera_view is not None
        assert camera_view in node_from.cameras

        node_from_median_model = node_from[camera_view].model(ModelTypes.MedianAppearance)
        if node_from_median_model is None: # no model is there. do on the fly computation
            assert False, "deprecated"
            # need to collect the data and take median
            node_from_appearance = node_from[camera_view].feature(FeatureTypes.ReID)
            if len(node_from_appearance) > self._budget:
                node_from_appearance = node_from_appearance[-self._budget:]
            node_from_appearance = np.expand_dims(node_from_appearance, axis=0)
            node_to_appearance = [node[camera_view].feature(FeatureTypes.ReID) for node in node_to]
            distances = self._metric(node_from_appearance, node_to_appearance)[0]
        else:
            distances = np.zeros((len(node_to), ), dtype=np.float32)
            distances[:] = np.inf
            # already has the model updated.
            node_from_appearance = node_from_median_model.get('rear')
            if node_from_appearance is not None:
                node_to_appearance = [ node[camera_view].model(ModelTypes.MedianAppearance).get('front') \
                                        for node in node_to ]
                valid_idx = [ i for i, f in enumerate(node_to_appearance) if f is not None ]
                if valid_idx:
                    valid_features = [ node_to_appearance[i] for i in valid_idx ]
                    distances[valid_idx] = cosine_distance(
                                node_from_appearance[np.newaxis, :],
                                valid_features)[0]

        # apply threshold
        distances[distances > self._abs_threshold] = np.Inf
        if self._return_cost:
            # change to cost format else we can do -np.log(v + 1e-10)
            return self._weight * distances, AffinityOutputType.Dense
        else:
            assert False, 'no similarity definition yet'

class NoneCompeteDirectedAffinity(BaseAffinityMetric):
    '''
    Compare the affinity difference between tracks - detections
    Filter-out ambiguous cases where the affinity diff < threshold
    '''
    def __init__(self, abs_threshold, sim_threshold, base_affinity, reject=True):
        super(NoneCompeteDirectedAffinity, self).__init__(abs_threshold)
        self._sim_threshold = sim_threshold
        self._base_affinity = base_affinity
        self._reject = reject
    
    def rejected(self):
        if self._reject:
            return list(self._row_reject), list(self._col_reject)
        else:
            return [], []

    def compute(self, node_from, node_to, camera_view = None, valid = None):
        data, atype = self._base_affinity.compute(node_from, node_to, camera_view, valid=valid)
        if atype == AffinityOutputType.Sparse:
            data = sparsemat2dense(data, (len(node_from), len(node_to)))
        # remove inf to avoid warning
        data[data > MAX_AFFINITY_VAL] = MAX_AFFINITY_VAL + 1

        # now check the filtering
        # Directed: row -> col connection always
        row_rejected = []
        col_rejected = []

        row_compete = []
        for row in range(data.shape[0]):
            idx = data[row].argmin()
            minv = data[row, idx]
            others = data[row, [ i != idx for i in range(data.shape[1]) ]]
            if others.size == 0:
                continue
            if (others - minv).min() < self._sim_threshold:
                row_compete.append(row)
                row_rejected.append(row)

        col_compete = []
        for col in range(data.shape[1]):
            idx = data[:, col].argmin()
            minv = data[idx, col]
            others = data[[ i != idx for i in range(data.shape[0]) ], col]
            if others.size == 0:
                continue
            if (others - minv).min() < self._sim_threshold:
                col_compete.append(col)
                col_rejected.append(col)
                # potentially reject tracks that are competing for one detection
                row_rejected += np.where( (data[:, col] - minv) < self._sim_threshold )[0].tolist()

        # remove ambiguous ones
        data[row_compete] = MAX_AFFINITY_VAL
        data[:, col_compete] = MAX_AFFINITY_VAL
        # apply threshold
        data[ data > self._abs_threshold] = MAX_AFFINITY_VAL

        # record these for later use
        self._row_reject = set(row_rejected)
        for row in range(data.shape[0]):
            if row in self._row_reject and np.any(data[row] <= self._abs_threshold):
                self._row_reject.remove(row)

        self._col_reject = set(col_rejected)
        for col in range(data.shape[1]):
            if col in self._col_reject and np.any(data[:, col] <= self._abs_threshold):
                self._col_reject.remove(col)

        return data, AffinityOutputType.Dense

    def __call__(self, node_from, node_to, camera_view = None):
        assert False, "Does not work with this"
