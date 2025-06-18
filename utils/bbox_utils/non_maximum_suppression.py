import numpy as np
from chainer.backends import cuda
import chainer

def _non_maximum_suppression_cpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)

def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    xp = cuda.get_array_module(bbox)
    flag = (xp == np)
    if flag == False:
        bbox = chainer.backends.cuda.to_cpu(bbox)
        # thresh = chainer.backends.cuda.to_cpu(thresh)
        if score is not None:
            score = chainer.backends.cuda.to_cpu(score)
        if limit is not None:
            limit = chainer.backends.cuda.to_cpu(limit)
    result = _non_maximum_suppression_cpu(bbox, thresh, score, limit)
    if flag == False:
        result = chainer.backends.cuda.to_gpu(result)
    return result

