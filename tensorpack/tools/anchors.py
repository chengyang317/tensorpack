import numpy as np
from tensorpack.framwork.memory import Memoized

__all__ = ['generate_anchors', 'anchors_shift']


@Memoized()
def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    :param base_size:
    :param ratios:
    :param scales:
    :return: (9, 4) matrix box
    """
    if not isinstance(ratios, np.ndarray):
        ratios = np.array(ratios)
    if not isinstance(scales, np.ndarray):
        scales = np.array(scales)
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in xrange(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


@Memoized()
def anchors_shift(anchors, width, height, feat_stride):
    """
    For an processed image(width*height), get a box for every pixel according to anchors in origin image.
    :param anchors:
    :param width:
    :param height:
    :param feat_stride:
    :return: (h,w,anchor_nums,4)
    """
    # Enumerate all shifts
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shape is (h,w,4), item form maybe like (0,16,0,16).
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=-1)

    # Enumerate all shifted anchors:
    # add anchor_nums anchors (1,1,anchor_nums,4) to cell shift_nums shifts (h,w,1,4) to
    # get shift anchors (h,w,anchor_nums,4)
    shift_anchors = anchors[np.newaxis, np.newaxis, :, :] + shifts[:, :, np.newaxis, :]
    return shift_anchors


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
