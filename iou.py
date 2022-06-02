import tensorflow as tf

def iou(bbox1, bbox2):
    y1, y2, x1, x2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    y3, y4, x3, x4 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]
    inter_w = tf.reduce_min(
        tf.stack([x2, x4]), 0) - tf.reduce_max(tf.stack([x1, x3]), 0)
    inter_h = tf.reduce_min(
        tf.stack([y2, y4]), 0) - tf.reduce_max(tf.stack([y1, y3]), 0)
    pos = tf.logical_or(inter_w <= 0, inter_h <= 0)
    inter_area = inter_w * inter_h
    union_area = (y2-y1) * (x2-x1) + (y4-y3) * (x4-x3) - inter_area
    result = tf.where(pos, 0.0, inter_area / union_area)
    return 1 - result
