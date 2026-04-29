"""
Various metrics for image segmentation for keras an numpy.
Expects one-hot encoded label maps or predictions from softmax.
"""
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf



def _true_positives(y_true, y_pred):
    # determine number of matching values and return the sum
    true_positives = tf.keras.metrics.TruePositives()(y_true, y_pred)
    return true_positives


def _true_negatives(y_true, y_pred):
    # invert and then treat as true positives
    true_negatives = tf.keras.metrics.TrueNegatives()(y_true, y_pred)

    return true_negatives


def _false_negatives(y_true, y_pred):
    false_negatives = tf.keras.metrics.FalseNegatives()(y_true, y_pred)
    return false_negatives


def _false_positives(y_true, y_pred):
    false_positives = tf.keras.metrics.FalsePositives()(y_true, y_pred)
    return false_positives

def precision(y_true, y_pred):
    # Precision = TP / (TP + FP)
    tp = _true_positives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    prec = tp / (tp + fp)
    return prec


def recall(y_true, y_pred):
     # Recall = TP / (TP + FN)
    tp = _true_positives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    rec = tp / (tp + fn)
    return rec


def specificity(y_true, y_pred):
    # Specificity = TN  / (TN + FP)
    tn = _true_negatives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    s = tn / (tn + fp)
    return s


def iou(y_true, y_pred):
    # IOU = TP / (TP + FN + FP)
    tp = _true_positives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    i = tp / (tp + fn + fp)
    return i

def f1(y_true, y_pred):
    # F1 2 * prec * rec / (prec + rec)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f = 2 * prec * rec / (prec + rec)
    return f


def mcc(y_true, y_pred):
    # mcc = https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    tp = _true_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return num / den


def accuracy(y_true, y_pred):
    # acc = (tp + tn) / tp + fp + tn + fn
    tp = _true_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    return (tp + tn) / (tp + fp + tn + fn)

def dice(y_true, y_pred):
    tp = _true_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    
    return (tp *2) / (tp *2+ fp + fn)


def jaccard(y_true, y_pred):
    tp = _true_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)

    return (tp ) / (tp + fp + fn)


def jaccard_coeff_z(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    score =  intersection / (union + K.epsilon())
    return score 

def dice_coeff_z(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

