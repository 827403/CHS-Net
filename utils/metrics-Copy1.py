"""
Various metrics for image segmentation for keras an numpy.
Expects one-hot encoded label maps or predictions from softmax.
"""
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf



def true_positives():
    # determine number of matching values and return the sum
    true_positives = tf.keras.metrics.TruePositives()
    return true_positives


def true_negatives():
    # invert and then treat as true positives
    true_negatives = tf.keras.metrics.TrueNegatives()

    return true_negatives


def false_negatives():
    false_negatives = tf.keras.metrics.FalseNegatives()
    return false_negatives


def false_positives():
    false_positives = tf.keras.metrics.FalsePositives()
    return false_positives

def precision():
    # Precision = TP / (TP + FP)
    tp = true_positives()
    fp = false_positives()
    prec = tp / (tp + fp)
    return prec


def recall():
     # Recall = TP / (TP + FN)
    tp = true_positives()
    fn = false_negatives()
    rec = tp / (tp + fn)
    return rec


def specificity():
    # Specificity = TN  / (TN + FP)
    tn = true_negatives()
    fp = false_positives()
    s = tn / (tn + fp)
    return s


def iou():
    # IOU = TP / (TP + FN + FP)
    tp = true_positives()
    fn = false_negatives()
    fp = false_positives()
    i = tp / (tp + fn + fp)
    return i

def f1():
    # F1 2 * prec * rec / (prec + rec)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f = 2 * prec * rec / (prec + rec)
    return f


def mcc():
    # mcc = https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    tp = true_positives()
    tn = true_negatives()
    fn = false_negatives()
    fp = false_positives()
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return num / den


def accuracy():
    # acc = (tp + tn) / tp + fp + tn + fn
    tp = true_positives()
    tn = true_negatives()
    fn = false_negatives()
    fp = false_positives()
    return (tp + tn) / (tp + fp + tn + fn)

def dice():
    tp = true_positives()
    tn = true_negatives()
    fn = false_negatives()
    fp = false_positives()
    
    return (tp *2) / (tp *2+ fp + fn)


def jaccard():
    tp = true_positives()
    tn = true_negatives()
    fn = false_negatives()
    fp = false_positives()

    return (tp ) / (tp + fp + fn)


# def jaccard_coeff_z(y_true, y_pred):
#     intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#     score =  intersection / (union + K.epsilon())
#     return score 

# def dice_coeff_z(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return score

