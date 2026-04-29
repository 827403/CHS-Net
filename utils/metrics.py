"""
Various metrics for image segmentation for keras an numpy.
Expects one-hot encoded label maps or predictions from softmax.
"""
import numpy as np
import tensorflow.keras.backend as K

import tensorflow as tf





def _true_positives(y_true, y_pred):
    # determine number of matching values and return the sum
    if tf.is_tensor(y_pred):
        return K.sum(y_true * K.round(y_pred))
    else:
        return (y_true * np.round(y_pred)).sum()


def _true_negatives(y_true, y_pred):
    # invert and then treat as true positives
    y_true = (y_true - 1) * -1
    y_pred = (y_pred - 1) * -1

    return _true_positives(y_true, y_pred)


def _false_negatives(y_true, y_pred):
    if tf.is_tensor(y_pred):
        return K.sum(K.clip((y_true - K.round(y_pred)), 0, 1))
    else:
        return (y_true - np.round(y_pred)).clip(0).sum()


def _false_positives(y_true, y_pred):
    if tf.is_tensor(y_pred):
        return K.sum(K.clip((K.round(y_pred) - y_true), 0, 1))
    else:
        return (np.round(y_pred) - y_true).clip(0).sum()

def _calc_mean_metric(y_true, y_pred, metric, starting_label=0):
    switcher = {
        'mPrec': precision,
        # TODO 'mAP': average_precision,
        'mf1': f1,
        'mAcc': accuracy,
        'mBacc': balanced_accuracy,
        'mRec': recall,
        'mIOU': iou,
        'mSpec': specificity,
        'mMcc': mcc,
        'mdice':dice,
        'mjaccard':jaccard,
        'mdice_coeff_z':dice_coeff_z,
        'mjaccard_coeff_z':jaccard_coeff_z    
        
    }

    try:
        func = switcher[metric]
    except KeyError:
        raise ValueError('Unkown Metric')
    if tf.is_tensor(y_true):  
        batch_size = tf.shape(y_pred)[0]  
        amount_labels = tf.shape(y_pred)[-1]  
        summation = tf.Variable(0.0, dtype=tf.float32)  # 确保 summation 是 float32 类型的张量  
        y_pred = tf.round(y_pred)  
  
        for batch_num in tf.range(batch_size):  
            for label in tf.range(starting_label, amount_labels):  
                # 使用 TensorFlow 操作进行 IOU 计算  
                iou_value = func(  
                    y_true[batch_num, :, :, label], y_pred[batch_num, :, :, label])  
                summation.assign_add(iou_value)  # 使用 assign_add 更新 summation  
  
    else:  
        amount_labels = y_pred.shape[-1]  
        batch_size = y_pred.shape[0]  
        summation = 0.0  # 确保 summation 是浮点数  
  
        for batch_num in range(batch_size):  
            for label in range(starting_label, amount_labels):  
                summation += func(  
                    y_true[batch_num, :, :, label], y_pred[batch_num, :, :, label])  
  
    # 如果 y_pred 是张量，确保 batch_size 和 amount_labels 也是张量并且数据类型是 float32  
    if tf.is_tensor(y_true):  
        batch_size = tf.cast(batch_size, tf.float32)  
        amount_labels = tf.cast(amount_labels, tf.float32)  
  
    return summation / (batch_size * amount_labels)
#     if tf.is_tensor(y_true):
#         amount_labels = K.int_shape(y_pred)[-1]
#         #batch_size = K.int_shape(y_pred)[0]
#         batch_size = tf.shape(y_pred)[0]  
#         summation = K.variable(0)
#         y_pred = K.round(y_pred)
#     else:
#         amount_labels = y_pred.shape[-1]
#         batch_size = y_pred.shape[0]
#         summation = 0

#     for batch_num in range(batch_size):
#         for label in range(starting_label, amount_labels):
#             summation = summation + func(
#                 y_true[batch_num, :, :, label], y_pred[batch_num, :, :, label])

#     return summation / (batch_size * amount_labels)


def recall(y_true, y_pred):
    # Recall = TP / P
    if tf.is_tensor(y_pred):
        return _true_positives(y_true, y_pred) / K.sum(y_true)
    else:
        return _true_positives(y_true, y_pred) / y_true.sum()


def specificity(y_true, y_pred):
    # Specificity = TN / N = TN / (TN + FN)
    return _true_positives(y_true, y_pred) / (
            _true_negatives(y_true, y_pred) +
            _false_negatives(y_true, y_pred))


def precision(y_true, y_pred):
    # Precision = TP / (TP + FP)
    tp = _true_positives(y_true, y_pred)
    return tp / (tp + _false_positives(y_true, y_pred))


def iou(y_true, y_pred):
    # IOU = TP / (TP + FN + FP)
    I = _true_positives(y_true, y_pred)
    U = I + _false_negatives(y_true, y_pred) + _false_positives(y_true, y_pred)

    if tf.is_tensor(y_pred):
        return K.switch(K.equal(U, 0), 1.0, I / U)
    else:
        return 1 if U is 0 else I / U


def balanced_accuracy(y_true, y_pred):
    # BA = (TPR + TNR) / 2 = (Recall + Specificity) / 2
    return (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def f1(y_true, y_pred):
    # F1 2 * prec * rec / (prec + rec)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec / (prec + rec))


def mcc(y_true, y_pred):
    # mcc = https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    tp = _true_positives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return num / den


def accuracy(y_true, y_pred):
    # acc = (tp + tn) / tp + fp + tn + fn
    tp = _true_positives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)

    return (tp + tn) / tp + fp + tn + fn

def dice(y_true, y_pred):
    
    tp = _true_positives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)

    return (tp *2) / (tp *2+ fp + fn)

def dice_coeff_z(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def jaccard(y_true, y_pred):
    
    tp = _true_positives(y_true, y_pred)
    fp = _false_positives(y_true, y_pred)
    tn = _true_negatives(y_true, y_pred)
    fn = _false_negatives(y_true, y_pred)

    return (tp ) / (tp + fp + fn)

def jaccard_coeff_z(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    score =  intersection / (union + K.epsilon())
    return score 


def mean_jaccard_coeff_z(y_true, y_pred):
    return _calc_mean_metric(y_true, y_pred, 'mjaccard_coeff_z')
def mean_dice_coeff_z(y_true, y_pred):
    return _calc_mean_metric(y_true, y_pred, 'mdice_coeff_z')

def mean_dice(y_true, y_pred):
    return _calc_mean_metric(y_true, y_pred, 'mdice')

def mean_jaccard(y_true, y_pred):
    return _calc_mean_metric(y_true, y_pred, 'mjaccard')


def mean_mcc(y_true, y_pred):
    """ Calculates the mean accuracy of two images for given colours.
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean accuracy
    """
    return _calc_mean_metric(y_true, y_pred, 'mMcc')


def mean_accuracy2(y_true, y_pred):
    """ Calculates the mean accuracy of two images for given colours.
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean accuracy
    """
    return _calc_mean_metric(y_true, y_pred, 'mAcc')


def mean_iou(y_true, y_pred):
    """" Calculates the mean IOU of two images for given colours.
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean IOU
    """
    return _calc_mean_metric(y_true, y_pred, 'mIOU')


def mean_precision(y_true, y_pred):
    """ Calculates the mean precision of two images for given colours.
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean precision
    """
    return _calc_mean_metric(y_true, y_pred, 'mPrec')


def mean_recall(y_true, y_pred):
    """ Calculates the mean recall of two images for given colours.
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean recall
    """
    return _calc_mean_metric(y_true, y_pred, 'mRec')


# def mean_average_precision(gt, pred, colours=[[0,0,0], [255,255,255]]):
#     """ Calculates the mean recall of two images for given colours.

#     # Arguments:
#         See parameters in :func:`_calc_mean_metric`

#     # Returns:
#         The mean average precision
#     """
#     return _calc_mean_metric(gt, pred, colours, 'mAP')

def mean_f1(y_true, y_pred):
    """ Calculates the mean f1 score.
    Currently calculates f1 score for each colour and then averages that. Can be calculated in many ways,
    see: http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean f1 score
    """
    return _calc_mean_metric(y_true, y_pred, 'mf1')


def mean_balanced_accuracy(y_true, y_pred):
    """ Calculates the mean f1 score.
    Currently calculates f1 score for each colour and then averages that. Can be calculated in many ways,
    see: http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean f1 score
    """
    return _calc_mean_metric(y_true, y_pred, 'mBacc')


def mean_specificity(y_true, y_pred):
    """ Calculates the mean specificity of two images for given colours.
    # Arguments:
        See parameters in :func:`_calc_mean_metric`

    # Returns:
        The mean specificity
    """
    return _calc_mean_metric(y_true, y_pred, 'mSpec')