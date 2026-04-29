import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K


def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    loss = 1. - 2 * intersection / (union + K.epsilon())
    return loss

def ce_boundary_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)
    def boundary_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate gradients for both ground truth and predicted masks
        grad_true_x = tf.image.sobel_edges(y_true)[..., 0]
        grad_true_y = tf.image.sobel_edges(y_true)[..., 1]

        grad_pred_x = tf.image.sobel_edges(y_pred)[..., 0]
        grad_pred_y = tf.image.sobel_edges(y_pred)[..., 1]

        # Calculate boundary strength for both ground truth and predicted masks
        boundary_strength_true = tf.sqrt(grad_true_x**2 + grad_true_y**2)
        boundary_strength_pred = tf.sqrt(grad_pred_x**2 + grad_pred_y**2)

        # Calculate boundary loss using Huber loss to improve numerical stability
        huber_loss = tf.keras.losses.Huber(delta=1.0)
        boundary_loss = huber_loss(boundary_strength_true, boundary_strength_pred)

        # Check for NaN values in the loss and replace them with zero
        boundary_loss = tf.where(tf.math.is_nan(boundary_loss), tf.zeros_like(boundary_loss), boundary_loss)

        return boundary_loss
    boundary_loss = boundary_loss(y_true, y_pred)
    loss = ce_loss + boundary_loss
    return loss/2


def ja_dice_loss(alpha=0.7):
    def jd_loss(y_true, y_pred):
        a = alpha
        b = 1-a
        ja_loss = jaccard_loss(y_true, y_pred)

        dice = dice_loss(y_true, y_pred)
        loss = a * ja_loss + b * dice
        return loss
    return jd_loss



def ce_dice_loss(alpha=0.7):
    def cd_loss(y_true, y_pred):
        a = alpha
        b = 1-a
        ce_loss = binary_crossentropy(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        loss = a * ce_loss + b * dice
        return loss
    return cd_loss

def ce_dice_loss1(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    dice_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
    loss = ce_loss + dice_loss
    return loss

def mean_squared_error(y_true, y_pred):  
    """  
    计算均方误差损失。  
  
    :param y_true: 真实值  
    :param y_pred: 预测值  
    :return: 均方误差损失  
    """  
    # 计算平方差  
    squared_diff = tf.square(y_true - y_pred)  
    # 计算平均值  
    mean_squared_diff = tf.reduce_mean(squared_diff)  
    return mean_squared_diff  


def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    loss = 1 - iou
    return loss


def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    loss = 1. - intersection / (union + K.epsilon())
    return loss


def ce_jaccard_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
    loss = ce_loss + jaccard_loss
    return loss/2

# def focal_tversky_loss(alpha=0.5):
#     def ft(y_true, y_pred):
#         a = alpha
#         b = 1-a
#         focal = focal_loss()

#         tversky = tversky_loss(y_true, y_pred)
#         loss = a * focal + b * tversky
#         return loss
#     return ft

# def tversky_loss(y_true, y_pred):
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#     false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#     alpha = 0.7
#     return 1 - (true_pos + K.epsilon())/(true_pos + alpha * false_neg + (1-alpha) * false_pos + K.epsilon())
def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

# def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):  
#     """  
#     Compute focal loss for binary classification tasks.  
#     FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)  
#     where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.  
  
#     Args:  
#         y_true: A tensor of the same shape as `y_pred` with binary labels (0 or 1).  
#         y_pred: A tensor resulting from a sigmoid (binary case).  
#         alpha: A scalar tensor for weighting the importance of positive vs negative examples.  
#         gamma: A scalar tensor modulating the shape of the modulating factor (1-p_t)**gamma.  
  
#     Returns:  
#         Focal loss tensor.  
#     """  
#     # Clip predictions to prevent forming NaNs  
#     epsilon = tf.keras.backend.epsilon()  
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  
  
#     # Calculate p_t  
#     p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)  
  
#     # Calculate alpha_t  
#     alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)  
  
#     # Calculate focal loss  
#     loss = -alpha_factor * tf.math.pow(1. - p_t, gamma) * tf.math.log(p_t)  
  
#     # Sum the losses in mini_batch  
#     return tf.reduce_mean(loss)  


# def focal_loss(gamma=2., alpha=0.25):
#     def focal_loss_fixed(y_true, y_pred):
#         y_true = K.flatten(y_true)
#         y_pred = K.flatten(y_pred)
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         loss = - y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
#         return K.mean(loss)
#     return focal_loss_fixed

# def tversky_loss(y_true, y_pred, alpha=0.7):
#     def tversky(y_true, y_pred, alpha=0.7):
#         y_true_pos = K.flatten(y_true)
#         y_pred_pos = K.flatten(y_pred)
#         true_pos = K.sum(y_true_pos * y_pred_pos)
#         false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#         false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#         loss = 1 - (true_pos + K.epsilon())/(true_pos + alpha * false_neg + (1-alpha) * false_pos + K.epsilon())
#         return loss
#     return tversky
# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), 1 - y_pred, tf.zeros_like(y_pred))
#         loss = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) \
#                - (1 - alpha) * K.pow(pt_0, gamma) * K.log(pt_0)
#         return K.mean(loss)
#     return focal_loss_fixed

def weighted_cross_entropy_loss(y_true, y_pred, weights=0.25):  
    """  
    计算加权交叉熵损失。  
  
    :param y_true: 真实标签，形状通常为 [batch_size, num_classes]  
    :param y_pred: 预测输出，形状通常为 [batch_size, num_classes]  
    :param weights: 每个样本的权重，形状为 [batch_size]  
    :return: 加权交叉熵损失  
    """  
    # 确保y_pred的值在[0, 1]范围内  
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0., clip_value_max=1.)  
  
    # 计算交叉熵  
    cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)  
  
    # 应用权重  
    weighted_cross_entropy = tf.multiply(cross_entropy, weights)  
  
    # 计算平均加权损失  
    loss = tf.reduce_mean(weighted_cross_entropy)  
  
    return loss  




def boundary_loss_grad(y_true, y_pred):
    # 计算目标边界的梯度
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

    # 计算梯度差异的平方和
    grad_diff = tf.square(dy_true - dy_pred) + tf.square(dx_true - dx_pred)

    # 计算边界损失
    loss = tf.reduce_mean(grad_diff)

    return loss

# def ja_dice_loss(alpha=0.7):
#     def jd_loss(y_true, y_pred):
#         a = alpha
#         b = 1-a
#         ja_loss = jaccard_loss(y_true, y_pred)

#         dice = dice_loss(y_true, y_pred)
#         loss = a * ja_loss + b * dice
#         return loss
#     return jd_loss

# def boundary_Ja_dice_loss(a=0.1, b=0.9):
#     c = a
#     d = b
    
#     def boundary_dice_loss(y_true, y_pred):
#         alpha=0.7
#         boundary = boundary_loss_grad(y_true, y_pred)
#         dice = dice_loss(y_true, y_pred)
#         loss = alpha * boundary + (1 - alpha) * dice
#         return loss

#     def boundary_Ja_loss(y_true, y_pred):
#         alpha=0.5
#         boundary = boundary_loss_grad(y_true, y_pred)
#         Ja = jaccard_loss(y_true, y_pred)
#         loss = alpha * boundary + (1 - alpha) * Ja
#         return loss

#     def b_J_dice(y_true, y_pred):
#         ja = boundary_Ja_loss(y_true, y_pred)
#         dice = boundary_dice_loss(y_true, y_pred)

#         b_J_dice_value = c * ja + d * dice
#         return b_J_dice_value

#     return b_J_dice

def boundary_Ja_dice_loss(a=0.6, b=0.2):
    c = a
    d = b
    e = 1-c-d
    def b_J_dice(y_true, y_pred):
        alpha=0.7
        boundary = boundary_loss_grad(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        Ja = jaccard_loss(y_true, y_pred)
        loss = c * boundary + d * dice + e * Ja
        return loss
    return b_J_dice


def boundary_dice_loss(alpha=0.7):
    def b_dice(y_true, y_pred):
        a = alpha
        b = 1-a
        boundary = boundary_loss_grad(y_true, y_pred)

        dice = dice_loss(y_true, y_pred)
        loss = a * boundary + b * dice
        return loss
    return b_dice

def boundary_Ja_loss(alpha=0.7):
    def b_Ja(y_true, y_pred):
        a = alpha
        b = 1-a
        boundary = boundary_loss_grad(y_true, y_pred)

        Ja = jaccard_loss(y_true, y_pred)
        loss = a * boundary + b * Ja
        return loss
    return b_Ja



def boundary_jaccard_loss(y_true, y_pred):
    a = 0.7
    b = 1-a
    boundary = boundary_loss_grad(y_true, y_pred)

    jaccard = jaccard_loss(y_true, y_pred)
    loss = a * boundary + b * jaccard
    return loss

def boundary_loss_sobel(y_true, y_pred):
    # 计算边界图像
    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)

    # 计算 Boundary Loss
    loss = tf.reduce_mean(tf.abs(edge_true - edge_pred))

    return loss

def boundary_jaccard_loss8(y_true, y_pred):
    a = 0.8
    b = 1-a
    boundary = boundary_loss_grad(y_true, y_pred)

    jaccard = jaccard_loss(y_true, y_pred)
    loss = a * boundary + b * jaccard
    return loss
def boundary_jaccard_loss9(y_true, y_pred):
    a = 0.9
    b = 1-a
    boundary = boundary_loss_grad(y_true, y_pred)

    jaccard = jaccard_loss(y_true, y_pred)
    loss = a * boundary + b * jaccard
    return loss



# def focal_loss(gamma=2., alpha=.25):  
#     def focal_loss_fixed(y_true, y_pred):  
#         pt = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))  
#         pt = tf.clip_by_value(pt, 1e-9, 1. - 1e-9)  
#         return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))  
#     return focal_loss_fixed  

# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    
#     return focal_loss_fixed


