import tensorflow as tf
from tensorflow import keras
from keras import layers as ln
from keras import Model
from keras import activations as atv
from keras import utils



class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with tf.stop_gradient():
            self._mask = torch.triu(tf.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
'''
ones = tf.ones_like(x) # create a tensor all ones
mask = tf.greater(x, ones) # boolean tensor, mask[i] = True iff x[i] > 1
slice_y_greater_than_one = tf.boolean_mask(y, mask)

or 


x = tf.constant([1, 2, 0, 4])
y = tf.Variable([1, 2, 0, 4])
mask = x > 1
slice_y_greater_than_one = tf.boolean_mask(y, mask)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(slice_y_greater_than_one)) # [2 4]
'''


    @property
    def mask(self):
        return self._mask
    

class AnomalyAttention(utils.Sequence):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = ln.Dropout(attention_dropout)
        window_size = win_size
        self.distances = tf.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)
