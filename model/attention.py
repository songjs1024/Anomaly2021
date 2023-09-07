import os
import numpy as np
import math
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from keras import layers as ln
from keras import Model
from keras import activations as atv
from keras import utils



class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        if device == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        mask_shape = [B, 1, L, L]
        with tf.stop_gradient():
            ones = tf.ones_like(x)
            slice_y_greater_than_one = tf.boolean_mask(ones, mask_shape)
            self._mask = tf.linalg.band_part(slice_y_greater_than_one, -1,0)

    @property
    def mask(self):
        return self._mask
    
'''
*test
x = tf.constant([1, 2, 0, 4])
y = tf.Variable([1, 2, 0, 4])
mask = x > 1
slice_y_greater_than_one = tf.boolean_mask(y, mask)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(slice_y_greater_than_one)) # [2 4]
'''


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
      
    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
    #https://baekyeongmin.github.io/dev/einsum/
    #https://pytorch.org/docs/stable/generated/torch.einsum.html
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = tf.math.sigmoid(sigma * 5) + 1e-5
        sigma = tf.math.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * tf.math.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(tf.nn.softmax(attn, dim=-1))#11
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)
