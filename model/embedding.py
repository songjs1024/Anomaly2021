import math
import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import layers as ln

class PositionalEmbedding(utils.Sequence):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = tf.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = tf.range(0, max_len).float().unsqueeze(1)
        div_term = (tf.range(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = tf.math.sin(position * div_term)
        pe[:, 1::2] = tf.math.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]





class DataEmbedding(utils.Sequence):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = ln.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
