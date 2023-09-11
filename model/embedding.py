import math
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras import layers as ln

class PositionalEmbedding(Model):
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


class TokenEmbedding(Model):
    def __init__(self, c_in, d_model,x):
        super(TokenEmbedding, self).__init__()
        #tensorflow not use version 1.x  if tensorflow.__version__ >= 2.x  padding = 1 
        #torch 1 if torch.__version__ >= '1.5.0' else 2
        input_length  = c_in[0]
        padding_size = 1
        padded_input_data = tf.pad(x, [[padding_size, padding_size]])
        filter_data = tf.constant([0.5, 1.0, 0.5], dtype=tf.float32)
        padded_input_length = input_length + 2 * padding_size
        padded_input_data = tf.reshape(padded_input_data, [1, padded_input_length, 1])
        filter_data = tf.reshape(filter_data, [filter_data.shape[0], 1, 1])
        self.tokenConv = ln.conv1d(padded_input_data, filter_data,input_shape=c_in, stride=1, padding='VALID',
                                    stride=1,kernel_size=3,use_bias=False)
       
       #module cshnge
        for m in self.modules():
            if isinstance(m, ln.Conv1d):
                tf.compat.v1.keras.initializers.glorot_uniform(seed=None,dtype=tf.dtypes.float32)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(Model):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = ln.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


