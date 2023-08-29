from tensorflow import keras
from keras import layers as ln
from keras import Model
from keras import activations as atv

class EncoderLayer(Model):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.3, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = ln.Conv1d(kernel_size=1, name='conv_1')
        self.conv2 = ln.Conv1d(kernel_size=1, name='conv_2')
        self.norm1 = ln.LayerNormalization(d_model, name='laynorm1')
        self.norm2 = ln.LayerNormalization(d_model, name='laynorm2')
        self.dropout = ln.Dropout(dropout)
        self.activation = atv.relu if activation == "relu" else atv.gelu
        