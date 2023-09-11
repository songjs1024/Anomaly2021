from tensorflow import keras
from keras import layers as ln
from keras import Model
from keras import activations as atv
from keras import Model
from .embedding import DataEmbedding, TokenEmbedding
from .attention import AnomalyAttention, AttentionLayer

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

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma
    
    
    
class Encoder(Model):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = Model(attn_layers)
        self.norm = norm_layer


class AnomalyTransformer(Model):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=ln.LayerNormalization(d_model)
        )

        self.projection = ln.Dense(d_model, c_out, bias=True)#수정 필요

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
