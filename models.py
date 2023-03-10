import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,LSTM
from keras.optimizers import Adam
from keras import Model
from keras.layers import Input, Dense, Bidirectional
from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense

def gru(data,classes):
    input_shape = data.shape[1],data.shape[2]
    input1 = Input(shape=input_shape) 
   # mask_input = keras.Input(data.shape[1])
    x = keras.layers.GRU(16, return_sequences=True)(input1)#,mask = mask_input
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(classes, activation="softmax")(x)
    rnn_model = keras.Model([input1], output)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    rnn_model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=[keras.metrics.CategoricalAccuracy()])
    rnn_model.summary()
    return rnn_model
# define model for simple BI-LSTM + DNN based binary classifier
def lstm(data,classes):
    input_shape = data.shape[1],data.shape[2]
    input1 = Input(shape=input_shape) 
    lstm1 = Bidirectional(LSTM(units=32))(input1)
    dnn_hidden_layer1 = keras.layers.Dense(3, activation='relu')(lstm1)
    dnn_output = keras.layers.Dense(classes, activation='softmax')(dnn_hidden_layer1)
    model = Model(inputs=[input1],outputs=[dnn_output])
    # compile the model
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[keras.metrics.CategoricalAccuracy()])
    model.summary()
    return model

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation=tf.nn.gelu),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def transformer(train_data,classes):
    sequence_length = train_data.shape[1]
    embed_dim = train_data.shape[2]
    dense_dim = 4
    num_heads = 1
    input_shape = train_data.shape[1],train_data.shape[2]
    inputs = Input(shape=input_shape)
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[keras.metrics.CategoricalAccuracy()])
    model.summary()
    return model
