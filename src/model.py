import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
np.random.seed(1335)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )

class model():
    def __init__(self, max_features=20000, maxlen=200, sequence_length = 500):
        self.max_features = max_features
        self.maxlen = maxlen
        self.sequence_length = sequence_length




    def build_model_biLSTM(self):
        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=2000,
            output_mode="int",
            output_sequence_length=500,
        )

        inputs = keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(inputs)
        x = layers.Embedding(self.max_features + 1, 128)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        outputs = layers.Dense(6, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def build_model_FastText(self):

        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=1000,
            output_mode="int",
            output_sequence_length=500,
        )

        inputs = keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(inputs)
        x = layers.Embedding(self.max_features + 1, 128)(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(6, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model

    def build_model_LSTM(self):

        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=2000,
            output_mode="int",
            output_sequence_length=500,
        )

        inputs = keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(inputs)
        x = layers.Embedding(self.max_features + 1, 128)(x)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
        outputs = Dense(6, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model

    def build_model_ConvNet(self):

        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=2000,
            output_mode="int",
            output_sequence_length=500,
        )

        inputs = keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(inputs)
        x = layers.Embedding(self.max_features + 1, 128)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.GlobalMaxPooling1D()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = Dense(6, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model




