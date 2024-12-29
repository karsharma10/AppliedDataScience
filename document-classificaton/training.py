import os
import shutil

import os
# *** IMPORTANT: Set the environment variable TF_USE_LEGACY_KERAS to 1 before importing tensorflow, due to incompatibility issues*** #
os.environ['TF_USE_LEGACY_KERAS']='1'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)


text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')


def build_classifier_model():
    # Input layer for raw text
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    # Preprocessing layer
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)

    # BERT encoder layer
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder') # we don't want to update the parameters in the encoder layer
    outputs = encoder(encoder_inputs)

    # Use the pooled_output for classification
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)

    # Dense layer for three-class classification
    net = tf.keras.layers.Dense(3, activation='softmax', name='classifier')(net)

    # Return the model
    return tf.keras.Model(text_input, net)

# Build the model
classifier_model = build_classifier_model()

# Compile the model
classifier_model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=3e-5), # running on arm max chip so using legacy optimizer
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Summary of the model
classifier_model.summary()