
## Setup


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab?
IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    !pip install -q -U tensorflow-addons
    !pip install -q -U transformers

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

from google.colab import files
uploaded = files.upload()

import pandas as pd

"""## Loading data"""

train_data = pd.read_csv("./te.translit.sampled.train.tsv", sep="\t", header=None)
val_data = pd.read_csv("./te.translit.sampled.dev.tsv", sep="\t", header=None)
test_data = pd.read_csv("./te.translit.sampled.test.tsv", sep="\t", header=None)
train_data.tail()

train_X, train_Y = train_data[1], train_data[0]
val_X, val_Y = val_data[1], val_data[0]

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
embed_size = 128

"""## Preparing Data"""

def prepare_data(file_path):
  input_texts = []
  target_texts = []
  input_characters = set()
  target_characters = set()
  with open(file_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    i = 1
  for line in lines[:-1]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

  return input_texts, target_texts, input_characters, target_characters


train_input_texts, train_target_texts, train_input_characters, train_target_characters = prepare_data("./te.translit.sampled.train.tsv")
val_input_texts, val_target_texts, val_input_characters, val_target_characters = prepare_data("./te.translit.sampled.dev.tsv")
test_input_texts, test_target_texts, test_input_characters, test_target_characters = prepare_data("./te.translit.sampled.test.tsv")

def tokenizer(data):
  fn = keras.preprocessing.text.Tokenizer(char_level=True, lower=False)
  fn.fit_on_texts([str(x) for x in data])
  return fn

#tokenizer.sequences_to_texts([[20, 6, 9, 8, 1]])

if __name__=="__main__":

    tokenizer_X = tokenizer(train_X)
    encoded_train_X = np.array(tokenizer_X.texts_to_sequences([str(x) for x in train_X]))
    encoded_val_X = np.array(tokenizer_X.texts_to_sequences([str(x) for x in val_X]))
    encoded_test_X = np.array(tokenizer_X.texts_to_sequences([str(x) for x in test_X]))
    num_encoder_tokens = len(tokenizer_X.word_index)

    tokenizer_Y = tokenizer(train_Y)
    encoded_train_Y = tf.convert_to_tensor(tokenizer_Y.texts_to_sequences([str(x) for x in train_Y]))
    encoded_val_Y = np.array(tokenizer_Y.texts_to_sequences([str(x) for x in val_Y]))
    encoded_test_Y = np.array(tokenizer_Y.texts_to_sequences([str(x) for x in test_Y]))
    num_decoder_tokens = len(tokenizer_Y.word_index)

    import tensorflow_addons as tfa

    encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    #sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

    embeddings_encoder = keras.layers.Embedding(num_encoder_tokens, embed_size)
    embeddings_decoder = keras.layers.Embedding(num_decoder_tokens, embed_size)
    encoder_embeddings = embeddings_encoder(encoder_inputs)
    decoder_embeddings = embeddings_decoder(decoder_inputs)

    encoder = keras.layers.LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embeddings)

    encoder_state = [state_h, state_c]

    sampler = tfa.seq2seq.sampler.TrainingSampler()

    decoder_cell = keras.layers.LSTMCell(256)
    output_layer = keras.layers.Dense(num_decoder_tokens)
    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler, output_layer=output_layer)

    final_outputs, final_state, final_sequence_lengths = decoder(decoder_embeddings, initial_state=encoder_state)

    Y_proba = tf.nn.softmax(final_outputs.rnn_output)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    history = model.fit(encoded_train_X, encoded_train_Y, epochs=5)



