from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Input, Flatten, Masking, merge, Reshape, Lambda, TimeDistributed, Dropout
from keras.layers.merge import Concatenate, Add
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import random
import sys
import re
import string
import tensorflow as tf
from itertools import compress
import utils


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session=session)

CORRUPTION_PR = 0.0
DROPOUT = 0.2
SENTENCE_TRAIN_BATCH_SIZE = 64
SENTENCE_VALIDATION_BATCH_SIZE = 256
LSTM_WIDTH = 512
SENTENCE_START = '#'
SENTENCE_END = '_'

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    # if "\'" in text: text = text.replace("\'", " ")
    text = text.replace(" &apos;", "\'")
    text = text.replace("&quot;", '"')
    text = text.replace(". ", "." + SENTENCE_END + "<stop> ")
    text = text.replace("? ", "?" + SENTENCE_END + "<stop> ")
    text = text.replace("! ", "!" + SENTENCE_END + "<stop> ")
    text = text.replace("<prd> ", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    out = []
    valid_characters = set(string.printable)
    for s in sentences:
        if (len(s) > 30) and (len(s) < 500):

            out.append(SENTENCE_START + s)
    return out


path_shakespeare = get_file('shakespeare.txt', origin='http://norvig.com/ngrams/shakespeare.txt')
text_shakespeare = open(path_shakespeare).read()
text_shakespeare = text_shakespeare.lower().replace('\n', ' ').replace('=', ' ').replace(r"\\'", " ")
print('corpus length, Shakespeare:', len(text_shakespeare))

# path_wmt = get_file('WMT2014_train.en', origin='')
path_wmt = 'WMT2014_train.en'
text_wmt = open(path_wmt).read()
text_wmt = text_wmt.lower().replace('\n', ' ').replace('=', ' ').replace(r"\\'", " ")
# text_wmt = text_wmt.encode('ascii',errors='ignore')
text_wmt = re.sub(r'[^\x00-\x7f]',r'', text_wmt)
print('corpus length, WMT:', len(text_wmt))

# nltk.download()
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# tokenized = tokenizer.tokenize(text)

sentences_shakespeare = np.array(split_into_sentences(text_shakespeare))
sentences_shakespeare = sorted(sentences_shakespeare, key=len)
chars_shakespeare = sorted(list(set("".join(sentences_shakespeare))))

sentences_wmt = np.array(split_into_sentences(text_wmt))
sentences_wmt = sorted(sentences_wmt, key=len)
chars_wmt = sorted(list(set("".join(sentences_wmt))))

print('total chars, Shakespeare:', len(chars_shakespeare))
print('total chars, WMT:', len(chars_wmt))

chars = sorted(list(set(chars_wmt + chars_shakespeare)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))





print('Build model...')


def concat_context(inputs):
    seq = inputs[0]
    c = inputs[1]
    c_tiled = K.tile(K.reshape(c, [-1, 1, 512]), (1, K.shape(seq)[1], 1))
    out = K.concatenate([seq, c_tiled], axis=2)

    boolean_mask = K.any(K.not_equal(seq, 0), axis=-1, keepdims=True)

    # K.print_tensor( out * K.cast(boolean_mask, K.floatx()) )

    return out * K.cast(boolean_mask, K.floatx())


def get_encoder(lstm_width, dropout):
    context_input = Input(shape=(None, len(chars)))
    x = Masking(mask_value=0)(context_input)
    x = GRU(lstm_width, return_sequences=True, go_backwards=True, dropout=dropout, recurrent_dropout=dropout)(x)
    x = GRU(lstm_width, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(x)
    # xf = GRU(LSTM_WIDTH, return_sequences=True, go_backwards=False, dropout=0.0)(x)
    # x = Concatenate(axis=2)([xf, xb])
    encoder_output = GRU(lstm_width, return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(x)

    return Model(inputs=[context_input], outputs=[encoder_output])


def get_decoder_shared(encoder_in, lstm_width, dropout):

    context_input = Input(shape=(None, len(chars)))
    encoder_output = encoder_in(context_input)

    teacher_input = Input(shape=(None, len(chars)))
    decoder_input = Masking(mask_value=0)(teacher_input)

    context_layer1 = Lambda(concat_context)
    decoder_input_c = context_layer1([decoder_input, encoder_output])

    y1 = GRU(lstm_width, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(decoder_input_c)
    y2 = GRU(lstm_width, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(y1)

    return Model(inputs=[context_input, teacher_input], outputs=[y2, encoder_output])


def get_decoder_split(decoder_shared_in, lstm_width, dropout):

    context_input = Input(shape=(None, len(chars)))
    teacher_input = Input(shape=(None, len(chars)))
    shared_output = decoder_shared_in([context_input, teacher_input])

    y2 = shared_output[0]
    encoder_output = shared_output[1]

    y3 = GRU(lstm_width, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(y2)

    context_layer2 = Lambda(concat_context)
    decoder_appended = context_layer2([y3, encoder_output])

    decoder_appended = TimeDistributed(Dropout(0.5))(decoder_appended)
    decoder_appended = TimeDistributed(Dense(lstm_width, activation='relu'))(decoder_appended)
    decoder_appended = TimeDistributed(Dropout(0.5))(decoder_appended)
    decoder_output = TimeDistributed(Dense(len(chars), activation='softmax'))(decoder_appended)

    return Model(inputs=[context_input, teacher_input], outputs=[decoder_output])

encoder = get_encoder(lstm_width=LSTM_WIDTH, dropout=DROPOUT)
decoder_shared = get_decoder_shared(encoder_in=encoder, lstm_width=LSTM_WIDTH, dropout=DROPOUT)

shakespeare_autoencoder = get_decoder_split(decoder_shared_in=decoder_shared, lstm_width=LSTM_WIDTH, dropout=DROPOUT)
wmt_autoencoder = get_decoder_split(decoder_shared_in=decoder_shared, lstm_width=LSTM_WIDTH, dropout=DROPOUT)

optimizer = Adam(clipnorm=1.0)
shakespeare_autoencoder.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer, sample_weight_mode="temporal")
wmt_autoencoder.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer, sample_weight_mode="temporal")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

shakespeare_train_idx = np.random.uniform(size=(len(sentences_shakespeare),)) < 0.9
shakespeare_train_gen = utils.text_generator_random(list(compress(sentences_shakespeare, shakespeare_train_idx)), char_indices, SENTENCE_TRAIN_BATCH_SIZE, corruption_pr=CORRUPTION_PR)
shakespeare_validation_gen = utils.text_generator_deterministic(list(compress(sentences_shakespeare, np.invert(shakespeare_train_idx))), char_indices, SENTENCE_VALIDATION_BATCH_SIZE, corruption_pr=CORRUPTION_PR)

wmt_train_idx = np.random.uniform(size=(len(sentences_wmt),)) < 0.9
wmt_train_gen = utils.text_generator_random(list(compress(sentences_wmt, wmt_train_idx)), char_indices, SENTENCE_TRAIN_BATCH_SIZE, corruption_pr=CORRUPTION_PR)
wmt_validation_gen = utils.text_generator_deterministic(list(compress(sentences_wmt, np.invert(wmt_train_idx))), char_indices, SENTENCE_VALIDATION_BATCH_SIZE, corruption_pr=CORRUPTION_PR)

for iteration in range(0, 1000):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    shakespeare_autoencoder.save('model_shakespeare.hd5')
    wmt_autoencoder.save('model_wmt.hd5')

    if iteration < 10:
        shakespeare_autoencoder.fit_generator(shakespeare_train_gen,
                                              steps_per_epoch=25,
                                              epochs=1, verbose=1, workers=1)

        wmt_autoencoder.fit_generator(wmt_train_gen,
                                      25,  # steps_per_epoch=len(sentences_wmt) / SENTENCE_TRAIN_BATCH_SIZE - 10,
                                      epochs=1, verbose=1, workers=1)

    else:
        shakespeare_autoencoder.fit_generator(shakespeare_train_gen,
                                              steps_per_epoch=250, # sum(shakespeare_train_idx) / SENTENCE_TRAIN_BATCH_SIZE - 10,
                                              validation_data=shakespeare_validation_gen,
                                              validation_steps=sum(np.invert(
                                                  shakespeare_train_idx)) / SENTENCE_VALIDATION_BATCH_SIZE - 10,
                                              epochs=1, verbose=1, workers=1)
        if iteration % 1 == 0:
            wmt_autoencoder.fit_generator(wmt_train_gen,
                                          250,  # steps_per_epoch=len(sentences_wmt) / SENTENCE_TRAIN_BATCH_SIZE - 10,
                                          validation_data=wmt_validation_gen,
                                          validation_steps=sum(
                                              np.invert(wmt_train_idx)) / SENTENCE_VALIDATION_BATCH_SIZE - 10,
                                          epochs=1, verbose=1, workers=1)



