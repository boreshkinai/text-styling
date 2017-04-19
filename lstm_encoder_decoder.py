from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Input, Flatten, Masking, merge, Reshape, Lambda, TimeDistributed
from keras.layers.merge import Concatenate, Add
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import random
import sys
import re

DROPOUT = 0.1
SENTENCE_BATCH_SIZE = 64
LSTM_WIDTH = 256
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
    text = text.replace(". ", "." + SENTENCE_END + "<stop> ")
    text = text.replace("? ", "?" + SENTENCE_END + "<stop> ")
    text = text.replace("! ", "!" + SENTENCE_END + "<stop> ")
    text = text.replace("<prd> ", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    out = []
    for s in sentences:
        if (len(s) > 30) and (len(s) < 500):
            out.append(SENTENCE_START + s)
    return out


path_shakespeare = get_file('shakespeare.txt', origin='http://norvig.com/ngrams/shakespeare.txt')
text_shakespeare = open(path_shakespeare).read()
text_shakespeare = text_shakespeare.lower().replace('\n', ' ').replace('=', ' ').replace(r"\\'", " ")
print('corpus length:', len(text_shakespeare))

# nltk.download()
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# tokenized = tokenizer.tokenize(text)

sentences_shakespeare = np.array(split_into_sentences(text_shakespeare))
sentences_shakespeare = sorted(sentences_shakespeare, key=len)
chars = sorted(list(set("".join(sentences_shakespeare))))

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def text_generator(sentences):
    cum_count = 0
    while 1:
        count = 0
        cum_count += 1
        for i in range(0, len(sentences), SENTENCE_BATCH_SIZE):  # len(sentences)
            print('batch number: ', count, ', cumulative batch number: ', cum_count)
            count += 1

            sentence_batch = sentences[i:i + SENTENCE_BATCH_SIZE]
            maxlen_batch = len(max(sentence_batch, key=len))

            X = np.zeros((SENTENCE_BATCH_SIZE, maxlen_batch, len(chars)), dtype=np.int32)
            y = np.zeros((SENTENCE_BATCH_SIZE, maxlen_batch, len(chars)), dtype=np.int32)
            w = np.zeros((SENTENCE_BATCH_SIZE, maxlen_batch), dtype=np.int32)

            for i, sentence in enumerate(sentence_batch):

                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1

                for t in range(len(sentence) - 1):
                    taget_pos = t + 1
                    y[i, t, char_indices[sentence[taget_pos]]] = 1
                    w[i, t] = 1

            yield ([X, X], y, w)


print('Build model...')


def concat_context(inputs):
    seq = inputs[0]
    c = inputs[1]
    c_tiled = K.tile(K.reshape(c, [-1, 1, K.shape(c)[1]]), (1, K.shape(seq)[1], 1))
    out = K.concatenate([seq, c_tiled], axis=2)

    boolean_mask = K.any(K.not_equal(seq, 0), axis=-1, keepdims=True)

    # K.print_tensor( out * K.cast(boolean_mask, K.floatx()) )

    return out * K.cast(boolean_mask, K.floatx())


def get_encoder(lstm_width, dropout):
    context_input = Input(shape=(None, len(chars)))
    x = Masking(mask_value=0)(context_input)
    x = GRU(lstm_width, return_sequences=True, go_backwards=True, dropout=dropout)(x)
    # xf = GRU(LSTM_WIDTH, return_sequences=True, go_backwards=False, dropout=0.0)(x)
    # x = Concatenate(axis=2)([xf, xb])
    x = GRU(lstm_width, return_sequences=True, dropout=dropout)(x)
    encoder_output = GRU(lstm_width, return_sequences=False, dropout=dropout)(x)

    return Model(inputs=[context_input], outputs=[encoder_output])


def get_autoencoder(encoder, lstm_width, dropout):
    context_input = Input(shape=(None, len(chars)))
    encoder_output = encoder(context_input)

    teacher_input = Input(shape=(None, len(chars)))
    decoder_input = Masking(mask_value=0)(teacher_input)

    context_layer = Lambda(concat_context)
    decoder_input_c = context_layer([decoder_input, encoder_output])

    y1 = GRU(lstm_width, return_sequences=True, dropout=dropout)(decoder_input_c)
    y2 = GRU(lstm_width, return_sequences=True, dropout=dropout)(y1)
    y3 = GRU(lstm_width, return_sequences=True, dropout=dropout)(y2)

    decoder_appended = context_layer([y3, encoder_output])

    decoder_appended = TimeDistributed(Dense(lstm_width, activation='relu'))(decoder_appended)
    decoder_output = TimeDistributed(Dense(len(chars), activation='softmax'))(decoder_appended)

    return Model(inputs=[context_input, teacher_input], outputs=[decoder_output])


encoder = get_encoder(lstm_width=LSTM_WIDTH, dropout=DROPOUT)
shakespeare_autoencoder = get_autoencoder(encoder, lstm_width=LSTM_WIDTH, dropout=DROPOUT)

optimizer = Adam(clipnorm=1.0)
shakespeare_autoencoder.compile(loss='categorical_crossentropy', optimizer=optimizer, sample_weight_mode="temporal")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


shakespeare_gen = text_generator(sentences_shakespeare)

for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    shakespeare_autoencoder.save('model_shakespeare.hd5')

    shakespeare_history = shakespeare_autoencoder.fit_generator(shakespeare_gen,
                                                                steps_per_epoch=len(
                                                                    sentences_shakespeare) / SENTENCE_BATCH_SIZE - 10,
                                                                epochs=1, verbose=1, workers=1)
