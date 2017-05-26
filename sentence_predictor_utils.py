from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Input, Flatten, Masking, merge, Reshape, Lambda, TimeDistributed, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import re
import string
import tensorflow as tf
from itertools import compress
import utils
import keras
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler


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
    text = text.replace("\n\n", "\n")
    text = text.replace("\n\n\n", "\n")
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

    if ".'" in text: text = text.replace(".'", "' .")
    if "!'" in text: text = text.replace("!'", "'!")
    if "?'" in text: text = text.replace("?'", "'?")

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

    text = text.replace(".\n", "." + SENTENCE_END + "<stop> ")
    text = text.replace("?\n", "?" + SENTENCE_END + "<stop> ")
    text = text.replace("!\n", "!" + SENTENCE_END + "<stop> ")
    text = text.replace(";\n", ";" + SENTENCE_END + "<stop> ")
    text = text.replace(",\n", ";" + SENTENCE_END + "<stop> ")
    text = text.replace(" \n", " " + SENTENCE_END + "<stop> ")
    text = text.replace(":\n", ":" + SENTENCE_END + "<stop> ")

    text = text.replace("<prd> ", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    out = []
    valid_characters = set(string.printable)
    for s in sentences:
        if (len(s) > 10) and (len(s) < 500):
            out.append(SENTENCE_START + s.replace('\n', ' '))

    return out








