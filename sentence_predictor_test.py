'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Input, Flatten
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import random
import sys
import utils
import re
import sentence_predictor_utils



path_shakespeare = get_file('shakespeare.txt', origin='http://norvig.com/ngrams/shakespeare.txt')
text_shakespeare = open(path_shakespeare).read()
text_shakespeare = text_shakespeare.lower().replace('=', ' ').replace('2', ' ').replace('&c', ' ')
print('corpus length, Shakespeare:', len(text_shakespeare))


sentences_shakespeare = np.array(sentence_predictor_utils.split_into_sentences(text_shakespeare))
chars_shakespeare = sorted(list(set("".join(sentences_shakespeare))))

print('total chars, Shakespeare:', len(chars_shakespeare))

chars = sorted(list(set(chars_shakespeare)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def beam_search(model, sentence_enc, start_frag_enc, beam_width, num_candidates):
    num_chars = sentence_enc.shape[-1]
    EPS = 1e-10

    candidates = {0: start_frag_enc}
    candidate_scores = {0: 0.0}
    complete_sentences = []
    complete_scores = []

    # candidate_symbols = (-prediction).argsort(axis=None)[:beam_width]
    while len(complete_sentences) < num_candidates:

        score_matrix = -np.inf * np.ones((num_chars, beam_width))
        for i, candidate in candidates.items():
            p = model.predict([sentence_enc, candidate])[0, -1, :]
            p[p < 0.00000] = 0.00000
            p[p > 0.999999999] = 0.999999999
            score_matrix[:, i] = np.log(p + EPS) + candidate_scores[i]
        new_symbols = (-score_matrix).argsort(axis=None)[:beam_width]
        idxs = np.unravel_index(new_symbols, score_matrix.shape)

        count = 0
        new_candidates = {}
        new_candidate_scores = {}
        for idx in zip(idxs[0], idxs[1]):
            new_symbol_enc = np.zeros((1, 1, num_chars))
            new_symbol_enc[0, 0, idx[0]] = 1

            new_candidates[count] = np.concatenate([candidates[idx[1]], new_symbol_enc], axis=1)
            new_candidate_scores[count] = score_matrix[idx[0], idx[1]]

            if (indices_char[idx[0]] == utils.SENTENCE_END):
                complete_sentences.append(new_candidates[count])
                complete_scores.append(new_candidate_scores[count])
                new_candidate_scores[count] = -1e10

            count += 1

        candidates = new_candidates
        candidate_scores = new_candidate_scores

        for i, candidate in candidates.items():
            sentence = utils.sentence_decode(candidate, chars)
            print("Candidate %s: " % (i), sentence, 'Score: ', candidate_scores[i])
        print()

        for i, candidate in enumerate(complete_sentences):
            sentence = utils.sentence_decode(candidate, chars)
            print("Complete %s: " % (i), sentence, 'Score: ', complete_scores[i])
        print()


MODEL_NAME = 'logs/sentence_predictor_shakespeare_gru_prenet_conv_lr_anneal.00999-1.289-0.603.hdf5'

test_sentence = '#have i forgotten myself so far that i have not even told you his name ?_'
test_sentence = "#i do not know what you are talking about !_"
test_sentence = "#i believe robots are the future !_"
test_sentence = "#this is not important ._"
#test_sentence = "#please let me live in piece !_"

#test_sentence = "#i do not understand you !_"


model = load_model(MODEL_NAME, custom_objects={'categorical_accuracy_nonzero': utils.categorical_accuracy_nonzero})
# model = wmt_autoencoder
# model = shakespeare_autoencoder_forward

test_sentence = '#to be , or not to be : that is the question :_'

sentence_enc = utils.sentence_encode(test_sentence, chars)
fragment_enc = utils.sentence_encode('#', chars)

sentence_predicted = beam_search(model, sentence_enc, fragment_enc, beam_width=20, num_candidates=20)

a = 1;
