import numpy as np
import re
import tensorflow as tf
import keras.backend as K


SENTENCE_START = '#'
SENTENCE_END = '_'


caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def categorical_accuracy_nonzero(y_true, y_pred):
    score_array = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())

    weights = K.cast( K.any(K.not_equal(y_true, 0), axis=-1, keepdims=False), K.floatx())
    score_array *= weights
    score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))

    return K.mean(score_array)



def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    #if "\'" in text: text = text.replace("\'", " ")
    text = text.replace(" &apos;", "\'")
    text = text.replace("&quot;", '"')
    text = text.replace(". ","."+SENTENCE_END+"<stop> ")
    text = text.replace("? ","?"+SENTENCE_END+"<stop> ")
    text = text.replace("! ","!"+SENTENCE_END+"<stop> ")
    text = text.replace("<prd> ",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    out = []
    for s in sentences:
        if (len(s) > 30) and (len(s) < 500):
            out.append(SENTENCE_START+s)
    return out



def sentence_encode(sentence, chars):
    char_indices = dict((c, i) for i, c in enumerate(chars))
    y = np.zeros((1, len(sentence), len(char_indices)), dtype=np.int32)
    for t, char in enumerate(sentence):
        y[0, t, char_indices[char]] = 1
    return y


def sentence_decode(sentence_enc, chars):
    indices_char = dict((i, c) for i, c in enumerate(chars))
    predicted_sec = ''
    predicted_symbols = np.argmax(sentence_enc, axis=-1).ravel()
    for t in range(len(predicted_symbols)):
        predicted_sec += indices_char[predicted_symbols[t]]
    return predicted_sec


def sentence_predict_generator_context_empty(sentences, sentence_idxs, char_indices, batch_size, corruption_pr):
    cum_count = 0
    MAX_CONTEXT_LEN = 5

    batch_size = min(batch_size, len(sentence_idxs))
    sentence_idxs = np.asarray([i for i in np.arange(len(sentence_idxs)) if sentence_idxs[i] == True])
    sentence_idxs = sentence_idxs[sentence_idxs > MAX_CONTEXT_LEN]

    while 1:
        cum_count += 1

        num_context_sentences = np.random.random_integers(1, MAX_CONTEXT_LEN, 1)[0]

        batch_idxs = sentence_idxs[np.random.random_integers(num_context_sentences, len(sentence_idxs)-1, batch_size)]
        sentence_batch = sentences[batch_idxs]
        maxlen_teacher_batch = len(max(sentence_batch, key=len))

        maxlen_context_batch = 0
        for i, sentence in enumerate(sentence_batch):
            context = sentences[batch_idxs[i]-num_context_sentences:batch_idxs[i]]
            context_len = sum(len(s) for s in context)
            maxlen_context_batch = max(maxlen_context_batch, context_len)

        X_context = np.zeros((batch_size, maxlen_context_batch + num_context_sentences, len(char_indices)), dtype=np.int32)
        for i, sentence in enumerate(sentence_batch):
            context = sentences[batch_idxs[i] - num_context_sentences:batch_idxs[i]]
            context = " ".join(context)
            for t, char in enumerate(context):
                char = ' '
                X_context[i, t, char_indices[char]] = 1

        X_teacher = np.zeros((batch_size, maxlen_teacher_batch, len(char_indices)), dtype=np.int32)
        y = np.zeros((batch_size, maxlen_teacher_batch, len(char_indices)), dtype=np.int32)
        w = np.zeros((batch_size, maxlen_teacher_batch), dtype=np.int32)

        for i, sentence in enumerate(sentence_batch):
            corruption_flag = np.random.uniform(0.0, 1.0, len(sentence)) < corruption_pr
            corrupted_sym = np.random.random_integers(low=0, high=len(char_indices)-1, size=len(sentence))

            for t, char in enumerate(sentence):
                if corruption_flag[t]:
                    X_teacher[i, t, corrupted_sym[t]] = 1
                else:
                    X_teacher[i, t, char_indices[char]] = 1

            for t in range(len(sentence) - 1):
                taget_pos = t + 1
                y[i, t, char_indices[sentence[taget_pos]]] = 1
                w[i, t] = 1

        yield ([X_context, X_teacher], y, w)


def sentence_predict_generator_random(sentences, sentence_idxs, char_indices, batch_size, corruption_pr):
    cum_count = 0
    MAX_CONTEXT_LEN = 5

    # len_answers = list(map(len, sentences))
    # sorted_idx = sorted(range(len(sentence_idxs)), key=len_answers.__getitem__)
    batch_size = min(batch_size, len(sentence_idxs))
    sentence_idxs = np.asarray([i for i in np.arange(len(sentence_idxs)) if sentence_idxs[i] == True])
    sentence_idxs = sentence_idxs[sentence_idxs > MAX_CONTEXT_LEN]

    while 1:
        cum_count += 1

        num_context_sentences = np.random.random_integers(1, MAX_CONTEXT_LEN, 1)[0]

        batch_idxs = sentence_idxs[np.random.random_integers(num_context_sentences, len(sentence_idxs)-1, batch_size)]
        sentence_batch = sentences[batch_idxs]
        maxlen_teacher_batch = len(max(sentence_batch, key=len))

        maxlen_context_batch = 0
        for i, sentence in enumerate(sentence_batch):
            context = sentences[batch_idxs[i]-num_context_sentences:batch_idxs[i]]
            context_len = sum(len(s) for s in context)
            maxlen_context_batch = max(maxlen_context_batch, context_len)

        X_context = np.zeros((batch_size, maxlen_context_batch + num_context_sentences, len(char_indices)), dtype=np.int32)
        for i, sentence in enumerate(sentence_batch):
            context = sentences[batch_idxs[i] - num_context_sentences:batch_idxs[i]]
            context = " ".join(context)
            for t, char in enumerate(context):
                X_context[i, t, char_indices[char]] = 1

        X_teacher = np.zeros((batch_size, maxlen_teacher_batch, len(char_indices)), dtype=np.int32)
        y = np.zeros((batch_size, maxlen_teacher_batch, len(char_indices)), dtype=np.int32)
        w = np.zeros((batch_size, maxlen_teacher_batch), dtype=np.int32)

        for i, sentence in enumerate(sentence_batch):
            corruption_flag = np.random.uniform(0.0, 1.0, len(sentence)) < corruption_pr
            corrupted_sym = np.random.random_integers(low=0, high=len(char_indices)-1, size=len(sentence))

            for t, char in enumerate(sentence):
                if corruption_flag[t]:
                    X_teacher[i, t, corrupted_sym[t]] = 1
                else:
                    X_teacher[i, t, char_indices[char]] = 1

            for t in range(len(sentence) - 1):
                taget_pos = t + 1
                y[i, t, char_indices[sentence[taget_pos]]] = 1
                w[i, t] = 1

        yield ([X_context, X_teacher], y, w)

def text_generator_deterministic(sentences, char_indices, batch_size, corruption_pr):
    cum_count = 0
    while 1:
        count = 0
        cum_count += 1
        batch_size = min(batch_size, len(sentences))
        for i in range(0, len(sentences), batch_size):  # len(sentences)
            #print('batch number: ', count, ', cumulative batch number: ', cum_count)
            count += 1

            sentence_batch = sentences[i:i + batch_size]
            maxlen_batch = len(max(sentence_batch, key=len))

            X = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
            X_teacher = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
            y = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
            w = np.zeros((batch_size, maxlen_batch), dtype=np.int32)

            for i, sentence in enumerate(sentence_batch):
                corruption_flag = np.random.uniform(0.0, 1.0, len(sentence)) < corruption_pr
                corrupted_sym = np.random.random_integers(low=0, high=len(char_indices)-1, size=len(sentence))
                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1
                    if corruption_flag[t]:
                        X_teacher[i, t, corrupted_sym[t]] = 1
                    else:
                        X_teacher[i, t, char_indices[char]] = 1

                for t in range(len(sentence) - 1):
                    taget_pos = t + 1
                    y[i, t, char_indices[sentence[taget_pos]]] = 1
                    w[i, t] = 1

            yield ([X, X_teacher], y, w)


def text_generator_random(sentences, char_indices, batch_size, corruption_pr):
    cum_count = 0
    while 1:
        count = 0
        cum_count += 1
        batch_size = min(batch_size, len(sentences))
        batch_start = np.random.random_integers(0, len(sentences)-batch_size, 1)[0]

        count += 1

        sentence_batch = sentences[batch_start:batch_start + batch_size]
        maxlen_batch = len(max(sentence_batch, key=len))

        X = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
        X_teacher = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
        y = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
        w = np.zeros((batch_size, maxlen_batch), dtype=np.int32)

        for i, sentence in enumerate(sentence_batch):
            corruption_flag = np.random.uniform(0.0, 1.0, len(sentence)) < corruption_pr
            corrupted_sym = np.random.random_integers(low=0, high=len(char_indices)-1, size=len(sentence))
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
                if corruption_flag[t]:
                    X_teacher[i, t, corrupted_sym[t]] = 1
                else:
                    X_teacher[i, t, char_indices[char]] = 1

            for t in range(len(sentence) - 1):
                taget_pos = t + 1
                y[i, t, char_indices[sentence[taget_pos]]] = 1
                w[i, t] = 1

        yield ([X, X_teacher], y, w)


def text_backwards_generator_random(sentences, char_indices, batch_size, corruption_pr):
    cum_count = 0
    while 1:
        count = 0
        cum_count += 1
        batch_size = min(batch_size, len(sentences))
        batch_start = np.random.random_integers(0, len(sentences)-batch_size, 1)[0]

        count += 1

        sentence_batch = list(map(revert_sentense, sentences[batch_start:batch_start + batch_size]))

        maxlen_batch = len(max(sentence_batch, key=len))

        X = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
        X_teacher = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
        y = np.zeros((batch_size, maxlen_batch, len(char_indices)), dtype=np.int32)
        w = np.zeros((batch_size, maxlen_batch), dtype=np.int32)

        for i, sentence in enumerate(sentence_batch):
            corruption_flag = np.random.uniform(0.0, 1.0, len(sentence)) < corruption_pr
            corrupted_sym = np.random.random_integers(low=0, high=len(char_indices)-1, size=len(sentence))
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
                if corruption_flag[t]:
                    X_teacher[i, t, corrupted_sym[t]] = 1
                else:
                    X_teacher[i, t, char_indices[char]] = 1

            for t in range(len(sentence) - 1):
                taget_pos = t + 1
                y[i, t, char_indices[sentence[taget_pos]]] = 1
                w[i, t] = 1

        yield ([X, X_teacher], y, w)


def revert_sentense(sentence):
    return sentence[::-1]