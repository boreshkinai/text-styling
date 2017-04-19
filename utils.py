import numpy as np
import re


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