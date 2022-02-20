import nltk
import numpy as np
# import wordcloud
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(s):
    return nltk.word_tokenize(s)

def stem(w):
    return stemmer.stem(w.lower())

def bag_of_words(tokenized_s, all_words):
    tokenized_s = [stem(w) for w in tokenized_s]
    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_s:
            bag[idx] = 1.0
    return bag

