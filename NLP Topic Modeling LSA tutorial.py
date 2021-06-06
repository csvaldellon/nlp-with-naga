# from https://www.youtube.com/watch?v=TKjjlp5_r7o
# from https://stackoverflow.com/questions/66759852/no-module-named-pyldavis
# from https://raw.githubusercontent.com/wjbmattingly/topic_modeling_textbook/main/data/ushmm_dn.json
# from https://github.com/github/hub/issues/2711
# from https://stackoverflow.com/questions/43317056/pyldavis-unable-to-view-the-graph
# from https://www.youtube.com/watch?v=UEn3xHNBXJU&list=PL2VXyKi-KpYttggRATQVmgFcQst3z6OlX&index=11
# from https://www.youtube.com/watch?v=D2V1okCEsiE
# from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#9createbigramandtrigrammodels
# https://stackoverflow.com/questions/24688116/how-to-filter-out-words-with-low-tf-idf-in-a-corpus-with-gensim/35951190
import nltk
import numpy as np
import json
import glob

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

import spacy
import en_core_web_sm
from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim_models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


stopwords = stopwords.words("english")
# print(stopwords)
data = load_data("D:/ushmm_dn.json")["texts"]
print(data[0][0:90])


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out


lemmatized_texts = lemmatization(data)
print(lemmatized_texts[0][0:90])


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final


data_words = gen_words(lemmatized_texts)
print(data_words[0][0:20])

bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigrams_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)


def make_bigrams(texts):
    return [bigram[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]


data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)
print(data_bigrams_trigrams[0])

id2word = corpora.Dictionary(data_bigrams_trigrams)
texts = data_bigrams_trigrams
corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus, id2word=id2word)
low_value = 0.03
words = []
words_missing_in_tfidf = []

for i in range(0, len(corpus)):
    bow = corpus[i]
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow


def no_bi_tri():
    id2word = corpora.Dictionary(data_words)
    corpus = []
    for text in data_words:
        new = id2word.doc2bow(text)
        corpus.append(new)

    # print(corpus[0][0:20])
    word = id2word[[0][:1][0]]
    # print(word)


lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=7,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto"
)

# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
pyLDAvis.save_html(vis, 'lda3.html')

