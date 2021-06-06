# from https://www.youtube.com/watch?v=X2vAabgKiuM&t=1114s
# from https://python.gotrained.com/frequency-distribution-in-nltk/
import nltk
import nltk.corpus
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.tokenize import blankline_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, wordnet, WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk import ne_chunk


def preliminaries():
    # nltk.download("brown")
    # nltk.download("gutenberg")
    # nltk.download("punkt")
    # print(brown.words())
    # print(nltk.corpus.gutenberg.fileids())
    hamlet = nltk.corpus.gutenberg.words("shakespeare-hamlet.txt")
    # print(hamlet)

    # for word in hamlet[:500]:
    # print(word, sep=" ", end=" ")


def tokenization():
    AI = "I am the best data scientist in the world! I will conquer the world! I am INSERT NAME HERE"
    # print(type(AI))
    AI_tokens = word_tokenize(AI)
    # print(AI_tokens)
    # print(len(AI_tokens))

    fdist = FreqDist()
    for word in AI_tokens:
        fdist[word.lower()] += 1

    # print(fdist["the"])
    # fdist.plot()

    # print(len(fdist))
    fdist_top10 = fdist.most_common(10)
    # print(fdist_top10)

    AI_blank = blankline_tokenize(AI)
    # print(len(AI_blank))

    string = "This video will provide you with a comprehensive and detailed knowledge of Natural Language Processing, " \
             "popularly known as NLP. You will also learn about the different steps involved in processing the human " \
             "language like Tokenization, Stemming, Lemmatization and more. Python, NLTK, & Jupyter Notebook are used " \
             "to " \
             "demonstrate the concepts."

    string_tokens = word_tokenize(string)
    fdist_string = FreqDist()
    for word in string_tokens:
        fdist_string[word.lower()] += 1

    # print(fdist["the"])
    # fdist_string.plot()

    string_bigrams = list(nltk.bigrams(string_tokens))
    string_trigrams = list(nltk.trigrams(string_tokens))
    string_ngrams = list(nltk.ngrams(string_tokens, 5))
    # print(string_bigrams)
    # print(string_trigrams)
    # print(string_ngrams)


def stemming():
    pst = PorterStemmer()
    print(pst.stem("having"))
    words_to_stem = ["give", "giving", "given", "gave"]
    for words in words_to_stem:
        print(words, ":", pst.stem(words))

    lst = LancasterStemmer()
    print(lst.stem("having"))
    words_to_stem = ["give", "giving", "given", "gave"]
    for words in words_to_stem:
        print(words, ":", lst.stem(words))

    sbst = SnowballStemmer("english")
    print(sbst.stem("having"))
    words_to_stem = ["give", "giving", "given", "gave"]
    for words in words_to_stem:
        print(words, ":", sbst.stem(words))


def lemmatization():
    # nltk.download('wordnet')
    word_lem = WordNetLemmatizer()
    print(word_lem.lemmatize("corpora"))
    words_to_stem = ["give", "giving", "given", "gave"]
    for words in words_to_stem:
        print(words, ":", word_lem.lemmatize(words))


def stop_words():
    # nltk.download('stopwords')
    print(stopwords.words("english"))
    print(len(stopwords.words("english")))
    fdist = FreqDist()
    string = "This video will provide you with a comprehensive and detailed knowledge of Natural Language Processing, " \
             "popularly known as NLP. You will also learn about the different steps involved in processing the human " \
             "language like Tokenization, Stemming, Lemmatization and more. Python, NLTK, & Jupyter Notebook are used " \
             "to " \
             "demonstrate the concepts."
    string_tokens = word_tokenize(string)
    for word in string_tokens:
        fdist[word.lower()] += 1

    # print(len(fdist))
    fdist_top10 = fdist.most_common(10)
    print(fdist_top10)
    punctuation = re.compile(r'[-.?!,:;()|0-9]')
    post_punctuation = []
    for words in string_tokens:
        word = punctuation.sub("", words)
        if len(word) > 0:
            post_punctuation.append(word)
    print(post_punctuation)
    print(len(post_punctuation))


def parts_of_speech_tagging():
    # nltk.download('averaged_perceptron_tagger')
    sent = "INSERT NAME HERE is a natural when it comes to data science and analytics!"
    sent_tokens = word_tokenize(sent)
    for token in sent_tokens:
        print(nltk.pos_tag([token]))

    sent2 = "INSERT NAME HERE is performing a delicious time-series forecasting!"
    sent2_tokens = word_tokenize(sent2)
    for token in sent2_tokens:
        print(nltk.pos_tag([token]))


def named_entity_recognition():
    # NER Entities List: Facility, Location, Organization, Person, Geo-Socio-Political Group, Geo-Political Entity,
    # Facility
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    NE_sent = "Val is part of M1Classifiers."
    NE_tokens = word_tokenize(NE_sent)
    NE_tags = nltk.pos_tag(NE_tokens)
    NE_NER = ne_chunk(NE_tags)
    print(NE_NER)


def chunking():
    new = "The big cat ate the little mouse who was after the fresh cheese."
    new_tokens = nltk.pos_tag(word_tokenize(new))
    print(new_tokens)

    grammar_np = r"NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar_np)
    chunk_result = chunk_parser.parse(new_tokens)
    print(chunk_result)

    new = "The new scientist performed an excellent analysis on the big dataset"
    new_tokens = nltk.pos_tag(word_tokenize(new))
    print(new_tokens)

    grammar_np = r"NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar_np)
    chunk_result = chunk_parser.parse(new_tokens)
    print(chunk_result)

