import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def data_prep():
    df = pd.read_csv('bbc.csv', encoding='latin1')  # read dataset from bbc.csv

    ##Dropping irrelevant variable. 'TEXT' variable left
    df.drop(['URI', 'NAME', 'FILTERD', 'LANGUAGE', 'CREATED', 'ACCESSED', 'MODIFIED', 'TRUNCATED', 'OMITTED', 'EXTENSION', 'SIZE', 'FILTEREDSIZE'], axis=1, inplace=True)

    # print out the first 200 characters of the first row of text column
    print(df.get_value(index=0, col='TEXT')[:200])

    # average length of text column
    print(df['TEXT'].apply(lambda x: len(x)).mean())

    return df



# list of unnecessary punctuation
def lemmatize(token, tag):

    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)

    df_stop = pd.read_json('bbc.csv', encoding='latin1')
    stopwords = set(df_stop['Term']).union(set(sw.words('english')))

    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)

    return lemmatizer.lemmatize(token, tag)


def cab_tokenizer(document):
    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)

    df_stop = pd.read_json('bbc.csv', encoding='latin1')
    stopwords = set(df_stop['Term']).union(set(sw.words('english')))
    # initialize token list
    tokens = []

    # split the document into sentences
    for sent in sent_tokenize(document):
        # split the document into tokens and then create part of speech tag for each token
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # preprocess and remove unnecessary characters
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # If stopword, ignore token and continue
            if token in stopwords:
                continue

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            # Lemmatize the token and add back to the token
            lemma = lemmatize(token, tag)
            tokens.append(lemma)

    return tokens
def vectorize():
    df = data_prep()
    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)

    df_stop = pd.read_json('bbc.csv', encoding='latin1')
    stopwords = set(df_stop['Term']).union(set(sw.words('english')))
    # tf idf vectoriser
    tfidf_vec = TfidfVectorizer(tokenizer=cab_tokenizer, ngram_range=(1, 2))
    X = tfidf_vec.fit_transform(df['TEXT'])

    # see the number of unique tokens produced by the vectorizer. Lots of them...
    print(len(tfidf_vec.get_feature_names()))