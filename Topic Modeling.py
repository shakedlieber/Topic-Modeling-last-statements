# Run in python console
from wordcloud import WordCloud
import inline
import matplotlib
import nltk;

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import nltk
#nltk.download('stopwords')
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import csv

def main():
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



    sentences = []
    data_words = []
    with open(r'dataset.csv', newline='') as csvfile:
        prisoners = csv.DictReader(csvfile)
        #print(prisoners)
        for prison in prisoners:
            last_statement = prison['Last statement']
            last_statement = last_statement.lower()
            if last_statement.lower() != 'none' and last_statement.lower() != 'declined' and last_statement.lower() != 'decline':
                sentences.append(last_statement)

    print("=========== SENTENCES =========")
    print("=================================")
    print(sentences)

    print ("=================================")
    print("=================================")


    data_words = list(sent_to_words(sentences))
    #print(data_words)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])



    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]


    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]


    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            arr = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            if 'd' in arr:
                arr.remove('d')
            if 's' in arr:
                arr.remove('s')
            texts_out.append(arr)
        return texts_out



    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    #python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])







    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus)


    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha=0.91,
                                               eta=0.91,
                                               per_word_topics=True)



    # Print the Keyword in the topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]



    #======================================
    #======  Compute Coherence Score ======
    #==================================
    def compute_coherence_score():
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)



        def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
            """
            Compute c_v coherence for various number of topics

            Parameters:
            ----------
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

            Returns:
            -------
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
            """
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                # Build LDA model
                model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                            id2word=id2word,
                                                            num_topics=num_topics,
                                                            random_state=100,
                                                            update_every=1,
                                                            chunksize=100,
                                                            passes=10,
                                                            alpha='auto',
                                                            per_word_topics=True)
                # model = gensim.models.wrappers.LdaMallet(lda_model, corpus=corpus, num_topics=num_topics, id2word=id2word)
                model_list.append(model)
                coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                print(coherencemodel.get_coherence())
                coherence_values.append(coherencemodel.get_coherence())

            return model_list, coherence_values

        # Can take a long time to run.
        model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
                                                                start=2, limit=40, step=6)

        # Show graph
        limit = 40;
        start = 2;
        step = 6;
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        print("LDA Gensim Printing")
        # Print the coherence scores
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))



    #============================
    #======  VISUALIZATION ======
    #============================
    def visualize():

        # ===========================================================================
        # Visual WORDCLOUD FOR EACH TOPIC
        for t in range(lda_model.num_topics):
            plt.figure()
            plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(t, 200))))
            plt.axis("off")
            plt.title("Topic #" + str(t))
            plt.show()
        # ===========================================================================

        # ===========================================================================
        # Visual LDA VISUALIZATION
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(vis, 'LDA_Visualization.html')
        # ===========================================================================


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
