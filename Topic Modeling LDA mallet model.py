# Run in python console
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
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            print(coherencemodel.get_coherence())
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values



    def compute_coherence_values(corpus, dictionary, k, a, b):

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=10,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               alpha=a,
                                               eta=b,
                                               per_word_topics=True)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
                                             coherence='c_v')

        return coherence_model_lda.get_coherence()



    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

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
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out





    sentences = []
    data_words = []
    with open(r'dataset.csv', newline='') as csvfile:
        prisoners = csv.DictReader(csvfile)
        #print(prisoners)
        for prison in prisoners:
            last_statement = prison['Last statement']
            last_statement = last_statement.lower()
            if last_statement != 'none' and last_statement != 'declined' and last_statement != 'decline':
                sentences.append(last_statement)

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



    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    #python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # # View
    # print(corpus[:1])
    #
    # #
    #
    # # Build LDA model
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                            id2word=id2word,
    #                                            num_topics=6,
    #                                            random_state=100,
    #                                            update_every=1,
    #                                            chunksize=100,
    #                                            passes=10,
    #                                            alpha='auto',
    #                                            per_word_topics=True)
    #
    #
    #
    # # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
    #
    #
    #
    #
    # # Compute Perplexity
    # print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    # pyLDAvis.save_html(vis, 'LDA_Visualization.html')




    import os

    from gensim.models.wrappers import LdaMallet

    os.environ['MALLET_HOME'] = r'C:\Users\lieber\Desktop\project\project\project\mallet-2.0.8'
    mallet_path = r'C:\Users\lieber\Desktop\project\project\project\mallet-2.0.8\bin\mallet'

    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    mallet_path = r'C:\Users\lieber\Desktop\project\project\project\mallet-2.0.8\bin\mallet'  # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=5, alpha=0.91,  id2word=id2word)

    # Show Topics
    pprint(ldamallet.show_topics(formatted=False))

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word,
                                               coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)



    # # Can take a long time to run.
    # model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
    #                                                         start=2, limit=40, step=6)
    #
    # # Show graph
    # limit = 40;
    # start = 2;
    # step = 6;
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()
    #
    #


    # print("LDA MALLET Printing")
    # # Print the coherence scores
    # for m, cv in zip(x, coherence_values):
    #     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    # supporting function


    #
    # print('==============================================================')
    # print('=================  GETTING THE RIGHT PARAMETERS  ==============================')
    # import numpy as np
    # import tqdm
    #
    # grid = {}
    # grid['Validation_Set'] = {}
    # # Topics range
    # min_topics = 2
    # max_topics = 11
    # step_size = 1
    # topics_range = range(min_topics, max_topics, step_size)
    # # Alpha parameter
    # alpha = list(np.arange(0.01, 1, 0.3))
    # alpha.append('symmetric')
    # alpha.append('asymmetric')
    # # Beta parameter
    # beta = list(np.arange(0.01, 1, 0.3))
    # beta.append('symmetric')
    # # Validation sets
    # num_of_docs = len(corpus)
    # print(num_of_docs)
    # corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
    #     # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
    #     gensim.utils.ClippedCorpus(corpus,591),
    #     corpus]
    # corpus_title = ['75% Corpus', '100% Corpus']
    # model_results = {'Validation_Set': [],
    #                  'Topics': [],
    #                  'Alpha': [],
    #                  'Beta': [],
    #                  'Coherence': []
    #                  }
    # # Can take a long time to run
    # if 1 == 1:
    #     pbar = tqdm.tqdm(total=540)
    #
    #     # iterate through validation corpuses
    #     for i in range(len(corpus_sets)):
    #         # iterate through number of topics
    #         for k in topics_range:
    #             # iterate through alpha values
    #             for a in alpha:
    #                 # iterare through beta values
    #                 for b in beta:
    #                     # get the coherence score for the given parameters
    #                     cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b)
    #                     # Save the model results
    #                     model_results['Validation_Set'].append(corpus_title[i])
    #                     model_results['Topics'].append(k)
    #                     model_results['Alpha'].append(a)
    #                     model_results['Beta'].append(b)
    #                     model_results['Coherence'].append(cv)
    #
    #                     pbar.update(1)
    #     pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    #     pbar.close()



# __name__
if __name__ == "__main__":
    main()
