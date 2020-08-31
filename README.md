# Topic-Modeling-last-statements
LDA Topic modeling in death row prisoners last statements
Scraping data + topic modeling and visualization

1) scraping data:

Inorder to create our dataset we used BeautifulSoup python library to scrape data from 2 websites:
http://www.clarkprosecutor.org/html/death/usexecute.htm \
https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html \
For each prisoner a Prisoner object was created including all his details from both website (age, race, sex, d.o.b, last statement and more) and performing cross-checking on the information.

An array of Prisoners object was created including all the details needed.
Using CSV python library we created a CSV file which hold all the collected data.

Helpful websites:  \
https://www.dataquest.io/blog/web-scraping-beautifulsoup/ \

2) Topic Modeling

For performing the topic modeling we first needed to clean our data for optimal results.
The topic modeling based mainly on the Gensim python package.
1. First we opened the CSV file we created and generated an array of last statements excluding all prisoners which had no last statement or declined.
2. Tokenizing each last statement using gensim.utils.simple_preprocess.
3. Created bigrams and trigrams from the tokens using gensim.models.phrases.Phraser
4. Removed all the stop words from our data, We used NLTK stopwords library.
5. Made bigrams after cleaning the stopwords
6. Using Spacy package we lemmatized the bigrams for better results.
7. using gensim.corpora.Dictionary created a dictionay fillled with id's and words.
8. To create or final corpus we used doc2bow for creating our bag of words.
9. Then we applied gensim.models.ldamodel for the creating our topic modeling algorithm.

Helpful websites:  \
https://www.tutorialspoint.com/gensim/gensim_creating_lda_topic_model.htm \
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/ \
https://medium.com/@krunal18/topic-modeling-with-latent-dirichlet-allocation-lda-decomposition-scikit-learn-and-wordcloud-1ff0b8e8a8eb \

3) Results and Visualiztion

For validating our results we calculated coherence score using gensim.models.CoherenceModel
And for visualization we used WordCloud package and pyLDAvis.

Helpful websites: \
Word clouds using https://stackoverflow.com/questions/40278023/how-do-i-print-lda-topic-model-and-the-word-cloud-of-each-of-the-topics \
Graphs were created using https://plotly.com \

