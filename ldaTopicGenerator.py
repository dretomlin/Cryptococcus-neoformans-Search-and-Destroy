#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
LDA implementation using gensim
note: inital framework of code was dervived by
priya dwivedi https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb
Collaborators: DeAndre Tomlinson, Emmet Flynn, Paul Brunts
'''


# In[ ]:


import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# In[ ]:

# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'http', 'https', 'high', 'time', 
                    'table', 'read', 'number', 'also', 'show', 'elsevi'
                  ])

# In[ ]:

#Functions:
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def stemWords(texts):
    ps = PorterStemmer()
    stemmed = [[ps.stem(word) for word in text] for text in texts]
    return stemmed


# In[ ]:


#Initializing initial words from documents, positive/negative labels,
#and the filename list for later use in the document
data=[]
testingData=[]
trainingNameList=[]
testingNameList=[]
positiveNameList=[]
trainingLabelList=[]
testingLabelList=[]

#Loads in the names of the files as names,
#and  matches up to the names of the files in the same order
for filename in os.listdir('positiveTextFiles'):
    positiveNameList.append(filename)

# In[ ]:

for filename in os.listdir('trainingFiles'):
    trainingNameList.append(filename)
    filePath='trainingFiles/'+filename
    with open(filePath, 'r') as file:
        x=file.read()
        data.append(x)
        trainingLabelList.append(1 if filename in positiveNameList else 0)

for filename in os.listdir('testingFiles'):
    testingNameList.append(filename)
    filePath='testingFiles/'+filename
    with open(filePath, 'r') as file:
        x=file.read()
        testingData.append(x)
        testingLabelList.append(1 if filename not in positiveNameList else 0)

# In[ ]:

#Input is an empty list, initialized in the previous set
#This function does the data cleaning, remove punctionation
#stop words, lemmatization, and enforces a minimum character limit
#Output is a list of lists, with words from each document as a list
def dataCleaning(data):
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    #remove words less than 4 characters
    data = [re.sub(r'\b\w{1,3}\b', '', sent) for sent in data]
    # Remove Stop Words
    data = remove_stopwords(data)


    # Lemmatization keeping only noun, adj, vb 
    data = lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    #Stem words
    data = stemWords(data)

    data_cleaned = data
    return(data_cleaned)

trainD = dataCleaning(data)

# In[ ]:
#Processes the list of list through gensim function, returns the cleaned data as a list of list again
trainDwords = list(sent_to_words(trainD))

# In[ ]:

#Function to create the appropiate object and dataframe for inital word analysis
def wordsPreAnalysis(data):
    words_list = [j for sub in data for j in sub]
    freqwords = nltk.FreqDist(words_list)
    words_df = pd.DataFrame({'word':list(freqwords.keys()), 'count':list(freqwords.values())})
    return(freqwords, words_df)

#Function to filter out words that cause outliers problems in the data (such as publisher names)
def wordsFilter2(data):
    # using list comprehension + list slicing 
    # Removing element from list of lists 
    wordfilter = ['elsevi']
    for sub in data: 
        sub[:] = [ele for ele in sub if ele not in wordfilter] 
    return(data)


# In[ ]:

freqwords, words_df = wordsPreAnalysis(trainD)
filt_words = wordsFilter2(trainD)
filt_Dwords = list(sent_to_words(filt_words))

# In[ ]:


#Check plot to see if there are words that are erroneous/need cleaning. 
#Up to user discretion
#freqwords.plot(20)
#dflist = words_df.sort_values(by=["count"], ascending=False)
#dflist[dflist["count"]<5]

#Check to see if word was really filtered
#filt_freqwords, filt_wordsDF = wordsPreAnalysis(filt_words)
#filt_freqwords.plot(10)


# In[ ]:

def graphSaver(filename):
    '''
    Function to save graphs to user's desktop
    Saves to folder called "graph_pictures"
    Input is a string with what you want the file to be called
    '''
    
    directory = 'graph_pictures'

    if not os.path.exists(directory):
        os.makedirs(directory)

    savepath = directory+'/'+filename
    plt.savefig(savepath)

# In[ ]:

# Graph that selects top N-most frequent words and charts them 
def freqWordsGraph(df, num, filename):
    d = df.nlargest(columns="count", n = num) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    
    graphSaver(filename)
    plt.show()

# In[ ]:

freqWordsGraph(words_df, 25, filename='unfiltered_words.png')

# In[ ]:


def createDict(words, freq):
    # Create Dictionary
    id2word = corpora.Dictionary(words)
    id2word.filter_extremes(no_below = freq)
    # Term Document Frequency
    corpus = [id2word.doc2bow(word) for word in words]
    return(id2word, corpus)

id2word, corpus = createDict(filt_Dwords, 5)


# In[ ]:


#ldamodel for a single model. Used more as a trial run to test timing for the creation
#for a particular topic number, and vary other parameters. 
#Precursor to the comput_eval_value function below
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=3, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:

lda_model.show_topics()

# In[ ]:

def compute_eval_values(dictionary, corpus, texts, limit, start=2, step=5):
    """
        Compute c_v coherence and perplexity for various number of topics

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
        perplexity_values
    """
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(model.log_perplexity(corpus))

    return model_list, coherence_values, perplexity_values

# In[ ]:

model_list, coherence_values, perplexity_values = compute_eval_values(dictionary=id2word, corpus=corpus, texts=trainDwords, start=2, limit=10, step=1)

# In[ ]:

# Compute Coherence Score of a singular model, precursor to the compute_eval_function
coherence_model_lda = CoherenceModel(model=model_list[2], texts=filt_Dwords, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:

#These will save the model to the desktop for later use, if necesary
lda_model.save('cneoformans_3kmodel')
model_list[2].save('cneoformans_4kmodel')


# In[ ]:

def evalsGrapher(coherences, perplexities, limit, start, step, filename):
    '''
    Function that graphs the coherences and perplexities of the previous function
    '''
    #clear other plot
    #plt.clf()
    #plt.cla()
    
    x = range(start, limit, step)
    for m, cv, pv in zip(x, coherences, perplexities):
        print("Num Topics =", m, "has a Coherence Value of ", round(cv, 4), " and has Perplexity Value of", round(np.exp(-1. * pv), 4))
    
    # Graph coherence and perplexity over kTopics
    fig, axs = plt.subplots(2,1, sharex=True)
    plt.xlabel("Num Topics")
    axs[0].plot(x, coherence_values, color='green')
    axs[0].set_ylabel("Coherence score")
    axs[0].legend(title ="Coherence", loc='best')

    axs[1].plot(x, perplexity_values, color='blue')
    axs[1].set_ylabel("Perplexity score")
    axs[1].legend(title="Perplexity", loc='best')
    
    graphSaver(filename)
    #plt.show()

# In[ ]:

evalsGrapher(coherence_values, perplexity_values, 10, 2, 1, filename='coh-perp-plots.png')

# In[ ]:

def wordCountImpMaker(model, data, filename):
    '''
    Function to display word count and weight of each word for each topic
    Good visual indicator for analysis
    Input is the model, 2d list of words, and a filename to save to
    '''
    topics = model.show_topics(formatted=False)
    data_flat = [w for w_list in data for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(1, 3, figsize=(14,5), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05) 
    
    graphSaver(filename)
    #plt.show()

# In[ ]:

from collections import Counter
import matplotlib.colors as mcolors
wordCountImpMaker(model_list[2], filt_Dwords, filename='word-importance-graph.png')

# In[ ]:


import pyLDAvis
import pyLDAvis.gensim
#Visualizaton of topics in an html format for more analysis purposes
lda_display = pyLDAvis.gensim.prepare(model_list[2], corpus, id2word, sort_topics=False)
#Visualization will appear after this call
pyLDAvis.display(lda_display)

# In[ ]:

#You can save the html with the call below
#pyLDAvis.save_html(lda_display, 'lda_4k_vis.html')


# #Processing for testing corpus

# In[ ]:

testD = dataCleaning(testingData)
testDwords = list(sent_to_words(testD))
freqTestwords, wordsTestDf = wordsPreAnalysis(testDwords)

# In[ ]:

freqTestwords.plot(20)
dflist = wordsTestDf.sort_values(by=["count"], ascending=False)
dflist[dflist["count"] < 5].head(30)

# In[ ]:

freqWordsGraph(wordsTestDf, 25, filename='unfiltered_test_words.png')

# In[ ]:

id2wordTest, corpusTest = createDict(testDwords, 5)

# In[ ]:


def format_topics_sentences(model, corpus, texts, kTopics):
    '''
    Function for grabbing topics and probabilities from lda model
    First part in process to export values to csvs
    Inputs are the model, text corpus, 2d list of words, and number of topics
    '''
    # Initialize dataframe to return and empty list
    sent_topics_df = pd.DataFrame()
    series_list=[]
    
    for i in range(kTopics):
        series_list.append(0)
    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            wp = model.show_topic(topic_num)
            series_list[topic_num]=prop_topic
            #sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
        sent_topics_df=sent_topics_df.append(pd.Series([series_list[i] for i in range(len(series_list))]), ignore_index=True)
    
    #sent_topics_df.columns = ['Topic_0', 'Topic_1', 'Topic_2']
    # Add original text to the end of the output
    sent_topics_df['contents'] = texts
    return(sent_topics_df)


# In[ ]:


def topicProbs2csv(df, labelList, nameList, kTopics, filename):
    '''
    Function to send topic probabilities to csv file
    '''
    
    df['label'] = labelList
    df['name'] = nameList
    
    # Show
    names = ['topic_{}'.format(i) for i in range(len(df.columns.values.tolist()) - 3)]+['contents', 'label', 'name'] 
    df.columns = names
    
    # Below we pull into the csv only the columns associated with the topics
    # and the labels. (this is the first kTopics columns and the second column from the end
    csv = df[names[:kTopics]+[names[-2]]].copy().to_csv(filename, index=False)
    csv_printout = df[names[:kTopics]+[names[-2]]].copy().to_csv(index=False)
    print(csv_printout)


# In[ ]:


df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, filt_Dwords, 3)


# In[ ]:


df_testing_topic_keywords = format_topics_sentences(lda_model, corpusTest, testDwords, 3)


# In[ ]:


testCsvfile = 'ltg_3k_testing.csv'
testCsv = topicProbs2csv(df_testing_topic_keywords, testingLabelList, testingNameList, 3, testCsvfile)


# In[ ]:


trainCsvfile = 'ltg_3k_training.csv'
trainCsv = topicProbs2csv(df_topic_sents_keywords, trainingLabelList, trainingNameList, 3, trainCsvfile)


# In[ ]:


df_4test_topkeywords = format_topics_sentences(model_list[2], corpusTest, testDwords, 4)
df_4train_topkeywords = format_topics_sentences(model_list[2], corpus, filt_Dwords, 4)


# In[ ]:


testCsvfile = 'ltg_4k_testing.csv'
testCsv = topicProbs2csv(df_4test_topkeywords, testingLabelList, testingNameList, 4, testCsvfile)
trainCsvfile = 'ltg_4k_training.csv'
trainCsv = topicProbs2csv(df_4train_topkeywords, trainingLabelList, trainingNameList, 4, trainCsvfile)


# In[ ]: 
'''
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

#BIGRAMS AND TRIGRAMS DO NOT WORK. UNIGRAM ONLY.
#not entirely sure about nthreshold
# Build the bigram and trigram models
#nthreshold=100
#bigram = gensim.models.Phrases(data_words, min_count=5, threshold=nthreshold) # higher threshold fewer phrases.
#trigram = gensim.models.Phrases(bigram[data_words], threshold=nthreshold)  

# Faster way to get a sentence clubbed as a trigram/bigram
#bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)
'''

# In[ ]:


'''
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )
trainDwords_vec = vectorizer.fit_transform(filt_words)

GRID SEARCH TAKES TOO LONG, BUT WOULD BE GREAT TO IMPLEMENT
# Define Search Param
search_params = {'n_components': [1,2,3]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

model.fit(trainDwords_vec)

# Best Model
#best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
'''

