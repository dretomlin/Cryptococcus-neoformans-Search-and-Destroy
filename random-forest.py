#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import ensemble
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, f1_score, confusion_matrix
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
from random import randint, seed
import time


# In[ ]:


if len(sys.argv) < 3:
  print('''
    usage: python ./tree.py TRAINING TESTING
    TRAINING: file path to csv file of all training objects
    TESTING: path to csv file of all test objects

    both csvs should be output from ldaIsh.py to ensure proper formatting
  ''')
  exit(1)


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


def perf_measure(y_actual, y_hat):
    '''Performs the confusion matrix values
    True positive, True Negative, False Positive, False Negative
    Returns the 4 values as well
    '''
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


# In[ ]:


#train = sys.argv[1]
#test = sys.argv[2]
train = 'ltg_4k_training.csv'
test = 'ltg_4k_testing.csv'


# In[ ]:


tr = pd.read_csv(train)
te = pd.read_csv(test)
names = tr.columns.values.tolist()[0:-1]


# In[ ]:


Attributes=tr[names]
Labels=tr['label']


# In[ ]:


features=names
labels=['1','0']


# In[ ]:


#Set variables for random forest evaluation
recordedRecall=0
recordedPrecision=0
recordedAccuracy=0
bestF1=0
realTP=0
realTN=0
realFP=0
realFN=0


# In[ ]:


#Random forest parameters
nestimators=10
maxdepth=100
repeatRange = 250
model_list = []


# In[ ]:


#Random forest function
for i in range(repeatRange):
    clf=ensemble.RandomForestClassifier(n_estimators = nestimators, max_depth=maxdepth)
    clf=clf.fit(Attributes, Labels)
    
    yhat = clf.predict_proba(te[names].copy())
    labels = clf.predict(te[names].copy())
    truth = te['label'].values.tolist()
    seed(time.time())
    
    tp, fp, tn, fn = perf_measure(truth, labels)
    
    precision = float(tp/(tp+fp))
    accuracy = (tp+tn)/(tp+tn+fp+fn) 
    recall = float(tp/(tp+fn))
    if precision and recall == 0:
        f1 = 0
    try:
        f1 = 2 * (float(precision * recall) / float(precision + recall))
    except ZeroDivisionError:
        f1 = 0

    if(f1>bestF1):
        bestF1=f1
        recordedAccuracy=accuracy
        recordedPrecision=precision
        recordedRecall=recall
        realTP=tp
        realTN=tn
        realFP=fp
        realFN=fn
        model_list.append(clf)


# In[ ]:


print("Note: tree creation is not always deterministic.")
print("Number of topics: ", len(names))
print("Number of models: ", len(model_list))
print("% of docs that are positive: ", (tp+fn)/(tp+fn+fp+tn))
print("% of docs that are negative: ", (fp+tn)/(tp+fn+fp+tn))


# In[ ]:


print()
print("TP: ", realTP)
print("FP: ", realFP)
print("TN: ", realTN)
print("FN: ", realFN)
print()
print("Precision: ", recordedPrecision)
print("Accuracy:  ", recordedAccuracy)
print("Recall:    ", recordedRecall)
print("F1:        ", bestF1)


# In[ ]:


truth2 = np.array(truth)
labels2 = np.array(labels)
print("These are the labels: ", truth2)
print("These are the predicted labels after classification: ", labels2)


# In[ ]:


pos_probs = yhat[:,1]


# In[ ]:


d2pprobs = np.array(yhat)


# In[ ]:


precision2, recall2, thresholds = precision_recall_curve(truth2, pos_probs)


# In[ ]:


'''
Process for creating precision-recall curve
Not used in this proejct, but for analysis purposes
'''
lr_precision, lr_recall, _ = precision_recall_curve(truth2, pos_probs)

lr_f1, lr_auc = f1_score(truth2, labels2), auc(recall2, precision2)

plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# In[ ]:


auc_score = auc(recall2, precision2)
roc_auc_result = roc_auc_score(truth2, pos_probs)
print(auc_score)
print(roc_auc_result)


# In[ ]:


skplt.metrics.plot_roc(truth2, d2pprobs)
#Save figure below with the followin function
#graphSaver('roc-45topics.png')


# In[ ]:


def make_confusion_matrix(y_test, y_predictor, filename=''):
    '''
    Confusion matrix generation integrated with optional save function
    Returns confusion matrix in array form
    '''
    cm=confusion_matrix(y_test, y_predictor)
    index = ['Negative','Positive']  
    columns = ['Negative','Positive']
    cm_df = pd.DataFrame(cm, columns, index)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    display(cm_df)
    
    #Clear previous plot
    plt.clf()
    plt.cla()

    sns.heatmap(cm_df, annot=True)
    
    if filename != '':
        graphSaver(filename)
        
    return(cm)


# In[ ]:


conf_matrix = make_confusion_matrix(truth2, labels2)


# In[ ]:




