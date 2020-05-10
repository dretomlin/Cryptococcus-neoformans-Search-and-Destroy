#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import ensemble
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt
import graphviz
from random import randint
from random import seed
import time

if len(sys.argv) < 3:
  print('''
    usage: python ./tree.py TRAINING TESTING
    TRAINING: file path to csv file of all training objects
    TESTING: path to csv file of all test objects

    both csvs should be output from ldaIsh.py to ensure proper formatting
  ''')
  exit(1)


train = sys.argv[1]
test = sys.argv[2]

tr = pd.read_csv(train)
te = pd.read_csv(test)

names = tr.columns.values.tolist()[0:-1]

Attributes=tr[names]
Labels=tr['label']

nestimators=10
features=names
labels=['1','0']

recordedRecall=0
recordedPrecision=0
recordedAccuracy=0
bestF1=0
maxdepth=100
realTP=0
realTN=0
realFP=0
realFN=0

for i in range(250):
    clf=ensemble.RandomForestClassifier(n_estimators = nestimators, max_depth=maxdepth)
    clf=clf.fit(Attributes, Labels)
    
    yhat = clf.predict_proba(te[names].copy())
    labels = clf.predict(te[names].copy())
    truth = te['label'].values.tolist()
    seed(time.time())
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    
    for i in range(len(labels)):
      if labels[i] == truth[i]:
        if labels[i] == 1:
          tp = tp + 1
        else:
          tn = tn + 1
      else:
        if labels[i] == 1:
          fp = fp + 1
        else:
          fn = fn + 1
    
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
    



#plot_tree(clf)
#dot_data=tree.export_graphviz(clf, out_file=None, feature_names=features, 
#        class_names=labels)
#graph=graphviz.Source(dot_data)
#graph.render("tree")


#labels = [randint(0,1) for i in range(len(truth))]

print("Note: tree creation is not always deterministic.")
print("Number of topics: ", len(names))
print("% of docs that are positive: ", (tp+fn)/(tp+fn+fp+tn))
print("% of docs that are negative: ", (fp+tn)/(tp+fn+fp+tn))

print()
print("tp: ", realTP)
print("fp: ", realFP)
print("tn: ", realTN)
print("fn: ", realFN)
print()
print("precision: ", recordedPrecision)
print("accuracy:  ", recordedAccuracy)
print("recall:    ", recordedRecall)
print("F1:        ", bestF1)

truth2 = np.array(truth)
labels2 = np.array(labels)

pos_probs = yhat[:,1]

precision2, recall2, thresholds = precision_recall_curve(truth2, pos_probs)
print("These are the labels: ", labels)
print("These are the predicted labels after classification: ", truth2)
#print(precision2)
#print(recall2)
#print(thresholds)
#print(yhat)
#disp = plot_precision_recall_curve(clf, Attribute, Label)
"""
fig, ax = plt.subplots()

ax.plot(recall2, precision2, marker='.', label = 'Precision-Recall')
ax.set_title('Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
fig.savefig('p-recall.png')
plt.close(fig)
"""
#fig1, ax1 =plt.subplots()
#fpr, tpr, _ = roc_curve(truth2, pos_probs)

"""

print(fpr, tpr)
ax1.plot(fpr, tpr, marker=".", label='ROC AUC')
ax1.set_title('ROC-AUC Curve')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
fig1.savefig("roc_r.png")
plt.close(fig1)
"""

auc_score = auc(recall2, precision2)
roc_auc_result = roc_auc_score(truth2, pos_probs)
print(auc_score)
print(roc_auc_result)

#pprobs = np.array(yhat[:,1])
#d2pprobs = np.column_stack((1-pprobs, pprobs))
#real_posprobs = np.concatenate((1-pprobs, pprobs),axis=1)
d2pprobs = np.array(yhat)
skplt.metrics.plot_roc(truth2, d2pprobs)
plt.savefig('roc-3topics.png')

