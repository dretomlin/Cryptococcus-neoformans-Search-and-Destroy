## C. Neoformans Identification using Academic Paper Texts

### by DeAndre Tomlinson, Pfrender Lab 
#### Contributors: Emmett Flynn, and Paul Brunts

### Description
This repository contains code for processing and classifying papers along the
lines of whether the underlying studies found evidence of the C. Neoformans
bacteria. These classifications are made by examining the text of the paper,
not by examining the underlying dataset, which is computationally and manually
difficult.

### Pipeline

This repository contains the underlying papers to do appropriate classification.
To run the classification for yourself, first download the repo and run:

`python3 ldaTopicGenerator.py`

This will output two sets of topic weights to the command line in csv form. 
To form your own training an testing files, copy the first of these csv outputs
(it should have roughly 85 documents, or lines) to your training file, and the 
second (roughly 30 documents or lines) to your testing file.

To avoid this step, you can use one of the pre generated datasets. We recommend
that you use the data set in `ltg_*_training.csv` and 
`ltg_*_testing_.csv`. 

Then choose your classifier. Below are mappings of classifiers to the filename
associated with that classifier.

`python3 CLASSIFIER PATH_TO_TRAINING_CSV PATH_TO_TESTING_CSV`

For instance, to use the random forest classifier to train and predict using
the csvs, the command would be:

`python3 rforest_4k.py ltg_4k_training.csv ltg_4k_testing_.csv`

Package dependencies (with anacondaa3 as a base environment):
<ul>
<li>gensim</li><li>spacy</li><li>seaborn</li><li>scikit-plt</li><li>pyLDAvis</li>


