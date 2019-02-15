#############################################################
## ASSIGNMENT 1 CODE SKELETON
## RELEASED: 2/6/2019
## DUE: 2/15/2019
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_validate
from nltk.corpus import wordnet
import numpy as np
import gzip
import syllables

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    n = len(y_pred)
    fps, tns, tps, fns = 0,0,0,0
    for i in range(n):
        if y_pred[i] > y_true[i]:
            fps+=1
            continue;
        if y_pred[i] < y_true[i]:
            fns+=1
            continue
        if (y_pred[i] + y_true[i]) == 0:
            tns+=1
            continue
        if (y_pred[i] + y_true[i]) > 1:
            tps+=1
            continue
    precision = tps/(tps + fps)
    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    n = len(y_pred)
    fps, tns, tps, fns = 0,0,0,0
    for i in range(n):
        if y_pred[i] > y_true[i]:
            fps+=1
            continue;
        if y_pred[i] < y_true[i]:
            fns+=1
            continue
        if (y_pred[i] + y_true[i]) == 0:
            tns+=1
            continue
        if (y_pred[i] + y_true[i]) > 1:
            tps+=1
            continue    
    recall = tps/(fns + tps)
    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    recall = get_recall(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    fscore = 2/(1/precision + 1/recall)
    return fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

def load_test(test):
    words = []
    with open(test, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
            i += 1
    return words

### 2.1: A very simple baseline
# predict fscore = 0.5895627644569816
## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1 for i in range(len(words))] 

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    #preds = np.ones_like(labels)
    preds = all_complex_feature(words)
    precision = get_precision(preds,labels)
    recall = get_recall(preds, labels)
    fscore = get_fscore(preds, labels)
    performance = [precision, recall, fscore]

    return performance


### 2.2: Word length thresholding
# predict fscore = 0.712598 
## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    return [1 if len(i)>= threshold else 0 for i in words]

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    w_t, l_t = load_file(training_file)
    w_d, l_d = load_file(development_file)
    fscore_max_t = 0
    threshold = 0
    for i in range(20):
       pred = length_threshold_feature(w_t,i)
       fscore_t = get_fscore(pred,l_t)
       if(fscore_t > fscore_max_t):
           fscore_max_t = fscore_t
           threshold = i
    # threshold here is seven!
    pred_d = length_threshold_feature(w_d, threshold)
    pred_t = length_threshold_feature(w_t, threshold) 
    tprecision,trecall,tfscore = get_precision(pred_t,l_t),get_recall(pred_t,l_t),get_fscore(pred_t,l_t)
    dprecision,drecall,dfscore = get_precision(pred_d,l_d),get_recall(pred_d,l_d),get_fscore(pred_d,l_d)
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt', encoding="utf8") as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

# helper function for Q2_c
def find_max_frequency(counts, words):
    max_freq = 0
    for word in words:
        if word in counts:
            freq = counts.get(word)
            max_freq = freq if freq > max_freq else max_freq
    return max_freq

def find_min_frequency(counts, words):
    min_freq = 1000
    for word in words:
        if word in counts:
            freq = counts.get(word)
            min_freq = freq if freq < min_freq else min_freq
    return min_freq
## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    result = []
    for i in range(len(words)):
        wordFreq = counts.get(words[i].lower())
        if wordFreq is None:
            result.append(1)
        elif wordFreq <= threshold:
            result.append(1)
        elif wordFreq > threshold:
            result.append(0)
    return  result
def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    max_in_t = find_max_frequency(counts, twords)
    max_in_d = find_max_frequency(counts, dwords)
    min_in_t = find_min_frequency(counts, twords)
    fscore_max = 0
    freq = 0
    for i in range(min_in_t, max_in_t + 1):
        preds = frequency_threshold_feature(twords,i, counts)
        fscore = get_fscore(preds, tlabels)
        fscore_max,freq = (fscore,i) if fscore > fscore_max else (fscore_max,freq)

    print("best freqency %d"% freq)
    dpred = frequency_threshold_feature(dwords, freq, counts)
    tpred = frequency_threshold_feature(twords, freq,counts)

    dprecision,drecall,dfscore = get_precision(dpred,dlabels),get_recall(dpred,dlabels),get_fscore(dpred,dlabels)
    tprecision,trecall,tfscore = get_precision(tpred,tlabels),get_recall(tpred,tlabels),get_fscore(tpred,tlabels)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes
# predict fscore : 0.6304176516942475        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE

    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    tlength = [ 0 if len(word) == None else len(word)for word in twords]
    dlength = [ 0 if len(word) == None else len(word)for word in dwords]
    tfreq = [ 0 if counts.get(w) == None else counts.get(w) for w in twords]
    dfreq = [ 0 if counts.get(w) == None else counts.get(w) for w in dwords]
    
    tl = np.array(tlength)
    meanl = np.mean(tl)
    stdl  = np.std(tl)
    tl_scale = [(l - meanl)/stdl for l in tl]
    dl = np.array(dlength)
    dl_scale = [(l - meanl)/stdl for l in dl]
    tf = np.array(tfreq)
    meanf = np.mean(tf)
    stdf = np.std(tf)
    tf_scale = [(f - meanf)/stdf for f in tf]
    df = np.array(dfreq)
    df_scale = [(f - meanf)/stdf for f in df]
    X_train = np.matrix([tl_scale,tf_scale]).T
    X_test = np.matrix([dl_scale,df_scale]).T
    Y = tlabels
    clf = GaussianNB()
    clf.fit(X_train, Y)
    dpred = clf.predict(X_test)
    tpred = clf.predict(X_train)
    tprecision,trecall,tfscore = get_precision(tpred,tlabels),get_recall(tpred,tlabels),get_fscore(tpred,tlabels)
    dprecision,drecall,dfscore = get_precision(dpred,dlabels),get_recall(dpred,dlabels),get_fscore(dpred,dlabels)
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return training_performance,development_performance

### 2.5: Logistic Regression
# fscore : 0.6816693944353518
## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    tlength = [ 0 if len(word) == None else len(word)for word in twords]
    dlength = [ 0 if len(word) == None else len(word)for word in dwords]
    tfreq = [ 0 if counts.get(w) == None else counts.get(w) for w in twords]
    dfreq = [ 0 if counts.get(w) == None else counts.get(w) for w in dwords]
    
    tl = np.array(tlength)
    meanl = np.mean(tl)
    stdl  = np.std(tl)
    tl_scale = [(l - meanl)/stdl for l in tl]
    dl = np.array(dlength)
    dl_scale = [(l - meanl)/stdl for l in dl]
    tf = np.array(tfreq)
    meanf = np.mean(tf)
    stdf = np.std(tf)
    tf_scale = [(f - meanf)/stdf for f in tf]
    df = np.array(dfreq)
    df_scale = [(f - meanf)/stdf for f in df]
    X_train = np.matrix([tl_scale,tf_scale]).T
    X_test = np.matrix([dl_scale,df_scale]).T
    Y = tlabels
    
    lr = LogisticRegression(C=1000.0, random_state=0, solver='liblinear')
    lr.fit(X_train,Y)
    tpred = lr.predict(X_train)
    dpred = lr.predict(X_test)
    tprecision,trecall,tfscore = get_precision(tpred,tlabels),get_recall(tpred,tlabels),get_fscore(tpred,tlabels)
    dprecision,drecall,dfscore = get_precision(dpred,dlabels),get_recall(dpred,dlabels),get_fscore(dpred,dlabels)
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return training_performance, development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE
def get_syns_nums(words):
    syns = []
    for word in words:
        counts = 0
        for each in wordnet.synsets(word):
            counts += len(each.lemma_names())
        syns.append(counts)
    return syns

def get_syllables(words):
    return [syllables.count_syllables(word) for word in words]

def get_senses(words):
    return [len(wordnet.synsets(word)) for word in words]

def regularization(paras):
    data = np.array(paras)
    mean = np.mean(data)
    std = np.std(data)
    return [(each - mean)/std for each in data]

def random_forest_classifier(training, develop,test):
    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    pwords= load_test(test)
    total_words = twords + dwords
    total_labels = tlabels + dlabels
    syns = regularization(get_syns_nums(pwords))
    syll = regularization(get_syllables(pwords))
    sens = regularization(get_senses(pwords))
    length = regularization([ 0 if len(word) == None else len(word)for word in pwords])
    freq = regularization([ 0 if counts.get(w) == None else counts.get(w) for w in pwords])
    syns_t = regularization(get_syns_nums(total_words))
    syll_t = regularization(get_syllables(total_words))
    sens_t = regularization(get_senses(total_words))
    length_t = regularization([ 0 if len(word) == None else len(word)for word in total_words])
    freq_t = regularization([ 0 if counts.get(w) == None else counts.get(w) for w in total_words])

    X_train= np.matrix([length_t,freq_t,syns_t,syll_t,sens_t]).T
    X_test= np.matrix([length,freq,syns,syll,sens]).T
    rfc = RandomForestClassifier(n_estimators=400, max_depth=3, random_state=0, max_features=3,oob_score=True, min_samples_leaf=2, min_samples_split=7)
    rfc.fit(X_train,total_labels)
    tpred_rfc = rfc.predict(X_train)
    print(get_fscore(tpred_rfc,total_labels))
    # dpred_rfc = rfc.predict(X_test)
    pred_rfc = rfc.predict(X_test)
    results = np.array(pred_rfc).astype(np.int)
    np.savetxt("test_labels.txt",results,fmt="%d")


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    random_forest_classifier(training_file,development_file,test_file)
    
