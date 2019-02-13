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
import numpy as np
import gzip

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

### 2.1: A very simple baseline

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

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    return [1 if len(word[i])>= threshold else 0 for i in range(len(words))]

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
   with gzip.open(ngram_counts_file, 'rt') as f: 
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
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
