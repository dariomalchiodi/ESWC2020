#!/usr/bin/env python
# coding: utf-8

# In[3]:


import itertools as it
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

from possibilearn import flatten

rs = 20190105


# In[4]:


import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create STDERR handler
handler_video = logging.StreamHandler(sys.stderr)
# Create file handler
handler_file = logging.FileHandler('axiom-classification.log')

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler_video.setFormatter(formatter)
handler_file.setFormatter(formatter)

# Set STDERR handler as the only handler
logger.handlers = [handler_video, handler_file]


# In[5]:


mu = np.load('mu.npy')


# In[6]:


gram_jaccard = np.load('jaccard_similarity.npy')
dist_jaccard = 1 - gram_jaccard


# In[7]:


dist_length = np.load('length_distance.npy')
gram_length = 1 - dist_length


# In[8]:


dist_leven = np.load('levenshtein_distance.npy')
gram_leven = 1 - dist_leven


# In[9]:


dist_hamming = np.load('hamming_distance.npy')
gram_hamming = 1 - dist_hamming


# In[10]:


def alpha_cut(alpha):
    return lambda x: 1 if x >= alpha else 0


def cut_labels(mu, alpha):
    return np.array(list(map(alpha_cut(alpha), mu)))


# In[11]:


reduced_data_dir = 'data/reduction/'

def learn_experiment(data_matrix,
                     reduction_procedure, reduced_data_filename,
                     mu,
                     learning_algorithm_factory,
                     d=2,
                     alpha_levels=(.2, .4, .5, .75, .9),
                     num_holdouts=5,
                     percentages=(.8, 0, .2),
                     hyperparams_set = {},
                     validation_metric = metrics.accuracy_score,
                     test_metric = metrics.accuracy_score,
                     verbose=False):

    # this is to check that any time we have no parameters the validation
    # percentage is 0 and vice versa

    assert((hyperparams_set!={} or percentages[1]==0) and            (percentages[1]!=0 or hyperparams_set=={}))

    hyperparams_iterator = ParameterGrid(hyperparams_set)


    reduced_data_fullname = reduced_data_dir + reduced_data_filename + '.npy'
    if os.path.isfile(reduced_data_fullname):
        logger.info('Found cached reduced data. Retrieving {}'.format(reduced_data_filename))
        X = np.load(reduced_data_fullname)
    else:
        logger.info('Performing reduction procedure...')
        X = reduction_procedure(n_components=d,
                                random_state=rs).fit_transform(data_matrix)

        logger.info('Done!')
        np.save(reduced_data_fullname, X)

    label_set = [cut_labels(mu, alpha) for alpha in alpha_levels]

    n = len(X)
    assert(n==len(mu))

    ##paired_X = [X[i:i+2] for i in range(0, n, 2)]

    best_test_median = float('inf')
    best_test_metric = []

    for alpha, y in zip(alpha_levels, label_set):

        logger.info('Checking alpha={:.2f}'.format(alpha))

        ##paired_y = [y[i:i+2] for i in range(0, n, 2)]

        metric_train = []
        metric_test = []

        for h in range(num_holdouts):
            (X_train,
             X_validate_test,
             y_train,
             y_validate_test) = train_test_split(X, y,
                                                 train_size=percentages[0],
                                                 test_size=1-percentages[0],
                                                 stratify=y)

            if percentages[1]==0: # no validation set
                X_validate = []
                X_test = X_validate_test
                y_validate = []
                y_test = y_validate_test
            else:
                val_perc_rel = percentages[1]/(percentages[1]+percentages[2])
                (X_validate,
                 X_test,
                 y_validate,
                 y_test) = train_test_split(X_validate_test, y_validate_test,
                                            train_size=val_perc_rel,
                                            test_size=1-val_perc_rel,
                                            stratify=y_validate_test)

            logger.info('holdout {} of {}'.format(h+1, num_holdouts))

            best_hyperparams = {}
            best_err = np.inf

            for hyperparams in hyperparams_iterator:
                if not hyperparams:
                    break

                #logger.info('Checking {}'.format(hyperparams))

                learning_algorithm = learning_algorithm_factory(**hyperparams)
                learning_algorithm.fit(X_train, y_train)
                y_pred = learning_algorithm.predict(X_validate)
                error = validation_metric(y_validate, y_pred)
                if error < best_err:
                    best_err = error
                    best_hyperparams = hyperparams

            logger.info('Learning with best '
                        'hyperparams: {}'.format(best_hyperparams))
            learning_algorithm = learning_algorithm_factory(**best_hyperparams)
            if percentages[1] == 0: # no validation set
                X_train_val = X_train
                y_train_val = y_train
            else:
                X_train_val = np.vstack([X_train, X_validate])
                y_train_val = np.hstack([y_train, y_validate])
            assert(len(X_train_val) == len(X_train) + len(X_validate))
            assert(len(y_train_val) == len(y_train) + len(y_validate))
            learning_algorithm.fit(X_train_val, y_train_val)
            pred_train = learning_algorithm.predict(X_train_val)
            pred_test = learning_algorithm.predict(X_test)

            metric_train.append(test_metric(y_train_val, pred_train))
            metric_test.append(test_metric(y_test, pred_test))

        test_median = np.median(metric_test)
        if test_median < best_test_median:
            best_test_median = test_median
            best_test_metric = [alpha,
                                np.mean(metric_train),
                                np.median(metric_train),
                                np.std(metric_train),
                                np.mean(metric_test),
                                test_median,
                                np.std(metric_test)]
    names = ['alpha',
             'train_mean', 'train_median', 'train_std',
             'test_mean','test_median', 'test_std']
    return dict(zip(names, best_test_metric))


# In[12]:


def SVC_custom(*args, **kwargs):
    return SVC(*args, **kwargs, max_iter=5000)

from sklearn.preprocessing import MinMaxScaler

class ScaledSVC:
    def __init__(self, *args, **kwargs):
        self.svc = SVC(*args, **kwargs, max_iter=5000)
        self.scaler = MinMaxScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        return self.svc.fit(self.scaler.transform(X), y)

    def predict(self, X):
        self.scaler.fit(X)
        return self.svc.predict(self.scaler.transform(X))

def RandomForestClassifier_custom(*args, **kwargs):
    return RandomForestClassifier(*args, **kwargs, n_estimators=100)

def MLPClassifier_custom(*args, **kwargs):
    return MLPClassifier(*args, **kwargs, max_iter=5000)


# In[13]:


import json
import os

result_dir = './data/classification/'


# In[14]:


# considered learning algorithms
names = ['Decision tree', 'Random forest', 'Naive Bayes', 'LDA', 'MLP',
         'SVC (linear)', 'SVC (gaussian)']
algorithms = [DecisionTreeClassifier, RandomForestClassifier,
              GaussianNB, LinearDiscriminantAnalysis, MLPClassifier_custom,
              ScaledSVC, ScaledSVC]
hyperp_sets = [{'criterion': ['gini', 'entropy'], 'max_leaf_nodes': [None, 2, 5, 10, 50, 100],
                'max_features': [None, 'sqrt', 'log2'], 'max_depth': [None, 2, 5, 10]},
               {'n_estimators': [5, 10, 50, 100, 200], 'criterion': ['gini', 'entropy'], 'max_leaf_nodes': [None, 2, 5, 10, 50, 100],
                'max_features': [None, 'sqrt', 'log2'], 'max_depth': [None, 2, 5, 10]},
               {}, {}, {'hidden_layer_sizes': [[2], [4], [6], [10], [20]]},
               {'C': [.001, .01, .1, 1, 10], 'kernel': ['linear']},
               {'C': [.001, .01, .1, 1, 10], 'kernel': ['rbf'], 'gamma': [.001, .01, .1, 1, 10, 30]}]
percentages = [(.8, .1, .1), (.8, .1, .2), (.8, 0, .2), (.8, 0, .2),
               (.8, .1, .1),(.8, .1, .1), (.8, .1, .1)]

# considered data matrices
data_names = ['hamming', 'jaccard', 'length', 'levenshtein']
data_matrices = [gram_hamming, gram_jaccard, gram_length, gram_leven]

# considered number of extracted components
components = [2, 3, 5, 10, 30]


# In[16]:


re_generate_files = False
append = True

num_groups = 4
for g in range(num_groups):
    num_experiments = len(algorithms) *  len(data_matrices) *  len(components)

    learning_procedures = zip(names, algorithms, hyperp_sets, percentages)

    data = zip(data_names, data_matrices)

    experiments = it.product(learning_procedures, data, components)

    performed_experiments = 0

    logger.info('Started experiment group {} of {}'.format(g+1, num_groups))
    for experiment in experiments:
        learning_procedure = experiment[0]
        data_item = experiment[1]
        comp = experiment[2]
        out_file = result_dir + 'class-' +                    learning_procedure[0].replace(' ', '_') + '-' +                    'pca-' +                    data_item[0] + '-' +                    str(comp) + '.json'
        logger.info('Considering {}'.format(out_file))
        if os.path.isfile(out_file) and not re_generate_files and not append:
            logger.info('already exists, skipping')
        else:
            access_flag = 'w' if not append else 'a'
            if append:
                logger.info('appending to existing file')
            elif re_generate_files:
                logger.info('re-executing experiment')
            else:
                logger.info('does not exist, starting experiment')
            # perform experiment
            res = learn_experiment(data_item[1],
                                   KernelPCA, 'PCA-' + experiment[1][0],
                                   mu,
                                   learning_procedure[1],
                                   d=comp,
                                   alpha_levels=(.2, .4, .5, .6, .8, .9),
                                   percentages=learning_procedure[3],
                                   hyperparams_set = learning_procedure[2])

            if access_flag == 'a':
                logger.info('opening result file in append')
            else:
                logger.info('creating result file')
            with open(out_file, access_flag) as f:
                f.write('\n')
                json.dump(res, f)

            logger.info('experiment finished')
        performed_experiments += 1
        perc_complete = 100 * performed_experiments / num_experiments
        logger.info('completed {:.2f}%'.format(perc_complete))
    logger.info('Ended experiment group {} of {}'.format(g+1, num_groups))
