import pandas as pd
from sklearn.utils import resample
import os
import errno
import math
import numpy as np
from sklearn.model_selection import KFold, PredefinedSplit, train_test_split
def makedirs(folder):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def under_sampling(X_train, y_train):
    df_all = pd.concat((X_train, pd.DataFrame({'value': y_train}, index=y_train.index)), axis=1)

    df_majority = df_all[df_all.value == 0]
    df_minority = df_all[df_all.value == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=df_minority.shape[0],  # to match minority class
                                       random_state=123)
    # Join together class 0's target vector with the downsampled class 1's target vector
    df_downsampled = pd.concat([df_majority_downsampled, df_minority], axis=0)
    y_downsampled = df_downsampled.value
    X_downsampled = df_downsampled.drop('value', axis=1)
    return X_downsampled, y_downsampled

def over_sampling(X_train, y_train):
    df_all = pd.concat((X_train, pd.DataFrame({'value': y_train}, index=y_train.index)), axis=1)

    df_majority = df_all[df_all.value == 0]
    df_minority = df_all[df_all.value == 1]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=df_majority.shape[0],  # to match majority class
                                     random_state=123)  # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0)
    y_upsampled = df_upsampled.value
    X_upsampled = df_upsampled.drop('value', axis=1)

    return X_upsampled, y_upsampled


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def get_n_folds(all_drugs, n_splits=5, seed=42):
    test_folds = np.ones(all_drugs.shape[0])
    kf = KFold(n_splits, random_state=seed)

    j = 0
    for _, current_test_fold in kf.split(np.arange(all_drugs.shape[0])):

        test_folds[current_test_fold] = j
        j += 1

    return PredefinedSplit(test_folds)

def get_n_fold_by_drugs(all_drugs, n_splits=5):
    unique_drugs = np.unique(all_drugs, axis=0)
    test_folds = np.ones(all_drugs.shape[0])
    kf = KFold(n_splits, random_state=15)

    j = 0
    for _, validation_drugs in kf.split(np.arange(unique_drugs.shape[0])):
        val_inds = []

        for drug_ind in validation_drugs:
            willbe_added = list(np.where((~(all_drugs == unique_drugs[drug_ind, :])).sum(axis=1) == 0)[0])
            val_inds += willbe_added
        test_folds[val_inds] = j
        j += 1

    return PredefinedSplit(test_folds)

def train_val_test_split(drugs, proteins, Y, fold_id=0, seed=42):

    all_indices = np.arange(0, drugs.shape[0])

    tr_fold, test_fold = list(get_n_folds(all_indices, seed=seed).split())[fold_id]

    new_tr_fold, val_fold = train_test_split(all_indices[tr_fold], test_size=0.25, random_state=seed)

    print("Train: "+str(len(tr_fold))+" validation: "+str(len(val_fold))+\
                   " and test set:"+str(len(test_fold)))

    XD_train, XT_train, Y_train = drugs[new_tr_fold], proteins[new_tr_fold], Y[new_tr_fold]
    XD_val, XT_val, Y_val = drugs[val_fold], proteins[val_fold], Y[val_fold]
    XD_test, XT_test, Y_test = drugs[test_fold], proteins[test_fold], Y[test_fold]

    return XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, Y_train, Y_val, Y_test

def get_train_test_split_by_drugs(all_drugs, n_drugs_in_test=70, seed=42):
    if len(all_drugs.shape) == 1:
        all_drugs = np.reshape(all_drugs, (-1, 1))
    unique_drugs = np.unique(all_drugs, axis=0)

    assert n_drugs_in_test <= unique_drugs.shape[0]

    np.random.seed(seed)
    test_drugs = np.random.choice(unique_drugs.shape[0], n_drugs_in_test, )

    test_fold = np.ones(all_drugs.shape[0]) * -1

    test_samples = []
    for drug_ind in test_drugs:
        willbe_added = list(np.where((~(all_drugs == unique_drugs[drug_ind, :])).sum(axis=1) == 0)[0])
        test_samples += willbe_added

    assert len(test_samples) >= 1

    test_fold[test_samples] = 1

    return np.where(test_fold==-1)[0], np.where(test_fold==1)[0]