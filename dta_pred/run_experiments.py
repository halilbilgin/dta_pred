from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn
import binascii
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sacred.observers import MongoObserver
from keras.callbacks import ModelCheckpoint, Callback
from argparse import Namespace
### We modified Pahikkala et al. (2014) dta_pred code for cross-val process ###

import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


from .datahelper import *
#import logging
from itertools import product
from keras.models import Sequential, load_model

import sys, pickle, os
import math, json, time
import matplotlib.pyplot as plt

from .arguments import argparser, logging
from .emetrics import *
from .utils import over_sampling, under_sampling
from .model import standard_model

from sacred import Experiment

TABSY = "\t"
figdir = "figures/"

def train_model( FLAGS ):

    all_train_drugs, all_train_prots, all_train_Y = load_data(FLAGS)

    tr_fold, test_fold = get_train_test_split_by_drugs(all_train_drugs, 70, seed=FLAGS.seed)

    param_name = str(binascii.b2a_hex(os.urandom(8))).replace("'", '')

    checkpoint_dir = os.path.join(FLAGS.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25)
    results = []
    best_rmse_ind = 0

    for repeat_id in range(3):
        checkpoint_file = os.path.join(checkpoint_dir, 'davis_dtc_dta_' + param_name + 'repeat' + str(repeat_id) + '.h5')

        checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                              save_best_only=True)

        val_start_index = int(len(tr_fold) * 0.8)
        val_fold = tr_fold[val_start_index:]
        tr_fold = tr_fold[:val_start_index]

        if FLAGS.resampling == 'over':
            sampling_method = over_sampling
        elif FLAGS.resampling == 'down':
            sampling_method = under_sampling
        else:
            raise NotImplementedError()

        if FLAGS.resampling:
            new_tr_fold, _ = sampling_method(pd.Series(tr_fold), pd.Series(all_train_Y[tr_fold] > 7))
            new_tr_fold = new_tr_fold.values[:, 0]
            XD_train, XT_train, Y_train = all_train_drugs[new_tr_fold], all_train_prots[new_tr_fold], all_train_Y[
                new_tr_fold]
        else:
            XD_train, XT_train, Y_train = all_train_drugs[tr_fold], all_train_prots[tr_fold], all_train_Y[tr_fold]

        XD_val, XT_val, Y_val = all_train_drugs[val_fold], all_train_prots[val_fold], all_train_Y[val_fold]
        XD_test, XT_test, Y_test = all_train_drugs[test_fold], all_train_prots[test_fold], all_train_Y[test_fold]

        gridmodel = standard_model(FLAGS)

        gridres = gridmodel.fit(([XD_train, XT_train]), Y_train, batch_size=FLAGS.batch_size,
                      epochs=FLAGS.num_epoch,
                      validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val))
                      , callbacks=[early_stopping_callback, checkpoint_callback], verbose=2)

        gridmodel = load_model(checkpoint_file)

        predicted_labels = gridmodel.predict([np.array(XD_test), np.array(XT_test)])[:, 0]
        results.append({
            'test_loss': mean_squared_error(Y_test, predicted_labels),
            'test_cindex': get_cindex(Y_test, predicted_labels),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, predicted_labels)),
            'test_f1': f1_score(Y_test > 7, predicted_labels > 7),
            'train_val_hist': gridres.history
        })

        if results[best_rmse_ind]['test_rmse'] > results[repeat_id]['test_rmse']:
            best_rmse_ind = repeat_id

    return results[best_rmse_ind]

def run_experiment(_run, FLAGS):
    FLAGS = Namespace(**vars(FLAGS))

    results = train_model(FLAGS)

    for metric, val in results.items():
        if metric[:4] == 'test':
            _run.log_scalar(metric, val)

    for metric, vals in results['train_val_hist'].items():
        prefix = 'train_'
        if 'val' in metric:
            prefix = 'val_'

        for i, val in enumerate(vals):
            _run.log_scalar(prefix+metric, val, step=i)

if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = os.path(FLAGS.log_dir, FLAGS.experiment_name)

    ex = Experiment(FLAGS.experiment_name)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    mongo_conf = FLAGS.mongodb.split(':')
    if mongo_conf != None:
        ex.observers.append(MongoObserver.create(url=':'.join(mongo_conf[:2]), db_name=mongo_conf[2]))

    ex.main(run_experiment)
    ex.add_config(vars(FLAGS))

    r = ex.run(config_updated={'args': FLAGS})


