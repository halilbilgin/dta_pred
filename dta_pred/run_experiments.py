from __future__ import print_function
import binascii
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import Namespace

from .datahelper import *
from keras.models import load_model
import os
from .emetrics import *
from .utils import over_sampling, under_sampling, makedirs
from dta_pred.models.dnn_model import dnn_model
from .arguments import logging
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

def get_dataset(FLAGS):
    all_train_drugs, all_train_prots, all_train_Y = load_data(FLAGS)

    tr_fold, test_fold = get_train_test_split_by_drugs(all_train_drugs, 70, seed=FLAGS.seed)

    new_tr_fold, val_fold = get_train_test_split_by_drugs(all_train_drugs[tr_fold], 70, seed=FLAGS.seed)
    val_fold = tr_fold[val_fold]
    new_tr_fold = tr_fold[new_tr_fold]
    print("Train: "+str(len(new_tr_fold))+" validation: "+str(len(val_fold))+\
                   " and test set:"+str(len(test_fold)))

    sampling_method = None
    if FLAGS.resampling == 'over':
        sampling_method = over_sampling
    elif FLAGS.resampling == 'under':
        sampling_method = under_sampling

    if sampling_method:
        new_tr_fold, _ = sampling_method(pd.Series(new_tr_fold), pd.Series(all_train_Y[new_tr_fold] > 7))
        new_tr_fold = new_tr_fold.values[:, 0]
        XD_train, XT_train, Y_train = all_train_drugs[new_tr_fold], all_train_prots[new_tr_fold], all_train_Y[
            new_tr_fold]
    else:
        XD_train, XT_train, Y_train = all_train_drugs[tr_fold], all_train_prots[tr_fold], all_train_Y[tr_fold]

    XD_val, XT_val, Y_val = all_train_drugs[val_fold], all_train_prots[val_fold], all_train_Y[val_fold]
    XD_test, XT_test, Y_test = all_train_drugs[test_fold], all_train_prots[test_fold], all_train_Y[test_fold]

    return XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, Y_train, Y_val, Y_test

def train_model(FLAGS):

    XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, \
            Y_train, Y_val, Y_test = get_dataset(FLAGS)

    param_name = str(binascii.b2a_hex(os.urandom(8))).replace("'", '')

    checkpoint_dir = os.path.join(FLAGS.checkpoints_path)

    makedirs(checkpoint_dir)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25)
    results = []
    best_rmse_ind = 0

    for repeat_id in range(3):
        checkpoint_file = os.path.join(checkpoint_dir, 'davis_dtc_dta_' + param_name + 'repeat' + str(repeat_id) + '.h5')

        checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                              save_best_only=True)

        K.clear_session()

        gridmodel = dnn_model(FLAGS)

        gridres = gridmodel.fit(([XD_train, XT_train]), Y_train, batch_size=FLAGS.batch_size,
                      epochs=FLAGS.num_epoch,
                      validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val))
                      , callbacks=[early_stopping_callback, checkpoint_callback], verbose=2)
        K.clear_session()
        del gridmodel

        gridmodel = load_model(checkpoint_file)

        predicted_labels = gridmodel.predict([np.array(XD_test), np.array(XT_test)])
        results.append({
            'test_loss': mean_squared_error(Y_test, predicted_labels),
            'test_cindex': get_cindex(Y_test, predicted_labels),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, predicted_labels)),
            'test_f1': f1_score(Y_test > FLAGS.binary_th, predicted_labels > FLAGS.binary_th),
            'train_val_hist': gridres.history,
            'checkpoint_file': checkpoint_file
        })

        logging('--REPEAT' +str(repeat_id) + '--\n' + str(results[-1]) + '\n----\n', FLAGS)

        if results[best_rmse_ind]['test_rmse'] > results[repeat_id]['test_rmse']:
            best_rmse_ind = repeat_id

    return results[best_rmse_ind]


def run_experiment(_run, FLAGS):
    FLAGS = Namespace(**vars(FLAGS))

    results = train_model(FLAGS)

    logging('---BEST RUN test results---', FLAGS)

    for metric, val in results.items():
        if metric[:4] == 'test':
            _run.log_scalar(metric, val)
            logging(metric + '=' + str(val), FLAGS)

    #_run.add_artifact(results['checkpoint_file'], 'model_file')

    for metric, vals in results['train_val_hist'].items():
        prefix = 'train_'
        if 'val' in metric:
            prefix = 'val_'

        for i, val in enumerate(vals):
            _run.log_scalar(prefix+metric, val, step=i)
