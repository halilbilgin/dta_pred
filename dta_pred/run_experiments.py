from __future__ import print_function
import binascii
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import Namespace

from .datahelper import *
from keras.models import load_model, Model
from keras.layers import Embedding, Dense
from dta_pred.models.dnn_model import dnn_model, fully_connected_model, \
    simple_cnn_encoder, inception_encoder, get_pooling
from .emetrics import *
from .utils import over_sampling, under_sampling, makedirs
from .arguments import logging
import os


sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

def get_dataset(FLAGS):
    all_train_drugs, all_train_prots, all_train_Y = load_data(FLAGS)

    train_drugs, val_drugs = 70, 70
    if 'kiba' in FLAGS.datasets_included:
        train_drugs, val_drugs = 10, 300

    tr_fold, test_fold = get_train_test_split_by_drugs(all_train_drugs, train_drugs, seed=FLAGS.seed)

    new_tr_fold, val_fold = get_train_test_split_by_drugs(all_train_drugs[tr_fold], val_drugs, seed=FLAGS.seed)
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

def build_model(FLAGS, interaction_model=None):
    if interaction_model == None:
        interaction_model = fully_connected_model(FLAGS.n_fc_layers, FLAGS.n_neurons_fc,
                                              FLAGS.dropout, 'fc', FLAGS.apply_bn)

    smi_model = None
    smi_embedding = Embedding(input_dim=CHARISOSMILEN + 1, output_dim=128,
                              input_length=FLAGS.max_smi_len, name='smi_embedding')
    if FLAGS.smi_model == 'inception':
        smi_model = inception_encoder(FLAGS.num_windows, FLAGS.smi_window_length, 'smi_enc')
    elif FLAGS.smi_model == 'simple_cnn':
        smi_model = simple_cnn_encoder(FLAGS.n_cnn_layers, FLAGS.num_windows,
                                       FLAGS.smi_window_length, 'smi_enc')
    else:
        smi_embedding = None

    seq_model = None
    seq_embedding = Embedding(input_dim=CHARPROTLEN + 1, output_dim=128,
                              input_length=FLAGS.max_seq_len, name='seq_embedding')
    if FLAGS.seq_model == 'inception':
        seq_model = inception_encoder(FLAGS.num_windows, FLAGS.seq_window_length, 'seq_enc')
    elif FLAGS.seq_model == 'simple_cnn':
        seq_model = simple_cnn_encoder(FLAGS.n_cnn_layers, FLAGS.num_windows,
                                       FLAGS.seq_window_length, 'seq_enc')
    else:
        seq_embedding = None

    gridmodel = dnn_model(FLAGS.drug_format, FLAGS.protein_format, FLAGS.max_smi_len,
                          FLAGS.max_seq_len, interaction_model=interaction_model,
                          loss_fn=FLAGS.loss, smi_model=smi_model, seq_model=seq_model, smi_embedding=smi_embedding,
                          seq_embedding=seq_embedding, smi_pooling=get_pooling(FLAGS.pooling_type),
                          seq_pooling=get_pooling(FLAGS.pooling_type))
    return gridmodel

def train_multitask_model(FLAGS):

    multitask_flags = Namespace(**vars(FLAGS))
    multitask_flags.loss = 'mean_squared_error'
    multitask_flags.datasets_included=['kiba']
    multitask_flags.checkpoints_path = os.path.join(multitask_flags.checkpoints_path, 'kiba')
    makedirs(os.path.join(multitask_flags.checkpoints_path, 'kiba'))

    result = train_model(multitask_flags, 2)

    def new_model_base():
        model_kiba = load_model(result['checkpoint_file'])

        encode_interaction = model_kiba.get_layer('encode_interaction').output

        FC = fully_connected_model(FLAGS.n_fc_layers, FLAGS.n_neurons_fc, FLAGS.dropout, 'kd_fc',
                                   FLAGS.apply_bn)(encode_interaction)

        predictions = Dense(1, kernel_initializer='normal')(FC)

        interactionModel = Model(inputs=model_kiba.inputs, outputs=[predictions])

        interactionModel.compile(optimizer='adam', loss=FLAGS.loss, metrics=[cindex, f1])

        return interactionModel

    return train_model(FLAGS, 3, new_model_base)

def train_multitask_model_v2(FLAGS):
    XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, \
            Y_train, Y_val, Y_test = get_dataset(FLAGS)

    multitask_flags = Namespace(**vars(FLAGS))
    multitask_flags.loss = 'mean_squared_error'
    multitask_flags.datasets_included=['kiba']
    multitask_flags.checkpoints_path = os.path.join(multitask_flags.checkpoints_path, 'kiba')
    makedirs(os.path.join(multitask_flags.checkpoints_path, 'kiba'))

    XD_train_kiba, XD_val_kiba, XD_test_kiba, XT_train_kiba, XT_val_kiba, XT_test_kiba, \
            Y_train_kiba, Y_val_kiba, Y_test_kiba = get_dataset(multitask_flags)

    FC_kiba = fully_connected_model(FLAGS.n_fc_layers*2, FLAGS.n_neurons_fc,
                                      FLAGS.dropout, 'shared_fc', FLAGS.apply_bn)

    model_kiba = build_model(multitask_flags, FC_kiba)
    model_kiba.summary()
    encode_interaction = model_kiba.get_layer('shared_fc_'+str(FLAGS.n_fc_layers-1)).output

    FC = fully_connected_model(FLAGS.n_fc_layers, FLAGS.n_neurons_fc, FLAGS.dropout, 'kd_fc',
                               FLAGS.apply_bn)(encode_interaction)

    predictions = Dense(1, kernel_initializer='normal')(FC)

    model_kd = Model(inputs=model_kiba.inputs, outputs=[predictions])
    model_kd.compile(optimizer='adam', loss=FLAGS.loss, metrics=[cindex, f1])

    param_name = str(binascii.b2a_hex(os.urandom(8))).replace("'", '')
    checkpoint_dir = os.path.join(FLAGS.checkpoints_path)
    makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, 'davis_dtc_dta_' + param_name + '.h5')

    checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                          save_best_only=True)
    for i in range(FLAGS.num_epoch):
        gridres = model_kd.fit(([XD_train, XT_train]), Y_train, batch_size=FLAGS.batch_size,
                                epochs=1,
                                validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val))
                                , callbacks=[checkpoint_callback], verbose=2)
        model_kiba.fit(([XD_train_kiba, XT_train_kiba]), Y_train_kiba, batch_size=FLAGS.batch_size,
                     epochs=1,
                     validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val))
                     , callbacks=[checkpoint_callback], verbose=2)

    gridmodel = load_model(checkpoint_file)

    predicted_labels = gridmodel.predict([np.array(XD_test), np.array(XT_test)])

    return {
            'test_loss': mean_squared_error(Y_test, predicted_labels),
            'test_cindex': get_cindex(Y_test, predicted_labels),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, predicted_labels)),
            'test_f1': f1_score(Y_test > FLAGS.binary_th, predicted_labels > FLAGS.binary_th),
            'train_val_hist': gridres.history,
            'checkpoint_file': checkpoint_file
        }

def train_model(FLAGS, n_repeats=3, model_fn=None):

    XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, \
            Y_train, Y_val, Y_test = get_dataset(FLAGS)

    param_name = str(binascii.b2a_hex(os.urandom(8))).replace("'", '')
    checkpoint_dir = os.path.join(FLAGS.checkpoints_path)
    makedirs(checkpoint_dir)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25)
    results = []
    best_rmse_ind = 0

    for repeat_id in range(n_repeats):
        checkpoint_file = os.path.join(checkpoint_dir, 'davis_dtc_dta_' + param_name + 'repeat' + str(repeat_id) + '.h5')

        checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                              save_best_only=True)

        K.clear_session()

        if model_fn is None:
            gridmodel = build_model(FLAGS)
        else:
            gridmodel = model_fn()

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

    if FLAGS.model_name == 'multitask_model':
        results = train_multitask_model(FLAGS)
    elif FLAGS.model_name == 'multitask_model_v2':
        results = train_multitask_model_v2(FLAGS)
    else:
        results = train_model(FLAGS)

    logging('---BEST RUN test results---', FLAGS)

    for metric, val in results.items():
        if metric[:4] == 'test':
            if type(val) == np.ndarray:
                val = val[0]

            _run.log_scalar(metric, val)
            logging(metric + '=' + str(val), FLAGS)

    #_run.add_artifact(results['checkpoint_file'], 'model_file')

    for metric, vals in results['train_val_hist'].items():
        prefix = 'train_'
        if 'val' in metric:
            prefix = 'val_'

        for i, val in enumerate(vals):
            _run.log_scalar(prefix+metric, val, step=i)
