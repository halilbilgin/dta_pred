from __future__ import print_function
import binascii
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import Namespace

from .datahelper import *
from keras.models import load_model, Model
from keras.layers import Embedding, Dense
from dta_pred.models.dnn_model import auto_model, fully_connected_model, \
    simple_cnn_encoder, inception_encoder, get_pooling
from .emetrics import *
from .utils import over_sampling, under_sampling, makedirs
from .protein_encoding import auto_protein_encoding
from .drug_encoding import auto_drug_encoding

from .models import *
from .arguments import logging
import os

sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

def load_data(FLAGS):
    dataset_per_task = {}

    for dataset_name in FLAGS.datasets_included:
        dataset = DataSet( dataset_path = FLAGS.dataset_path,
                       dataset_name=dataset_name,
                       seqlen = FLAGS.max_seq_len,
                       smilen = FLAGS.max_smi_len,
                       protein_format=FLAGS.protein_format,
                       drug_format=FLAGS.drug_format,
                       mol2vec_model_path=FLAGS.mol2vec_model_path,
                       mol2vec_radius=FLAGS.mol2vec_radius,
                       biovec_model_path=FLAGS.biovec_model_path
                       )
        XD, XT, Y = dataset.parse_data()

        if dataset.interaction_type not in dataset_per_task:
            dataset_per_task[dataset.interaction_type] = {'drugs': None, 'proteins': None, 'Y': None}

        if dataset.interaction_type == 'Kd':
            Y = -np.log10(np.asarray(Y)/1e9)

        data_dict = dataset_per_task[dataset.interaction_type]

        if type(data_dict['drugs']) is not np.ndarray:
            data_dict['drugs'], data_dict['proteins'], data_dict['Y']= np.asarray(XD), np.asarray(XT), np.asarray(Y)
        else:
            data_dict['drugs'] = np.concatenate((np.asarray(data_dict['drugs']), np.asarray(XD)), axis=0)
            data_dict['proteins'] = np.concatenate((np.asarray(data_dict['proteins']), np.asarray(XT)), axis=0)
            data_dict['Y'] = np.concatenate((np.asarray(data_dict['Y']), np.asarray(Y)), axis=0)

    for interaction_type, data in dataset_per_task.items():
        shuffled_inds = np.asarray([i for i in range(data['proteins'].shape[0])])

        for i in range(0, 3):
            np.random.seed(FLAGS.seed)
            np.random.shuffle(shuffled_inds)

        data['drugs'], data['proteins'], data['Y'] = data['drugs'][shuffled_inds], data['proteins'][shuffled_inds], data['Y'][shuffled_inds]

    if len(dataset_per_task.keys()) == 1:
        return dataset_per_task[dataset.interaction_type]
    else:
        return dataset_per_task

def train_multitask_model_v2(datasets, inputs, encode_smiles, encode_protein, smi_model,
                             seq_model, interaction_model, output_path, batch_size,
                             num_epoch=100, fold_id=0, **kwargs):
    XDinput, encode_smiles = encode_smiles()
    XTinput, encode_protein = encode_protein()

    tasks = datasets.keys()

    shared_model = DTIModel(inputs, encode_smiles, encode_protein, smi_model(), seq_model(), interaction_model())
    shared_layers = shared_model.interaction_module

    losses = {}

    for key, dataset in datasets.items():
        datasets[key] = train_val_test_split(dataset['drugs'], dataset['proteins'], dataset['Y'], fold_id=fold_id, seed=kwargs['seed'])

        if key == 'Kd':
            losses[key] = 'crossentropy_mse_combined'
        else:
            losses[key] = kwargs['loss']

    task_specific_layers = {}
    for task in tasks:
        task_specific_layers[task] = fully_connected_model(name=task+'_fc', **kwargs)

    multitask_model = MultiTaskModel(inputs, shared_layers, task_specific_layers, tasks)

    models = multitask_model.compile(optimizers=kwargs['optimizer'], losses=losses)

    checkpoint_callbacks = {}

    checkpoints_path = os.path.join(output_path, 'checkpoints')
    makedirs(checkpoints_path)
    log_path = os.path.join(output_path, 'logs')
    makedirs(log_path)

    for task in tasks:
        param_name = str(binascii.b2a_hex(os.urandom(4))).replace("'", '')

        checkpoint_file = os.path.join(checkpoints_path, task + '_' + param_name + '.h5')

        checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                          save_best_only=True)

        checkpoint_callbacks[task] = checkpoint_callback

    for i in range(num_epoch):
        for task in tasks:
            XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, Y_train, Y_val, Y_test = datasets[task]

            gridres = models[task].fit(([XD_train, XT_train]), Y_train, batch_size=batch_size,
                                   epochs=1,
                                   validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val))
                                   , callbacks=[checkpoint_callbacks[task]], verbose=2)

    gridmodel = load_model(checkpoint_callbacks['Kd'])
    _, _, XD_test, _, _, XT_test, _, _, Y_test = datasets['Kd']

    predicted_labels = gridmodel.predict([np.array(XD_test), np.array(XT_test)])

    return {
            'test_loss': mean_squared_error(Y_test, predicted_labels),
            'test_cindex': get_cindex(Y_test, predicted_labels),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, predicted_labels)),
            'test_f1': f1_score(Y_test > kwargs['binary_th'], predicted_labels > kwargs['binary_th']),
            'train_val_hist': gridres.history,
            'checkpoint_file': checkpoint_file
        }

def train_model(dataset, encode_smiles, encode_protein, smi_model, seq_model, interaction_model,
                output_path, optimizer, loss, num_epoch, batch_size, n_repeats=3, **kwargs):

    XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, \
            Y_train, Y_val, Y_test = train_val_test_split(seed=kwargs['seed'], **dataset)

    param_name = str(binascii.b2a_hex(os.urandom(4))).replace("'", '')
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    makedirs(checkpoint_dir)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25)
    results = []
    best_rmse_ind = 0

    for repeat_id in range(n_repeats):
        checkpoint_file = os.path.join(checkpoint_dir,
                                       'davis_dtc_dta_' + param_name + 'repeat' + str(repeat_id) + '.h5')
        checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                              save_best_only=True)
        K.clear_session()
        XDinput, encode_smiles = encode_smiles()
        XTinput, encode_protein = encode_protein()

        model = DTIModel([XDinput, XTinput], encode_smiles, encode_protein, smi_model(), seq_model(), interaction_model())
        compiled_model = model.compile(optimizer=optimizer, loss=loss)


        gridres = compiled_model.fit(([XD_train, XT_train]), Y_train, batch_size=batch_size,
                      epochs=num_epoch,
                  validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val))
                      , callbacks=[early_stopping_callback, checkpoint_callback], verbose=2)

        K.clear_session()
        del compiled_model, model

        gridmodel = load_model(checkpoint_file)

        predicted_labels = gridmodel.predict([np.array(XD_test), np.array(XT_test)])
        results.append({
            'test_loss': mean_squared_error(Y_test, predicted_labels),
            'test_cindex': get_cindex(Y_test, predicted_labels),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, predicted_labels)),
            'test_f1': f1_score(Y_test > kwargs['binary_th'], predicted_labels > kwargs['binary_th']),
            'train_val_hist': gridres.history,
            'checkpoint_file': checkpoint_file
        })

        logging('--REPEAT' +str(repeat_id) + '--\n' + str(results[-1]) + '\n----\n')

        if results[best_rmse_ind]['test_rmse'] > results[repeat_id]['test_rmse']:
            best_rmse_ind = repeat_id

    return results[best_rmse_ind]

def run_experiment(_run, FLAGS):
    FLAGS = Namespace(**vars(FLAGS))

    data_per_task = load_data(FLAGS)

    encode_smiles = auto_drug_encoding(smi_input_dim=FLAGS.max_smi_len, **vars(FLAGS))
    encode_protein = auto_protein_encoding(seq_input_dim=FLAGS.max_seq_len, **vars(FLAGS))

    FLAGS.smi_model = auto_model(FLAGS.smi_model, kernel_size=FLAGS.smi_window_length, name='smi_enc', **vars(FLAGS))
    FLAGS.seq_model = auto_model(FLAGS.seq_model, kernel_size=FLAGS.seq_window_length, name='seq_enc', **vars(FLAGS))
    FLAGS.interaction_model = auto_model('fully_connected', name='interaction_fc', **vars(FLAGS))

    if 'drugs' not in data_per_task:
        results = train_multitask_model_v2(data_per_task, encode_smiles, encode_protein, **vars(FLAGS))
    else:
        results = train_model(data_per_task, encode_smiles, encode_protein, **vars(FLAGS))

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
