from __future__ import print_function
import binascii
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import Namespace
from keras import regularizers
from .data_helper import *
from keras.models import load_model, Model
from keras.layers import Embedding, Dense
from dta_pred.models.dnn_model import auto_model, fully_connected_model, \
    simple_cnn_encoder, inception_encoder, get_pooling
from .metrics import *
from .utils import *
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
        print(dataset_name)
        if dataset.interaction_type not in dataset_per_task:
            dataset_per_task[dataset.interaction_type] = {'drugs': [[]], 'proteins': [[]], 'Y': []}

        if dataset.interaction_type in ['Kd']:
            Y[Y<=0] = 1
            Y = -np.log10((np.asarray(Y))/1e9)
        elif dataset.interaction_type in ['IC50']:
            Y[Y<=0] = 1
            Y = -np.log10((np.asarray(Y))/1e9)

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

    return dataset_per_task

def train_multitask_model_v2(datasets, smile_encoding_fn, protein_encoding_fn, smi_model,
                             seq_model, interaction_model, output_path, batch_size,
                             num_epoch=100, fold_id=0, **kwargs):

    XDinput, encoded_smiles = smile_encoding_fn()
    XTinput, encoded_protein = protein_encoding_fn()
    inputs = [XDinput, XTinput]
    tasks = datasets.keys()

    shared_model = DTIModel(inputs, encoded_smiles, encoded_protein, smi_model(), seq_model(), interaction_model())
    shared_layers = shared_model.interaction_module

    losses = {}

    for key, dataset in datasets.items():

        if key == 'Kd':
            losses[key] = kwargs['loss']
        else:
            losses[key] = 'mean_squared_error'

    task_specific_layers = {}
    for task in tasks:
        kwargs['n_fc_neurons'] = int(kwargs['n_fc_neurons'] / 4)
        task_specific_layers[task] = fully_connected_model(name=task+'_fc', **kwargs)

    multitask_model = MultiTaskModel(inputs, shared_layers, task_specific_layers, tasks)

    multitask_model.compile(optimizers=kwargs['optimizer'], losses=losses)

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

    multitask_model.train(datasets, checkpoint_callbacks=checkpoint_callbacks, num_epoch=num_epoch, batch_size=batch_size, verbose=2)

    gridmodel = load_model(checkpoint_callbacks['Kd'].filepath)

    _, _, XD_test, _, _, XT_test, _, _, Y_test = datasets['Kd']
    predicted_labels = gridmodel.predict([np.array(XD_test), np.array(XT_test)])
    return {
            'test_loss': mean_squared_error(Y_test, predicted_labels),
            'test_cindex': get_cindex(Y_test, predicted_labels),
            'test_rmse': np.sqrt(mean_squared_error(Y_test, predicted_labels)),
            'test_f1': f1_score(Y_test > kwargs['binary_th'], predicted_labels > kwargs['binary_th']),
            'checkpoint_file': checkpoint_file
        }

def train_model(splitting_dataset, smiles_encoding_fn, protein_encoding_fn, smi_model, seq_model, interaction_model,
                output_path, optimizer, loss, num_epoch, batch_size, n_repeats=3, fold_id=0, **kwargs):
    """

    :param splitting_dataset:
    :param smiles_encoding_fn:
    :param protein_encoding_fn:
    :param smi_model:
    :param seq_model:
    :param interaction_model:
    :param output_path:
    :param optimizer:
    :param loss:
    :param num_epoch:
    :param batch_size:
    :param n_repeats:
    :param fold_id:
    :param kwargs:
    :return:
    """
    assert len(splitting_dataset.keys()) == 1

    task = splitting_dataset.keys()[0]

    XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, \
            Y_train, Y_val, Y_test = splitting_dataset[task]

    param_name = str(binascii.b2a_hex(os.urandom(4))).replace("'", '')
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    makedirs(checkpoint_dir)
    log_path = os.path.join(output_path, 'logs')
    makedirs(log_path)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25)
    results = []
    best_rmse_ind = 0

    for repeat_id in range(n_repeats):
        checkpoint_file = os.path.join(checkpoint_dir,
                                       'davis_dtc_dta_' + param_name + 'repeat' + str(repeat_id) + '.h5')
        checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_loss', mode='min', verbose=1,
                                              save_best_only=True)
        K.clear_session()
        XDinput, encoded_smiles = smiles_encoding_fn()
        XTinput, encoded_protein = protein_encoding_fn()

        model = DTIModel([XDinput, XTinput], encoded_smiles, encoded_protein, smi_model(), seq_model(), interaction_model())
        compiled_model = model.compile(optimizer=optimizer, loss=loss)

        gridres = compiled_model.fit(([XD_train, XT_train]), Y_train, batch_size=batch_size,
                      epochs=num_epoch,
                  validation_data=(([XD_val, XT_val]), np.array(Y_val))
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

        logging('--REPEAT' +str(repeat_id) + '--\n' + str(results[-1]) + '\n----\n', log_path)

        if results[best_rmse_ind]['test_rmse'] > results[repeat_id]['test_rmse']:
            best_rmse_ind = repeat_id

    return results[best_rmse_ind]

def run_experiment(_run, FLAGS):
    log_path = os.path.join(FLAGS.output_path, 'logs')

    FLAGS = Namespace(**vars(FLAGS))
    logging("Data is loading", log_path=log_path)
    data_per_task = load_data(FLAGS)
    logging("Data is loaded", log_path=log_path)

    encoded_smiles = auto_drug_encoding(smi_input_dim=FLAGS.max_smi_len, **vars(FLAGS))
    encoded_protein = auto_protein_encoding(seq_input_dim=FLAGS.max_seq_len, **vars(FLAGS))

    FLAGS.smi_model = auto_model(FLAGS.smi_model, kernel_size=FLAGS.smi_window_length, name='smi_enc', **vars(FLAGS))
    FLAGS.seq_model = auto_model(FLAGS.seq_model, kernel_size=FLAGS.seq_window_length, name='seq_enc', **vars(FLAGS))
    FLAGS.interaction_model = auto_model('fully_connected', name='interaction_fc',
                                         kernel_regularizer=regularizers.l2(FLAGS.l2_regularizer_fc),
                                         **vars(FLAGS))
    if len(data_per_task.keys()) > 1:
        train_fn = train_multitask_model_v2
    else:
        train_fn = train_model


    results = []
    logging("Folds are being created", log_path=log_path)
    for fold_id in range(5):
        splitting_per_task = {}

        for task, dataset in data_per_task.items():
            splitting_per_task[task] = train_val_test_split(fold_id=fold_id, seed=FLAGS.seed,
                                                            val_size=0.2 if task=='Kd' else 0.1,
                                                            n_splits=10 if task=='Kd' else 20,
                                                            **dataset)

        results.append(train_fn(splitting_per_task, encoded_smiles, encoded_protein, fold_id=fold_id, **vars(FLAGS)))
        K.clear_session()
    logging("Folds are created", log_path=log_path)

    mean_dict = {}
    std_dict = {}


    for key in results[0].keys():
        if 'test' not in key:
            continue

        cur_result = np.asarray([val[key] for val in results])

        mean_dict[key] = np.mean(cur_result)
        std_dict[key] = np.std(cur_result)

    logging('---BEST RUN test results---', log_path=log_path)

    for metric in mean_dict.keys():
        result_mean, result_std = mean_dict[metric], std_dict[metric]
        if type(result_mean) == np.ndarray:
            result_mean= result_mean[0]
        if type(result_std) == np.ndarray:
            result_std = result_std[0]

        _run.log_scalar(metric+'_mean', result_mean)
        _run.log_scalar(metric+'_std', result_std)

        logging(metric+'_mean' + '=' + str(result_mean), log_path=log_path)

    #_run.add_artifact(results['checkpoint_file'], 'model_file')

    #for metric, vals in results['train_val_hist'].items():
    #    prefix = 'train_'
    #    if 'val' in metric:
    #        prefix = 'val_'

    #    for i, val in enumerate(vals):
    #        _run.log_scalar(prefix+metric, val, step=i)
