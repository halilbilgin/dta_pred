from keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model
import tensorflow as tf
from dta_pred.emetrics import cindex, f1
import keras.metrics
keras.metrics.cindex = cindex
keras.metrics.f1  = f1

from dta_pred import CHARISOSMILEN, CHARPROTLEN

def smiles_encoder(FLAGS):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')

    encode_smiles = Embedding(input_dim=CHARISOSMILEN + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    for i in range(FLAGS.n_cnn_layers):
        encode_smiles = Conv1D(filters=FLAGS.num_windows, kernel_size=FLAGS.smi_window_length,
                               activation='relu', padding='valid', strides=1)(encode_smiles)

    if FLAGS.pooling_type == 'GlobalMaxPooling':
        encode_smiles = GlobalMaxPooling1D()(encode_smiles)
    elif FLAGS.pooling_type == 'GlobalAveragePooling':
        encode_smiles = GlobalAveragePooling1D()(encode_smiles)
    else:
        raise NotImplementedError()

    return XDinput, encode_smiles

def pssm_encoder(FLAGS):
    XTinput = Input(shape=(FLAGS.max_seq_len, 20), dtype='float32')
    encode_protein = XTinput
    for i in range(FLAGS.n_cnn_layers):
        encode_protein = Conv1D(filters=FLAGS.num_windows, kernel_size=FLAGS.seq_window_length, activation='relu',
                                padding='valid', strides=1)(encode_protein)

    if FLAGS.pooling_type == 'GlobalMaxPooling':
        encode_protein = GlobalMaxPooling1D()(encode_protein)
    elif FLAGS.pooling_type == 'GlobalAveragePooling':
        encode_protein = GlobalAveragePooling1D()(encode_protein)
    else:
        raise NotImplementedError()

    return XTinput, encode_protein

def seq_encoder(FLAGS):
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    encode_protein = Embedding(input_dim=CHARPROTLEN + 1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)

    for i in range(FLAGS.n_cnn_layers):
        encode_protein = Conv1D(filters=FLAGS.num_windows, kernel_size=FLAGS.seq_window_length, activation='relu',
                                padding='valid', strides=1)(encode_protein)

    if FLAGS.pooling_type == 'GlobalMaxPooling':
        encode_protein = GlobalMaxPooling1D()(encode_protein)
    elif FLAGS.pooling_type == 'GlobalAveragePooling':
        encode_protein = GlobalAveragePooling1D()(encode_protein)
    else:
        raise NotImplementedError()

    return XTinput, encode_protein

def standard_model(FLAGS):
    if FLAGS.drug_format == 'labeled_smiles':
        XDInput, encode_smiles = smiles_encoder(FLAGS)
    elif FLAGS.drug_format == 'mol2vec':
        XDinput = Input(shape=(FLAGS.mol2vec_output_dim, ), )
        encode_smiles = XDinput
    else:
        raise NotImplementedError()

    if FLAGS.protein_format == 'sequence':
        XTinput, encode_protein = seq_encoder(FLAGS)
    elif FLAGS.protein_format == 'pssm':
        XTinput, encode_protein = pssm_encoder(FLAGS)
    else:
        raise NotImplementedError()

    encode_interaction = concatenate([encode_smiles, encode_protein], axis=-1)

    # Fully connected
    FC = encode_interaction
    for i in range(FLAGS.n_fc_layers):
        FC = Dense(FLAGS.n_neurons_fc, activation='relu')(encode_interaction)

        FC = Dropout(FLAGS.dropout)(FC)
        if FLAGS.apply_bn:
            FC = BatchNormalization()(FC)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])


    interactionModel.compile(optimizer='adam', loss=FLAGS.loss, metrics=[cindex, f1])

    return interactionModel


def crossentropy_mse_combined(y_true, y_pred):
    loss = keras.losses.mean_squared_error(y_true, y_pred)
    loss += keras.losses.binary_crossentropy(tf.dtypes.cast(y_true>7, tf.float32),
                                             keras.activations.sigmoid(y_pred-7))

    return loss


keras.losses.crossentropy_mse_combined = crossentropy_mse_combined
