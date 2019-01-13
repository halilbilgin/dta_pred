from keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model
import tensorflow as tf
from dta_pred.emetrics import cindex, f1
import keras.metrics
keras.metrics.cindex = cindex
keras.metrics.f1 = f1
from keras import backend as K

from dta_pred import CHARISOSMILEN, CHARPROTLEN

def simple_cnn_encoder(n_cnn_layers, num_windows, kernel_size, name):

    def simple_cnn_encoder_base(inputs):
        encode_mols = inputs

        for i in range(n_cnn_layers):
            encode_mols = Conv1D(filters=num_windows, kernel_size=kernel_size,
                                 activation='relu', padding='valid', strides=1,
                                 name=name+'_'+str(i+1))(encode_mols)

        return encode_mols

    return simple_cnn_encoder_base

def inception_encoder(num_windows, kernel_size, name):
    def inception_base(inputs):
        name_prefix = name+'_'

        tower_one = MaxPooling1D(kernel_size, strides=1, padding='same',
                                 name=name_prefix+'_1')(inputs)
        tower_one = Conv1D(num_windows, kernel_size, activation='relu', padding='same',
                                 name=name_prefix+'_2')(tower_one)

        tower_two = Conv1D(num_windows, kernel_size*2, activation='relu', padding='same',
                                 name=name_prefix+'_3')(inputs)
        tower_two = Conv1D(num_windows, kernel_size*3, activation='relu', padding='same',
                                 name=name_prefix+'_4')(tower_two)

        tower_three = Conv1D(num_windows, kernel_size*2, activation='relu',
                             padding='same', name=name_prefix+'_5')(inputs)
        tower_three = Conv1D(num_windows, kernel_size*4, activation='relu',
                             padding='same', name=name_prefix+'_6')(tower_three)

        x = concatenate([tower_one, tower_two, tower_three], axis=2)

        return x

    return inception_base

def fully_connected_model(n_fc_layers, n_neurons, dropout, name='fc', apply_bn=False):
    def fully_connected_model_base(encode_interaction):
        FC = encode_interaction
        for i in range(n_fc_layers):
            FC = Dense(n_neurons, activation='relu', name=name+'_'+str(i))(FC)
            FC = Dropout(dropout)(FC)
            if apply_bn:
                FC = BatchNormalization()(FC)

        return FC

    return fully_connected_model_base

def get_cnn_output(drug_format, protein_format, smi_input_dim, seq_input_dim,
              smi_model=None, seq_model=None,
              seq_embedding=None, smi_embedding=None, smi_pooling=GlobalMaxPooling1D(),
              seq_pooling=GlobalMaxPooling1D()):
    if drug_format == 'labeled_smiles':
        XDinput = Input(shape=(smi_input_dim,), dtype='int32')
        encode_smiles = smi_embedding(XDinput)
    elif drug_format == 'mol2vec':
        XDinput = Input(shape=(smi_input_dim, ), )
        encode_smiles = XDinput
    else:
        raise NotImplementedError()

    if smi_model is not None:
        encode_smiles = smi_model(encode_smiles)
        encode_smiles = smi_pooling(encode_smiles)

    if protein_format == 'sequence':
        XTinput = Input(shape=(seq_input_dim,), dtype='int32')
        encode_protein = seq_embedding(XTinput)
    elif protein_format == 'pssm':
        XTinput = Input(shape=(seq_input_dim, 20), dtype='float32')
        encode_protein = XTinput
    elif protein_format == 'biovec':
        XTinput = Input(shape=(seq_input_dim,), )
        encode_protein = XTinput
    else:
        raise NotImplementedError()

    if seq_model is not None:
        encode_protein = seq_model(encode_protein)
        encode_protein = seq_pooling(encode_protein)

    encode_interaction = concatenate([encode_smiles, encode_protein], name='encode_interaction', axis=-1)

    return [XDinput, XTinput], encode_interaction

def dnn_model(drug_format, protein_format, smi_input_dim, seq_input_dim,
              interaction_model=None, loss_fn='mean_squared_error', smi_model=None, seq_model=None,
              seq_embedding=None, smi_embedding=None, smi_pooling=GlobalMaxPooling1D(),
              seq_pooling=GlobalMaxPooling1D()):

    inputs, encode_interaction = get_cnn_output(drug_format, protein_format, smi_input_dim, seq_input_dim,
                                        smi_model, seq_model, seq_embedding, smi_embedding, smi_pooling, seq_pooling)

    # Fully connected
    FC = encode_interaction
    if interaction_model is not None:
        FC = interaction_model(encode_interaction)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC)

    interactionModel = Model(inputs=inputs, outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss=loss_fn, metrics=[cindex, f1])

    return interactionModel

def crossentropy_mse_combined(y_true, y_pred):
    loss = keras.losses.mean_squared_error(y_true, y_pred)
    loss += keras.losses.binary_crossentropy(tf.dtypes.cast(y_true>7, tf.float32),
                                             keras.activations.sigmoid(y_pred-7))

    return loss


def get_pooling(pooling_type):
    if pooling_type == 'GlobalMaxPooling':
        output = GlobalMaxPooling1D()
    elif pooling_type == 'GlobalAveragePooling':
        output = GlobalAveragePooling1D()
    else:
        raise NotImplementedError()

    return output

keras.losses.crossentropy_mse_combined = crossentropy_mse_combined
