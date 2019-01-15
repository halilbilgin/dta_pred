from keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model
import tensorflow as tf
from dta_pred.metrics import cindex, f1
import keras.metrics
keras.metrics.cindex = cindex
keras.metrics.f1 = f1
from keras import regularizers

from keras import backend as K

from dta_pred import CHARISOSMILEN, CHARPROTLEN

def simple_cnn_encoder(n_cnn_layers, num_windows, kernel_size, name, **kwargs):

    def simple_cnn_encoder_base(inputs):
        encode_mols = inputs

        for i in range(n_cnn_layers):
            encode_mols = Conv1D(filters=num_windows, kernel_size=kernel_size,
                                 activation='relu', padding='valid', strides=1,
                                 name=name+'_'+str(i+1))(encode_mols)

        return encode_mols

    return simple_cnn_encoder_base

def inception_encoder(num_windows, kernel_size, name, **kwargs):
    def inception_base(inputs):
        name_prefix = name + '_'

        tower_one = MaxPooling1D(kernel_size, strides=1, padding='same',
                                 name=name_prefix + '_1')(inputs)
        tower_one = Conv1D(num_windows, kernel_size, activation='relu', padding='same',
                           name=name_prefix + '_2')(tower_one)

        tower_two = Conv1D(num_windows, kernel_size * 2, activation='relu', padding='same',
                           name=name_prefix + '_3')(inputs)
        tower_two = Conv1D(num_windows, kernel_size * 3, activation='relu', padding='same',
                           name=name_prefix + '_4')(tower_two)

        tower_three = Conv1D(num_windows, kernel_size * 2, activation='relu',
                             padding='same', name=name_prefix + '_5')(inputs)
        tower_three = Conv1D(num_windows, kernel_size * 4, activation='relu',
                             padding='same', name=name_prefix + '_6')(tower_three)

        x = concatenate([tower_one, tower_two, tower_three], axis=2)

        return x

    return inception_base


def fully_connected_model(n_fc_layers, n_fc_neurons, dropout, name='fc',
                          kernel_regularizer=None, apply_bn=False, **kwargs):
    def fully_connected_model_base(encode_interaction):
        FC = encode_interaction
        for i in range(n_fc_layers):
            FC = Dense(n_fc_neurons, activation='relu',
                       kernel_regularizer=kernel_regularizer, name=name+'_'+str(i))(FC)
            FC = Dropout(dropout)(FC)
            if apply_bn:
                FC = BatchNormalization()(FC)

        return FC

    return fully_connected_model_base

def crossentropy_mse_combined(y_true, y_pred):
    loss = keras.losses.mean_squared_error(y_true, y_pred)
    loss += keras.losses.binary_crossentropy(tf.dtypes.cast(y_true>7, tf.float32),
                                             keras.activations.sigmoid(y_pred-7))

    return loss

def auto_model(model_type, **kwargs):
    if model_type == 'fully_connected':
        return lambda : fully_connected_model(**kwargs)
    elif model_type == 'inception':
        return lambda : inception_encoder(**kwargs)
    elif model_type == 'simple_cnn':
        return lambda : simple_cnn_encoder(**kwargs)
    else:
        raise NotImplementedError()

def get_pooling(pooling_type):
    if pooling_type == 'GlobalMaxPooling':
        output = GlobalMaxPooling1D()
    elif pooling_type == 'GlobalAveragePooling':
        output = GlobalAveragePooling1D()
    else:
        raise NotImplementedError()

    return output

keras.losses.crossentropy_mse_combined = crossentropy_mse_combined
