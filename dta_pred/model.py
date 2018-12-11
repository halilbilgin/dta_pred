from keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model

def standard_model(FLAGS):
    if FLAGS.drug_format == 'SMILES':
        XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    elif FLAGS.drug_format == 'VAE_code':
        XDinput = Input(shape=(FLAGS.drug_vae_code_len,), )
    else:
        raise NotImplementedError()

    if FLAGS.protein_format == 'sequence':
        XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    elif FLAGS.protein_format == 'pssm':
        XTinput = Input(shape=(FLAGS.max_seq_len, 20), dtype='float32')
    else:
        raise NotImplementedError()

    if FLAGS.drug_format == 'SMILES':
        encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput)
        for i in range(FLAGS.n_cnn_layers):
            encode_smiles = Conv1D(filters=FLAGS.num_windows, kernel_size=FLAGS.smi_window_length,  activation='relu', padding='valid',  strides=1)(encode_smiles)

        if FLAGS.pooling_type == 'GlobalMaxPooling':
            encode_smiles = GlobalMaxPooling1D()(encode_smiles)
        elif FLAGS.pooling_type == 'GlobalAveragePooling':
            encode_smiles = GlobalAveragePooling1D()(encode_smiles)
        else:
            raise NotImplementedError()
    elif FLAGS.drug_format == 'VAE_code':
        encode_smiles = XDinput

    if FLAGS.protein_format == 'sequence':
        encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    else:
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
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score, tf.contrib.metrics.f1_score])

    return interactionModel