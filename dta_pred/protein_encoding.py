from keras.layers import Input, Embedding
from dta_pred.data_helper import CHARPROTLEN

def biovec_encoding(seq_input_dim, **kwargs):
    XTinput = Input(shape=(seq_input_dim,), )
    encoded_protein = XTinput

    return XTinput, encoded_protein

def pssm_encoding(seq_input_dim, **kwargs):
    XTinput = Input(shape=(seq_input_dim, 20), dtype='float32')
    encoded_protein = XTinput
    return XTinput, encoded_protein

def sequence_encoding(seq_input_dim, max_seq_len, **kwargs):
    XTinput = Input(shape=(seq_input_dim,), dtype='int32')
    encoded_protein = Embedding(input_dim=CHARPROTLEN + 1, output_dim=128,
                              input_length=max_seq_len, name='seq_embedding')(XTinput)

    return XTinput, encoded_protein

def auto_protein_encoding(protein_format, **kwargs):
    if protein_format == 'biovec':
        return lambda : biovec_encoding(**kwargs)
    elif protein_format == 'pssm':
        return lambda : pssm_encoding(**kwargs)
    elif protein_format == 'sequence':
        return lambda : sequence_encoding(**kwargs)
    else:
        raise NotImplementedError()