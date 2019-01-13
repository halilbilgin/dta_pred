from keras.layers import Input, Embedding
from dta_pred.datahelper import CHARPROTLEN

def biovec_encoding(seq_input_dim, **kwargs):
    XTinput = Input(shape=(seq_input_dim,), )
    encode_protein = XTinput

    return XTinput, encode_protein

def pssm_encoding(seq_input_dim, **kwargs):
    XTinput = Input(shape=(seq_input_dim, 20), dtype='float32')
    encode_protein = XTinput
    return XTinput, encode_protein

def sequence_encoding(seq_input_dim, max_seq_len, **kwargs):
    XTinput = Input(shape=(seq_input_dim,), dtype='int32')
    encode_protein = Embedding(input_dim=CHARPROTLEN + 1, output_dim=128,
                              input_length=max_seq_len, name='seq_embedding')(XTinput)

    return XTinput, encode_protein

def auto_protein_encoding(protein_format, **kwargs):
    if protein_format == 'biovec':
        return lambda : biovec_encoding(**kwargs)
    elif protein_format == 'pssm':
        return lambda : pssm_encoding(**kwargs)
    elif protein_format == 'sequence':
        return lambda : sequence_encoding(**kwargs)
    else:
        raise NotImplementedError()