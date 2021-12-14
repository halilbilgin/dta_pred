from keras.layers import Input, Embedding
from dta_pred.data_helper import CHARISOSMILEN


def smiles_encoding(smi_input_dim, max_smi_len, **kwargs):
    XDinput = Input(shape=(smi_input_dim,), dtype="int32")
    encoded_smiles = Embedding(
        input_dim=CHARISOSMILEN + 1,
        output_dim=128,
        input_length=max_smi_len,
        name="smi_embedding",
    )(XDinput)

    return XDinput, encoded_smiles


def mol2vec_encoding(smi_input_dim, **kwargs):
    XDinput = Input(
        shape=(smi_input_dim,),
    )
    encoded_smiles = XDinput

    return XDinput, encoded_smiles


def auto_drug_encoding(drug_format, **kwargs):
    if drug_format == "mol2vec":
        return lambda: mol2vec_encoding(**kwargs)
    elif drug_format == "labeled_smiles":
        return lambda: smiles_encoding(**kwargs)
    else:
        raise NotImplementedError()
