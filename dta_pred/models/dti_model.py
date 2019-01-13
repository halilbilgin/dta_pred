from dta_pred.models.dnn_model import *
class DTIModel():
    def __init__(self, inputs, encode_smiles, encode_protein, smi_model, seq_model,
                 interaction_model):
        self.inputs = inputs

        self.encode_smiles = encode_smiles
        self.smi_module = smi_model(encode_smiles)
        self.smi_module = GlobalMaxPooling1D()(self.smi_module)

        self.encode_protein = encode_protein
        self.seq_module = seq_model(encode_protein)
        self.seq_module = GlobalMaxPooling1D()(self.seq_module)

        self.encode_interaction = concatenate([self.smi_module, self.seq_module], name='encode_interaction', axis=-1)

        self.interaction_module = interaction_model(self.encode_interaction)

        self.output = Dense(1, kernel_initializer='normal')(self.interaction_module)

    def compile(self, optimizer='adam', loss='mean_squared_error', metrics=[cindex, f1]):
        interactionModel = Model(inputs=self.inputs, outputs=[self.output])

        interactionModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return interactionModel