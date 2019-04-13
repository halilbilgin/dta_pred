from dta_pred.models.dnn_model import *

class DTIModel():
    """Drug Target Interaction Model"""
    def __init__(self, inputs, encoded_smiles, encoded_protein, smi_layers, seq_layers,
                 interaction_model):
        """
        The constructor takes encoded smiles and proteins, as well as the smiles, protein and interaction layers
        then creates a drug target interaction model with a single output that returns the interaction property(e.g.
        affinity)

        :param inputs: list that takes Input instances from keras e.g. [XTinput, XDinput]
        :param encoded_smiles: an object that encodes a SMILES into an ndarray containing continuous values
        :param encoded_protein: an object that encodes a proteın ınto an ndarray containing continuous values
        :param smi_layers: Layers that takes encoded SMILES and returns a 1d output for interaction module
        :param seq_layers: Layers that takes encoded protein and returns a 1d output for interaction module
        :param interaction_model: Takes concatenation of the outputs of smi_layers and seq_layers and outputs a 1d output

        """
        self.inputs = inputs

        self.encode_smiles = encoded_smiles
        self.smi_module = smi_layers(encoded_smiles)
        self.smi_module = GlobalMaxPooling1D()(self.smi_module)

        self.encode_protein = encoded_protein
        self.seq_module = seq_layers(encoded_protein)
        self.seq_module = GlobalMaxPooling1D()(self.seq_module)

        self.encode_interaction = concatenate([self.smi_module, self.seq_module], name='encode_interaction', axis=-1)

        self.interaction_module = interaction_model(self.encode_interaction)

        self.output = Dense(1, kernel_initializer='normal')(self.interaction_module)

    def compile(self, optimizer='adam', loss='mean_squared_error', metrics=[cindex, f1]):
        """
        Compiles and returns the training-ready model.

        :param optimizer:  Name of an optimizer implemented in keras.optimizers or a callable
        :param loss: Name of a loss implemented in keras.losses or a callable
        :param metrics: A list of metrics
        :return: Model instance
        """
        interactionModel = Model(inputs=self.inputs, outputs=[self.output])

        interactionModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return interactionModel