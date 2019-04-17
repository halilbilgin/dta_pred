from keras.layers import Dense
from keras.models import Model
import numpy as np

class MultiTaskModel():
    def __init__(self, inputs, shared_layers, task_specific_layers, tasks):
        self.inputs = inputs
        self.shared_model = shared_layers
        self.tasks = tasks

        self.models = dict()

        for task_name in tasks:
            model = task_specific_layers[task_name](shared_layers)
            output = Dense(1, kernel_initializer='normal')(model)
            self.models[task_name] = output

    def compile(self, optimizers, losses, metrics=['spearmanr_corr']):
        if type(optimizers) is not dict:
            optimizers = {key:optimizers for key in self.models.keys()}

        if type(losses) is not dict:
            losses = {key:losses for key in self.models.keys()}

        assert len(losses.keys()) == len(self.models.keys())
        assert len(optimizers.keys()) == len(self.models.keys())

        self.compiled_models = {}
        for task, model in self.models.items():
            compiled_model = Model(inputs=self.inputs, outputs=[model])
            compiled_model.compile(optimizer=optimizers[task], loss=losses[task], metrics=metrics)
            self.compiled_models[task] = compiled_model

        return self.compiled_models

    def train(self, datasets, checkpoint_callbacks, num_epoch, batch_size, **kwargs):
        for i in range(num_epoch):
            for task in self.tasks:
                XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, Y_train, Y_val, Y_test = datasets[task]

                gridres = self.compiled_models[task].fit(([XD_train, XT_train]), Y_train, batch_size=batch_size, epochs=1,
                                           validation_data=(([np.array(XD_val), np.array(XT_val)]), np.array(Y_val)),
                                           callbacks=[checkpoint_callbacks[task]], **kwargs)
