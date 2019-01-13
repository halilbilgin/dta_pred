from keras.layers import Dense
from keras.models import Model

class MultiTaskModel():
    def __init__(self, inputs, shared_layers, task_specific_layers, tasks):
        self.inputs = inputs
        self.shared_model = shared_layers

        self.models = dict()

        for task_name in tasks:
            model = task_specific_layers[task_name](shared_layers)
            output = Dense(1, kernel_initializer='normal')(model)
            self.models[task_name] = output

    def compile(self, optimizers, losses, metrics=['cindex']):
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