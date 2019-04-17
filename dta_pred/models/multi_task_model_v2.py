from keras.layers import Dense
import math
from keras.models import Model
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, datasets, y, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.y = np.asarray(y)
        self.shuffle = shuffle
        self.datasets = datasets
        for i, dataset in enumerate(self.datasets):
            self.datasets[i] = np.asarray(dataset)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.datasets[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.datasets[0].shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        shuffled_datasets = []
        for dataset in self.datasets:
            shuffled_datasets.append(dataset[self.indexes])
        
        self.datasets = shuffled_datasets
        
        self.y = self.y[self.indexes]
        
    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        all_X = []

        for dataset in self.datasets:
            X = dataset[index*self.batch_size:(index+1)*self.batch_size]
            all_X.append(X)
    
        y = self.y[index*self.batch_size:(index+1)*self.batch_size]
        
        return all_X, y

class MultiTaskModelV2():
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
        
        data_generators = {}
        
        for task, dataset in datasets.items():
            XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, Y_train, Y_val, Y_test = datasets[task]
            data_generators[task] = DataGenerator([XD_train, XT_train], Y_train, batch_size)
            checkpoint_callbacks[task].set_model(self.compiled_models[task])
        
        for cur_epoch in range(num_epoch):
            print("Epoch", cur_epoch)

            for task, callback in checkpoint_callbacks.items():
                callback.on_epoch_begin(cur_epoch)
            number_of_steps = int(datasets['Kd'][0].shape[0]/batch_size)
            
            for i in range(number_of_steps):
                for task, callback in checkpoint_callbacks.items():
                    callback.on_batch_begin(i)
        
                task = self.tasks[math.floor(np.random.uniform(0,len(self.tasks)))]
                X, y = data_generators[task][i]
                gridres = self.compiled_models[task].train_on_batch(X, y)
                
                for task, callback in checkpoint_callbacks.items():
                    callback.on_batch_end(i)
            
            results_per_epoch = {}
            for task, dataset in datasets.items():
                XD_train, XD_val, XD_test, XT_train, XT_val, XT_test, Y_train, Y_val, Y_test = dataset
                res = self.compiled_models[task].test_on_batch(([XD_val[:1000], XT_val[:1000]]), Y_val[:1000])
                results_per_epoch[task] = {'val_'+self.compiled_models[task].metrics_names[i]:score
                                           for i, score in enumerate(res)}
            print(results_per_epoch['Kd'])
            for task, callback in checkpoint_callbacks.items():
                callback.on_epoch_end(cur_epoch, results_per_epoch[task])
            
            for task, data_generator in data_generators.items():
                data_generator.on_epoch_end()
                