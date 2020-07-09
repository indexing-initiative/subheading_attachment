import numpy as np
import os
import tensorflow.keras.backend as K
from   tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from   tensorflow.keras.layers import (Activation, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, 
                                       Input, Lambda, Layer, MaxPooling1D)
from   tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.models
from   tensorflow.keras.optimizers import Adam
from   tensorflow.keras.utils import plot_model


class EmbeddingWithDropout(Embedding):

    def __init__(self, dropout_rate, *args, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        _embeddings = K.in_train_phase(K.dropout(self.embeddings, self.dropout_rate, noise_shape=[self.input_dim,1]), self.embeddings) if self.dropout_rate > 0 else self.embeddings
        out = K.gather(_embeddings, inputs)
        return out

    def get_config(self):
        config = { 'dropout_rate': self.dropout_rate }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model:

    def build(self, model_config):

        word_embedding_layer = self._word_embedding_layer(model_config)
        conv_layers = self._create_conv_layers(model_config)

        title_input = Input(shape=(model_config.title_max_words,), name='title_input')
        batch_size = K.int_shape(title_input)[0]
        title_word_embeddings = word_embedding_layer(title_input)
        title_features = self._text_feature_extraction(model_config, conv_layers, title_word_embeddings, 1)
        
        abstract_input = Input(shape=(model_config.abstract_max_words,), name='abstract_input')
        abstract_word_embeddings = word_embedding_layer(abstract_input)
        abstract_features = self._text_feature_extraction(model_config, conv_layers, abstract_word_embeddings, model_config.num_pool_regions)

        pub_year_input, pub_year = self._create_time_period_input(model_config.num_pub_year_time_periods, 
                                                                  model_config.pub_year_dropout_rate, 
                                                                  'pub_year_input', 
                                                                  batch_size)
        year_completed_input, year_completed = self._create_time_period_input(model_config.num_year_completed_time_periods, 
                                                                              model_config.year_completed_dropout_rate, 
                                                                              'year_completed_input', 
                                                                              batch_size)

        journal_input, journal_embedding = self._journal_embedding(model_config.num_journals, 
                                                                   model_config.journal_embedding_size, 
                                                                   model_config.journal_dropout_rate)
   
        desc_input = Input(shape=(model_config.num_desc,), name='desc_input')

        hidden = Concatenate()([title_features, abstract_features, pub_year, year_completed, journal_embedding, desc_input])
        for layer_size in model_config.hidden_layer_sizes:
            hidden = Dense(layer_size, activation=None, use_bias=False)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation(model_config.hidden_layer_act)(hidden)
            hidden = Dropout(model_config.dropout_rate)(hidden)

        output = Dense(model_config.output_layer_size, activation=model_config.output_layer_act, name='labels')(hidden)
 
        model = tensorflow.keras.models.Model(inputs=[title_input, abstract_input, pub_year_input, year_completed_input, journal_input, desc_input], 
                                              outputs=[output]) 

        loss = binary_crossentropy
        optimizer = Adam(lr=model_config.init_learning_rate)
        metrics = [MicroFScore(model_config.init_threshold)]
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self._model = model

    def fit(self, root_config, training_data, dev_data, output_dir):
        train_config = root_config.train
    
        # Ensure output dir exists
        self._mkdir(output_dir)
        
        # Save config
        settings_filepath = os.path.join(output_dir, train_config.save_config.settings_filename)
        model_json_filepath = os.path.join(output_dir, train_config.save_config.model_json_filename)
        model_img_filepath = os.path.join(output_dir, train_config.save_config.model_img_filename)
        with open(settings_filepath, 'wt', encoding=train_config.save_config.encoding) as settings_file:
            settings_file.write(str(root_config))
        with open(model_json_filepath, 'wt', encoding=train_config.save_config.encoding) as model_json_file:
            model_json_file.write(self._model.to_json(indent=4))
        plot_model(self._model, model_img_filepath, show_shapes=True, show_layer_names=True, rankdir='TB')

        callbacks = []

        # Optimize fscore threshold
        if train_config.optimize_fscore_threshold.enabled:
            optimize_fscore_threshold_callback = OptimizeFscoreThresholdCallback(train_config.optimize_fscore_threshold, dev_data)
            callbacks.append(optimize_fscore_threshold_callback)

        # Save checkpoints
        checkpoint_config = root_config.model.checkpoint
        if checkpoint_config.enabled:
            checkpoint_dir = os.path.join(output_dir, checkpoint_config.dir)
            self._mkdir(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_config.filename)
            checkpoint_callback = ModelCheckpoint(checkpoint_path, 
                                                  monitor=train_config.monitor_metric, 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  mode=train_config.monitor_mode, 
                                                  save_weights_only=checkpoint_config.weights_only)
            callbacks.append(checkpoint_callback)

        # Terminate on NaN
        callbacks.append(TerminateOnNaN())

        # Reduce learning rate
        callbacks.append(ReduceLROnPlateau(train_config.monitor_metric, 
                                           factor=train_config.reduce_learning_rate.factor, 
                                           patience=train_config.reduce_learning_rate.patience, 
                                           verbose=0, 
                                           mode=train_config.monitor_mode, 
                                           min_delta=train_config.reduce_learning_rate.min_delta, 
                                           cooldown=0, 
                                           min_lr=0))

        # Early stopping
        callbacks.append(EarlyStopping(train_config.monitor_metric, 
                                       min_delta=train_config.early_stopping.min_delta, 
                                       patience=train_config.early_stopping.patience, 
                                       verbose=0, 
                                       mode=train_config.monitor_mode))
        
        # CSV logger
        csv_dir = os.path.join(output_dir, train_config.csv_logger.dir)
        self._mkdir(csv_dir)
        csv_path = os.path.join(csv_dir, train_config.csv_logger.filename)
        callbacks.append(CSVLogger(filename=csv_path))

        history = self._model.fit_generator(training_data, 
                                            epochs=train_config.max_epochs, 
                                            verbose=1, 
                                            callbacks=callbacks, 
                                            validation_data=dev_data, 
                                            shuffle=True, 
                                            initial_epoch=train_config.initial_epoch, 
                                            use_multiprocessing=train_config.use_multiprocessing, 
                                            workers=train_config.workers, 
                                            max_queue_size=train_config.max_queue_size)
        logs = history.history

        monitor_mode_func = globals()['__builtins__'][train_config.monitor_mode]
        monitor_metric_values = logs[train_config.monitor_metric]
        best_epoch_index = monitor_metric_values.index(monitor_mode_func(monitor_metric_values))
        best_epoch_logs = { key: logs[key][best_epoch_index] for key in logs.keys() }
        best_epoch_logs['best epoch'] = best_epoch_index 

        best_epoch_logs_txt = ' '.join(['{}: {:.9f}'.format(name, value) for name, value in sorted(best_epoch_logs.items(), key=lambda x:x[0])])
        print(best_epoch_logs_txt)

        best_epoch_logs_filepath = os.path.join(output_dir, train_config.csv_logger.dir, train_config.csv_logger.best_epoch_filename)
        with open(best_epoch_logs_filepath, 'wt', encoding=train_config.csv_logger.encoding) as best_epoch_logs_file:
            best_epoch_logs_file.write(best_epoch_logs_txt)

        return best_epoch_logs

        def _create_conv_layers(self, model_config):
        conv_layers = []
        for filter_size in model_config.conv_filter_sizes:
            conv_layer = Conv1D(model_config.conv_num_filters, filter_size, activation=None, padding='valid', strides=1, use_bias=False)
            conv_layers.append(conv_layer)
        return conv_layers

    def _create_time_period_input(self, num_time_periods, dropout_rate, name, batch_size):
        time_period_input = Input(shape=(num_time_periods,), name=name)
        time_period = Dropout(dropout_rate, noise_shape=[batch_size, 1])(time_period_input)
        return time_period_input, time_period

    def _journal_embedding(self, num_journals, journal_embedding_size, dropout_rate):
        journal_input = Input(shape=(1,), name='journal_input')
        journal_embedding = EmbeddingWithDropout(dropout_rate, num_journals, journal_embedding_size, trainable=True)(journal_input)
        journal_embedding = Flatten()(journal_embedding)
        return journal_input, journal_embedding

    def _mkdir(self, dir):
        if not os.path.isdir(dir): 
            os.mkdir(dir) 

    def _text_feature_extraction(self, model_config, conv_layers, word_embeddings, num_pool_regions):
        conv_blocks = []
        for conv_layer in conv_layers:
            conv = conv_layer(word_embeddings)
            conv = BatchNormalization()(conv)
            conv = Activation(model_config.conv_act)(conv)
            pool_size = K.int_shape(conv)[1] // num_pool_regions
            conv = MaxPooling1D(pool_size=pool_size, strides=pool_size, padding='valid')(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        text_features = Dropout(model_config.dropout_rate)(concat)
        return text_features

    def _word_embedding_layer(self, model_config):
        word_embedding_layer = EmbeddingWithDropout(model_config.word_embedding_dropout_rate, 
                                                    model_config.vocab_size, 
                                                    model_config.word_embedding_size, 
                                                    trainable=True)
        return word_embedding_layer


class MicroFScore(Layer):
    def __init__(self, threshold, b=1, name="fscore"):
        super().__init__(name=name)
        self.__name__ = name
        self.threshold = K.variable(value=threshold, dtype='float32') 
        self.b_squared = b*b
        self.true_positive_count = K.variable(value=0, dtype='int32') 
        self.pred_positive_count = K.variable(value=0, dtype='int32')
        self.act_positive_count =  K.variable(value=0, dtype='int32')
        self.stateful = True

    def reset_states(self):
        K.set_value(self.true_positive_count, 0)
        K.set_value(self.pred_positive_count, 0)
        K.set_value(self.act_positive_count,  0)

    def __call__(self, y_true, y_pred):
        # Batch
        y_act = K.cast(y_true, 'int32')
        y_pred_th = K.cast(y_pred >= self.threshold, 'int32')

        batch_true_positive_count =  K.cast(K.sum(y_act * y_pred_th), 'int32')
        batch_pred_positive_count =  K.cast(K.sum(y_pred_th), 'int32')
        batch_act_positive_count  =  K.cast(K.sum(y_act), 'int32')

        # Prev
        prev_true_positive_count = self.true_positive_count * 1
        prev_pred_positive_count = self.pred_positive_count * 1
        prev_act_positive_count  = self.act_positive_count * 1

        # Updates
        updates = [K.update_add(self.true_positive_count, batch_true_positive_count),
                    K.update_add(self.pred_positive_count, batch_pred_positive_count),
                    K.update_add(self.act_positive_count, batch_act_positive_count)]
        self.add_update(updates, inputs=[y_true, y_pred])

        # Compute Fscore
        current_true_positive_count = K.cast(prev_true_positive_count + batch_true_positive_count, 'float32')
        current_pred_positive_count = K.cast(prev_pred_positive_count + batch_pred_positive_count, 'float32')
        current_act_positive_count =  K.cast(prev_act_positive_count + batch_act_positive_count  , 'float32')
        
        current_precision = current_true_positive_count / (current_pred_positive_count + K.epsilon())
        current_recall =    current_true_positive_count / (current_act_positive_count + K.epsilon())
        current_fscore = (1 + self.b_squared)*current_precision*current_recall/((self.b_squared*current_precision) + current_recall + K.epsilon())

        return current_fscore


class OptimizeFscoreThresholdCallback(Callback):

    def __init__(self, config, opt_data):
        super().__init__()
        self.config = config
        self.opt_data = opt_data

    def set_model(self, model):
        super().set_model(model)
        for idx, metric in enumerate(self.model.metrics):
            if metric.__name__ == self.config.metric_name:
                self.threshold = metric.threshold
                self.metric_index = idx + 1

    def on_batch_end(self, batch, logs=None):
        step = batch + 1
        if step == self.params['steps']: # Have finished the last batch
            self.prev_threshold_value = K.get_value(self.threshold)
            alpha = self.config.alpha
            k = self.config.k
            candidate_thresholds = [(self.prev_threshold_value - (alpha*k)) + (x*alpha) for x in range(2*k + 1)]
            best_metric_value = 0
            best_threshold = self.prev_threshold_value
            for candidate_threshold in candidate_thresholds:
                K.set_value(self.threshold, candidate_threshold)
                metrics = self.model.evaluate_generator(self.opt_data, 
                                                        use_multiprocessing=self.config.use_multiprocessing, 
                                                        workers=self.config.workers, 
                                                        max_queue_size = self.config.max_queue_size)
                metric_value = metrics[self.metric_index]
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = candidate_threshold
            K.set_value(self.threshold, best_threshold)
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['threshold'] = self.prev_threshold_value
        logs['val_threshold'] = K.get_value(self.threshold)