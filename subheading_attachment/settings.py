import json
from .machine_settings import _MachineConfig
import os


def get_config():
    config = Config()
    return config


class _ConfigBase:
    def __init__(self, parent):    
        self._parent = parent
        machine_config = _MachineConfig()
        self._initialize(machine_config)

    def __str__(self):
        dict = {}
        self._toJson(dict, self)
        return json.dumps(dict, indent=4)

    def _initialize(self, machine_config):
        pass
    
    @classmethod
    def _toJson(cls, parent, obj):
        for attribute_name in dir(obj): 
            if not attribute_name.startswith('_'):
                attribute = getattr(obj, attribute_name)
                if isinstance(attribute, _ConfigBase):
                    child = {}
                    parent[attribute_name] = child
                    cls._toJson(child, attribute)
                else:
                    parent[attribute_name] = attribute 


class _TextConfig(_ConfigBase):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)
  
        self.encoding = 'utf8'
        

class _CheckpointConfig(_ConfigBase):
    def _initialize(self, _):
  
        self.enabled = True

        self.dir = 'checkpoints'
        self.filename = 'weights.{epoch:03d}-{val_fscore:.4f}.hdf5'
        self.weights_only = True
 

class _CrossValidationConfig(_TextConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.dev_limit = machine_config.dev_limit
        self.dev_set_ids_path = os.path.join(machine_config.data_dir,   'preprocessed/cross-validation/dev_set_db_ids.txt')
        self.train_limit = machine_config.train_limit
        self.train_set_ids_path = os.path.join(machine_config.data_dir, 'preprocessed/cross-validation/{}.txt')
      
       
class _CsvLoggerConfig(_TextConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.best_epoch_filename = 'best_epoch_logs.txt'
        self.dir = 'logs'
        self.filename  = 'logs.csv'
  

class _DatabaseConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.config = { 'user': '****',
                        'database': 'medline_2019',
                        'password': '****', 
                        'charset': 'utf8mb4', 
                        'collation': 'utf8mb4_unicode_ci', 
                        'use_unicode': True,
                        'host':  machine_config.database_host,
                        'port':  machine_config.database_port,}


class _EarlyStoppingConfig(_ConfigBase):
    def _initialize(self, _):

        self.min_delta = 0.001
        self.patience = 2


class _ModelConfig(_ConfigBase):
    def _initialize(self, _):

        self.checkpoint = _CheckpointConfig(self)

        self.word_embedding_size = 300
        self.word_embedding_dropout_rate = 0.
     
        self.conv_act = 'relu'
        self.num_conv_filter_sizes = 3
        self.min_conv_filter_size = 2
        self.conv_filter_size_step = 3
        self.total_conv_filters = 350
        self.num_pool_regions = 5

        self.num_journals = 31845 + 1
        self.journal_embedding_size = 50
        self.journal_dropout_rate = 0.

        self.pub_year_dropout_rate = 0.
        self.year_completed_dropout_rate = 0.

        self.has_desc_input = False

        self.num_hidden_layers = 1
        self.hidden_layer_size = 2048
        self.hidden_layer_act = 'relu'

        self.dropout_rate = 0.25

        self.output_layer_act = 'sigmoid'

        self.init_threshold = 0.325
        self.init_learning_rate = 0.001

    @property
    def conv_filter_sizes(self):
        sizes = [self.min_conv_filter_size + self.conv_filter_size_step*idx for idx in range(self.num_conv_filter_sizes)]
        return sizes

    @property
    def conv_num_filters(self):
        num_filters = round(self.total_conv_filters / len(self.conv_filter_sizes))
        return num_filters

    @property
    def hidden_layer_sizes(self):
        return [self.hidden_layer_size]*self.num_hidden_layers

    @property
    def _pp_config(self):
        return self._parent.preprocessing

    @property
    def vocab_size(self):
        return self._pp_config.vocab_size

    @property
    def title_max_words(self):
        return self._pp_config.title_max_words

    @property
    def abstract_max_words(self):
        return self._pp_config.abstract_max_words

    @property
    def num_pub_year_time_periods(self):
        return self._pp_config.num_pub_year_time_periods

    @property
    def num_year_completed_time_periods(self):
        return self._pp_config.num_year_completed_time_periods

    @property
    def output_layer_size(self):
        return self._pp_config.num_labels

    @property
    def num_desc(self):
        return self._pp_config.num_desc


class _PreprocessingConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.sentencepiece_model_path = os.path.join(machine_config.data_dir, 'preprocessed/word-embedding/bpe_64000_lc.model')
        self.label_id_mapping_path = os.path.join(machine_config.data_dir, 'preprocessed/label-mapping/{}_id_mapping.pkl')
        self.unknown_index = 1
        self.padding_index = 0
        self.title_max_words = 64
        self.abstract_max_words = 448
        self.max_labels = 77
        self.num_desc = 29351
        self.num_labels = 122542
        self.vocab_size = 64000
        self.min_year_completed = 1965
        self.max_year_completed = 2018
        self.num_year_completed_time_periods = 1 + self.max_year_completed - self.min_year_completed
        self.min_pub_year = 1902
        self.max_pub_year = 2019
        self.num_pub_year_time_periods = 1 + self.max_pub_year - self.min_pub_year

      
class _ProcessingConfig(_ConfigBase):
    def _initialize(self, machine_config):

        self.max_queue_size = machine_config.max_queue_size                                 
        self.use_multiprocessing = machine_config.use_multiprocessing                                
        self.workers = machine_config.workers                                                


class _ReduceLearningRateConfig(_ConfigBase):
    def _initialize(self, _):

        self.factor = 0.33
        self.min_delta = 0.001
        self.patience = 1
     

class _SaveConfig(_TextConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.model_json_filename = 'model.json'
        self.model_img_filename = 'model.png'
        self.settings_filename = 'settings.json'


class _OptimizeFscoreThresholdConfig(_ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)
        
        self.enabled = True
        
        self.alpha = 0.005
        self.k = 3
        self.metric_name = 'fscore'


class _TrainingConfig(_ProcessingConfig):
    def _initialize(self, machine_config):
        super()._initialize(machine_config)

        self.batch_size = 128
        self.dev_batch_size = 32
        self.max_avg_desc_per_citation = 12
        self.initial_epoch = 0
        self.max_epochs = 500
        self.train_limit = 1000000000
        self.dev_limit = 1000000000
        self.monitor_metric = 'val_fscore'
        self.monitor_mode = 'max'

        self.csv_logger = _CsvLoggerConfig(self)
        self.early_stopping = _EarlyStoppingConfig(self)
        self.optimize_fscore_threshold = _OptimizeFscoreThresholdConfig(self)
        self.reduce_learning_rate = _ReduceLearningRateConfig(self)
        self.save_config = _SaveConfig(self)

    
class Config(_ConfigBase):
    def __init__(self):
        super().__init__(self)

    def _initialize(self, machine_config):

        self.notes = ''
        self.root_dir = machine_config.runs_dir
        self.data_dir = machine_config.data_dir
        self.cross_val = _CrossValidationConfig(self)
        self.database = _DatabaseConfig(self)
        self.preprocessing = _PreprocessingConfig(self)
        self.train = _TrainingConfig(self)
        self.model = _ModelConfig(self)