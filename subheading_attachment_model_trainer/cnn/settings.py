import json


def get_config():
    config = Config()
    return config


class _ConfigBase:
    def __init__(self, parent):    
        self._parent = parent
        self._initialize()

    def __str__(self):
        _dict = {}
        self._toJson(_dict, self)
        return json.dumps(_dict, indent=4)

    def _initialize(self):
        pass
    
    @classmethod
    def _toJson(cls, parent, obj):
        for attribute_name in dir(obj): 
            if not attribute_name.startswith("_"):
                attribute = getattr(obj, attribute_name)
                if isinstance(attribute, _ConfigBase):
                    child = {}
                    parent[attribute_name] = child
                    cls._toJson(child, attribute)
                else:
                    parent[attribute_name] = attribute 


class _CheckpointConfig(_ConfigBase):
    def _initialize(self):
  
        self.enabled = True
        self.filename = "best_model.h5"
        self.weights_only = True


class _CsvLoggerConfig(_ConfigBase):
    def _initialize(self):

        self.optimum_threshold_filename = "optimum_threshold.txt"
        self.filename  = "logs.csv"
  

class _EarlyStoppingConfig(_ConfigBase):
    def _initialize(self):

        self.min_delta = 0.001
        self.patience = 2


class _ModelConfig(_ConfigBase):
    def _initialize(self):

        self.checkpoint = _CheckpointConfig(self)

        self.word_embedding_size = 300
        self.word_embedding_dropout_rate = 0.
     
        self.conv_act = "relu"
        self.num_conv_filter_sizes = 3
        self.min_conv_filter_size = 2
        self.conv_filter_size_step = 3
        self.total_conv_filters = 350
        self.num_pool_regions = 5

        self.num_journals = None # Remember +1 for unknown
        self.journal_embedding_size = 50
        self.journal_dropout_rate = 0.

        self.pub_year_dropout_rate = 0.
        self.year_completed_dropout_rate = 0.

        self.has_main_heading_input = None

        self.num_hidden_layers = 1
        self.hidden_layer_size = 2048
        self.hidden_layer_act = "relu"

        self.dropout_rate = None

        self.output_layer_act = "sigmoid"

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
    def num_main_headings(self):
        return self._pp_config.num_main_headings


class _PreprocessingConfig(_ConfigBase):
    def _initialize(self):

        self.unknown_index = 1
        self.padding_index = 0
        self.title_max_words = 64
        self.abstract_max_words = 448
        self.max_labels = 112
        self.num_main_headings = None
        self.num_labels = None
        self.vocab_size = 64000
        self.min_year_completed = 1965
        self.max_year_completed = None
        self.min_pub_year = 1902
        self.max_pub_year = None
        self.max_avg_main_headings_per_citation = 12

    @property
    def num_pub_year_time_periods(self):
        return (1 + self.max_pub_year - self.min_pub_year)

    @property
    def num_year_completed_time_periods(self):
        return (1 + self.max_year_completed - self.min_year_completed)
      
      
class _ProcessingConfig(_ConfigBase):
    def _initialize(self):

        self.max_queue_size = 30                              
        self.use_multiprocessing = True                                
        self.workers = 14                                            


class _ReduceLearningRateConfig(_ConfigBase):
    def _initialize(self):

        self.factor = 0.33
        self.min_delta = 0.001
        self.patience = 1
     

class _SaveConfig(_ConfigBase):
    def _initialize(self):

        self.model_json_filename = "model.json"
        self.model_img_filename = "model.png"
        self.settings_filename = "settings.json"


class _OptimizeFscoreThresholdConfig(_ProcessingConfig):
    def _initialize(self):
        super()._initialize()
        
        self.enabled = True
        self.alpha = 0.005
        self.k = 3
        self.metric_name = "f1_score"


class _TrainingConfig(_ProcessingConfig):
    def _initialize(self):
        super()._initialize()

        self.batch_size = 128
        self.dev_batch_size = 32
        self.initial_epoch = 0
        self.max_epochs = 500
        self.train_limit = 1000000000
        self.dev_limit = 1000000000
        self.monitor_metric = "val_f1_score"
        self.monitor_mode = "max"

        self.csv_logger = _CsvLoggerConfig(self)
        self.early_stopping = _EarlyStoppingConfig(self)
        self.optimize_fscore_threshold = _OptimizeFscoreThresholdConfig(self)
        self.reduce_learning_rate = _ReduceLearningRateConfig(self)
        self.save_config = _SaveConfig(self)

    
class Config(_ConfigBase):
    def __init__(self):
        super().__init__(self)

    def _initialize(self):

        self.notes = ""
        self.preprocessing = _PreprocessingConfig(self)
        self.train = _TrainingConfig(self)
        self.model = _ModelConfig(self)