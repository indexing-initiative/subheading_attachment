import json
from .model import EmbeddingWithDropout
import os
from . import settings
import tensorflow as tf
import tensorflow.keras.backend as K
from   tensorflow.keras.layers import Input, Lambda, Layer
import tensorflow.keras.models
from tensorflow_text.python.ops.sentencepiece_tokenizer import SentencepieceTokenizer


class EncodeBatchText(Layer):
    
    def __init__(self, model_proto, max_len, **kwargs):
        super(EncodeBatchText, self).__init__(**kwargs)
        self.model_proto = model_proto
        self.max_len = max_len
        
    def build(self, input_shape):
        self.built = True
    
    def call(self, text):
        max_len = self.max_len
        text = tf.reshape(text, [-1])
        ids = SentencepieceTokenizer(self.model_proto).tokenize(text).to_tensor(default_value=0)
        ids = ids[:, :max_len] 
        ids = tf.pad(ids, [ [0,0], [0, (max_len - tf.shape(ids)[1])] ])
        ids = tf.reshape(ids, [-1, max_len])
        return ids


class EncodeBatchYear(Layer):
    
    def __init__(self, min_year, max_year, **kwargs):
        super(EncodeBatchYear, self).__init__(**kwargs)
        self.min_year = min_year
        self.max_year = max_year
        
    def build(self, input_shape):
        self.built = True
    
    def call(self, year):
        year = tf.reshape(year, [-1, 1]) # [?, 1]
    
        year_indices = year - self.min_year # [?, 1]
        year_indices = tf.where(year_indices >= 0, year_indices, tf.zeros_like(year_indices)) # [?, 1]
        num_time_periods = (self.max_year - self.min_year)  + 1 # [1]
        tiled_year_indices = tf.tile(year_indices, [1, num_time_periods]) # [?, num_time_periods]

        arange = tf.range(num_time_periods) # [num_time_periods]
        arange = tf.reshape(arange, [1, -1]) # [1, num_time_periods]
        column_indices = tf.tile(arange, [tf.shape(year)[0], 1]) # [?, num_time_periods]

        year_encoding = column_indices <= tiled_year_indices # [?, num_time_periods]
        year_encoding = tf.cast(year_encoding, tf.float32) # [?, num_time_periods]
        return year_encoding
        

class GatherData(Layer):
    
    def __init__(self, data, **kwargs):
        super(GatherData, self).__init__(**kwargs)
        self.data = data
        
    def build(self, input_shape):
        self.built = True
    
    def call(self, x):
        return K.gather(self.data, x)


class LookupIds(Layer):
    
    def __init__(self, journal_id_lookup, **kwargs):
        super(LookupIds, self).__init__(**kwargs)
        keys = tf.constant(list(journal_id_lookup.keys()), dtype=tf.string)
        values = tf.constant(list(journal_id_lookup.values()), dtype=tf.int32)
        default_value = tf.constant(0, dtype=tf.int32)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value)
        
    def build(self, input_shape):
        self.built = True
    
    def call(self, uis):
        uis = tf.reshape(uis, [-1]) # [?]
        ids = tf.map_fn(lambda x: self.table.lookup(x), uis, dtype=tf.int32) # [?]
        ids = tf.reshape(ids, [-1, 1]) # [?, 1]
        return ids

      
def run(deploy_dir, main_heading_output_dir, subheading_output_dir, wrap_model_resources, dataset_properties):
    config = settings.get_config()
    model_cfg = config.model
    pp_config = config.preprocessing
    checkpoint_filename = model_cfg.checkpoint.filename
    model_json_filename = config.train.save_config.model_json_filename
    optimum_threshold_filename = config.train.csv_logger.optimum_threshold_filename

    main_heading_checkpoint_path = os.path.join(main_heading_output_dir, checkpoint_filename)
    main_heading_model_json_path = os.path.join(main_heading_output_dir, model_json_filename)
    main_heading_thresh = _read_optimum_threshold(main_heading_output_dir, optimum_threshold_filename)
    
    subheading_checkpoint_path = os.path.join(subheading_output_dir, checkpoint_filename)
    subheading_model_json_path = os.path.join(subheading_output_dir, model_json_filename)
    subheading_thresh = _read_optimum_threshold(subheading_output_dir, optimum_threshold_filename)

    abstract_max_words = model_cfg.abstract_max_words
    max_pub_year = dataset_properties["max_pub_year"]
    max_year_completed = dataset_properties["max_year_completed"]
    min_pub_year = pp_config.min_pub_year
    min_year_completed = pp_config.min_year_completed
    num_mainheadings = dataset_properties["num_main_headings"]
    title_max_words = model_cfg.title_max_words
    
    critical_subheading_id_mapping = wrap_model_resources["critical_subheading_id_mapping"]
    journal_id_lookup = wrap_model_resources["journal_id_lookup"] 
    main_heading_id_lookup = wrap_model_resources["main_heading_id_lookup"]
    subheading_id_lookup = wrap_model_resources["subheading_id_lookup"]
    sp_model_proto = wrap_model_resources["sp_model_proto"]

    subheading_ui_lookup = { _id: ui for ui, _id in subheading_id_lookup.items()}
    subheading_ui_list = [""] + [subheading_ui_lookup[_id] for _id, _mapped_id in sorted(critical_subheading_id_mapping.items(), key=lambda x: x[1])]
   
    main_heading_ui_list = [ui for ui, _id in sorted(main_heading_id_lookup.items(), key=lambda x: x[1])]

    main_heading_model = _load_model(main_heading_model_json_path, main_heading_checkpoint_path)
    subheading_model =   _load_model(subheading_model_json_path,   subheading_checkpoint_path)

    pmid_input =           Input(shape=(1,), dtype="string", name="pmid")
    title_input =          Input(shape=(1,), dtype="string", name="title")
    abstract_input =       Input(shape=(1,), dtype="string", name="abstract")
    pub_year_input =       Input(shape=(1,), dtype="int32",  name="pub_year")
    year_completed_input = Input(shape=(1,), dtype="int32",  name="year_indexed")
    journal_input =        Input(shape=(1,), dtype="string", name="journal_id")

    reshape_layer = Lambda(lambda x: K.reshape(x, [-1,1]))
    encoded_title =           EncodeBatchText(sp_model_proto, title_max_words)(title_input)
    encoded_abstract =        EncodeBatchText(sp_model_proto, abstract_max_words)(abstract_input)
    endcoded_pub_year =       EncodeBatchYear(min_pub_year, max_pub_year)(pub_year_input)
    endcoded_year_completed = EncodeBatchYear(min_year_completed, max_year_completed)(year_completed_input)
    journal_ids =             LookupIds(journal_id_lookup)(journal_input)

    main_heading_pred = main_heading_model([encoded_title, encoded_abstract, endcoded_pub_year, endcoded_year_completed, journal_ids]) # [None, 29640]
    main_heading_pred_thresh = Lambda(lambda x: K.greater_equal(x, K.constant(value=main_heading_thresh, dtype="float32")))(main_heading_pred) # [None, 29640]
    main_heading_coordinates = Lambda(lambda x: K.cast(tf.where(x), "int32"))(main_heading_pred_thresh) # [?, 2]

    batch_indices = Lambda(lambda x: x[:,0])(main_heading_coordinates) # [?]
    main_heading_indices =  Lambda(lambda x: x[:,1])(main_heading_coordinates) # [?]

    gather_layer = Lambda(lambda x: K.gather(x[0], x[1]))
    batch_pmid =           gather_layer([pmid_input,              batch_indices ]) # [?]
    batch_title =          gather_layer([encoded_title,           batch_indices ]) # [?, 64]
    batch_abstract =       gather_layer([encoded_abstract,        batch_indices ]) # [?, 448]
    batch_pub_year =       gather_layer([endcoded_pub_year,       batch_indices ]) # [?, 119]
    batch_year_completed = gather_layer([endcoded_year_completed, batch_indices ]) # [?, 55]
    batch_journal_ids =    gather_layer([journal_ids,             batch_indices ]) # [?, 1]

    batch_main_heading_pred = Lambda(lambda x: K.one_hot(x, num_mainheadings))(main_heading_indices) # [?, 29640]

    subheading_pred = subheading_model([batch_title, batch_abstract, batch_pub_year, batch_year_completed, batch_journal_ids, batch_main_heading_pred]) # [?, 17]

    subheading_pred_thresh = Lambda(lambda x: K.greater_equal(x, K.constant(value=subheading_thresh, dtype="float32")))(subheading_pred) # [?, 17]
    subheading_none_pred = Lambda(lambda x: tf.math.logical_not(K.any(x, axis=1, keepdims=True)))(subheading_pred_thresh) # [?, 1]
    subheading_pred_thresh = Lambda(lambda x: K.concatenate(x, axis=1))([subheading_none_pred, subheading_pred_thresh]) # [?, 18]
    subheading_coordinates = Lambda(lambda x: K.cast(tf.where(x), "int32"))(subheading_pred_thresh) # [??, 2]

    indices = Lambda(lambda x: x[:,0])(subheading_coordinates) # [??]
    subheading_indices = Lambda(lambda x: x[:,1])(subheading_coordinates) # [??]

    pred_pmids = gather_layer([batch_pmid, indices]) # [??]
    pred_main_heading_indices = gather_layer([main_heading_indices, indices]) # [??]
    pred_main_heading_uis = GatherData(main_heading_ui_list)(pred_main_heading_indices) # [??]
    pred_subheading_uis = GatherData(subheading_ui_list)(subheading_indices) # [??]

    pred_pmids = reshape_layer(pred_pmids)
    pred_main_heading_uis = reshape_layer(pred_main_heading_uis)
    pred_subheading_uis = reshape_layer(pred_subheading_uis)

    output = Lambda(lambda x: K.concatenate(x, axis=1))([pred_pmids, pred_main_heading_uis, pred_subheading_uis])
    output = Lambda(lambda x: tf.cond(tf.equal(tf.shape(x)[0], 0), lambda: tf.constant([["","",""]], dtype=tf.string), lambda: x), name="predictions")(output)

    wrapper_model = tensorflow.keras.models.Model(inputs=[pmid_input, title_input, abstract_input, pub_year_input, year_completed_input, journal_input], outputs=[output])

    tf.keras.backend.set_learning_phase(0)
    tf.saved_model.save(wrapper_model, deploy_dir)


def _load_model(json_path, checkpoint_path):
    with open(json_path) as json_file:
        model_json = json_file.read()
    model = tensorflow.keras.models.model_from_json(model_json, custom_objects={ EmbeddingWithDropout.__name__: EmbeddingWithDropout })
    model.load_weights(checkpoint_path, by_name=False)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[])
    return model


def _read_optimum_threshold(model_dir, optimum_threshold_filename):
    optimum_threshold_path = os.path.join(model_dir, optimum_threshold_filename)
    with open(optimum_threshold_path) as read_file:
        optimum_threshold_text = read_file.read()
    optimum_threshold = float(optimum_threshold_text.strip())
    return optimum_threshold