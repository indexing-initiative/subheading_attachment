from contextlib import closing
import math
from mysql.connector import connect
import numpy as np
import pickle
import random
import sentencepiece as sp
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils


SELECT_CITATION_DATA_SQL_TEMPLATE = '''
                                        SELECT 
                                            id, title, abstract, pub_year, date_completed, journal_id
                                        FROM 
                                            citations 
                                        WHERE 
                                            id IN ({})
                                    '''
SELECT_CITATION_DATA_AND_DESC_ID_SQL_TEMPLATE = '''
                                                    SELECT 
                                                        c.id, c.title, c.abstract, c.pub_year, c.date_completed, c.journal_id, 
                                                        mt.mesh_descriptor_id
                                                    FROM
                                                        citation_mesh_topics AS cmt,
                                                        citations AS c,
                                                        mesh_topics AS mt
                                                    WHERE
                                                        cmt.citation_id = c.id
                                                        AND cmt.mesh_topic_id = mt.id
                                                        AND c.id IN ({})
                                                    GROUP BY
                                                        c.id, mt.mesh_descriptor_id
                                                '''
SELECT_DESC_AND_QUAL_ID_SQL_TEMPLATE = '''
                                            SELECT
                                                cmt.citation_id, mt.mesh_descriptor_id, mt.mesh_qualifier_id
                                            FROM
                                                citation_mesh_topics as cmt,
                                                mesh_topics as mt
                                            WHERE
                                                cmt.mesh_topic_id = mt.id
                                                AND cmt.citation_id IN ({})
                                        '''
SELECT_DESC_ID_SQL_TEMPLATE =   '''
                                    SELECT 
                                        cmt.citation_id, mt.mesh_descriptor_id
                                    FROM 
                                        citation_mesh_topics AS cmt, 
                                        mesh_topics AS mt
                                    WHERE
                                        cmt.mesh_topic_id = mt.id
                                        AND cmt.citation_id IN ({})
                                    GROUP BY 
                                        cmt.citation_id, mt.mesh_descriptor_id
                                '''
SELECT_TOPIC_ID_SQL_TEMPLATE =  '''
                                    SELECT
                                        citation_id, mesh_topic_id
                                    FROM 
                                        citation_mesh_topics
                                    WHERE 
                                        citation_id IN ({})
                                '''


def create_sp_processor(model_path):
    sp_processor = sp.SentencePieceProcessor()
    sp_processor.Load(model_path)
    return sp_processor


def load_cross_validation_ids(config):
    train_set_ids = load_db_ids(config.train_set_ids_path, config.encoding, config.train_limit)
    dev_set_ids = load_db_ids(config.dev_set_ids_path, config.encoding, config.dev_limit)
    return train_set_ids, dev_set_ids


def load_delimited_data(path, encoding, delimiter):
    with open(path, 'rt', encoding=encoding) as file:
        data = tuple( tuple(data_item.strip() for data_item in line.strip().split(delimiter)) for line in file ) 
    return data


def load_db_ids(path, encoding, limit = 1000000000):
    db_ids = [int(id[0]) for id in load_delimited_data(path, encoding, ',')]
    db_ids = db_ids[:limit]
    return db_ids


def load_pickled_object(path):
    loaded_object = pickle.load(open(path, 'rb'))
    return loaded_object


class DatabaseGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, db_config, pp_config, sp_processor, label_id_mapping, db_ids, batch_size, max_examples = 1000000000):
        self._db_config = db_config
        self._pp_config = pp_config
        self._sp_processor = sp_processor
        self._label_id_mapping = label_id_mapping
        self._db_ids = db_ids
        self._batch_size = batch_size 
        self._num_examples = min(len(db_ids), max_examples)

    def __len__(self):
        length = int(math.ceil(self._num_examples/self._batch_size))
        return length

    def __getitem__(self, idx):
        batch_start = idx * self._batch_size
        batch_end = (idx + 1) * self._batch_size
        batch_ids = self._db_ids[batch_start:batch_end]
        
        with closing(connect(**self._db_config.config)) as db_conn:
            x_data = self._get_x_data(db_conn, batch_ids)
            y_data = self._get_y_data(db_conn, batch_ids)

        inputs = self._create_inputs(x_data)
        outputs = self._create_outputs(y_data)

        citation_count = len(batch_ids)
        example_count = len(x_data[0])
        s_idxs = self._get_sample_indices(citation_count, example_count)
        batch_x = self._format_inputs(inputs, s_idxs)
        batch_y = self._format_outputs(outputs, s_idxs)
        
        return batch_x, batch_y

    def _create_inputs(self, x_data):
        title, abstract, pub_year, year_completed, journal_id = x_data[0], x_data[1], x_data[2], x_data[3], x_data[4]

        title_input = self._vectorize_batch_text(title, self._pp_config.title_max_words)
        abstract_input = self._vectorize_batch_text(abstract, self._pp_config.abstract_max_words)
        
        pub_year = np.array(pub_year, dtype=np.int32).reshape(-1, 1)
        pub_year_indices = pub_year - self._pp_config.min_pub_year
        pub_year_input = self._to_time_period_input(pub_year_indices, self._pp_config.num_pub_year_time_periods)

        year_completed = np.array(year_completed, dtype=np.int32).reshape(-1, 1)
        year_completed_indices = year_completed - self._pp_config.min_year_completed
        year_completed_input = self._to_time_period_input(year_completed_indices, self._pp_config.num_year_completed_time_periods)

        journal_input = np.array(journal_id, dtype=np.int32).reshape(-1, 1)

        return title_input, abstract_input, pub_year_input, year_completed_input, journal_input

    def _create_outputs(self, y_data):
        if self._label_id_mapping is not None:
            label_ids = [[self._label_id_mapping[label_id] for label_id in label_ids if label_id in self._label_id_mapping] 
                         for label_ids in y_data]
        padded_label_ids = pad_sequences(label_ids, 
                                         maxlen=self._pp_config.max_labels, 
                                         dtype='int32', 
                                         padding='post', 
                                         truncating='post', 
                                         value=self._pp_config.padding_index)
        labels = self._to_categorical(padded_label_ids, self._pp_config.num_labels)
        return labels

    def _format_inputs(self, inputs, s_idxs):
        formatted = { 'title_input': inputs[0][s_idxs], 
                      'abstract_input': inputs[1][s_idxs], 
                      'pub_year_input': inputs[2][s_idxs], 
                      'year_completed_input': inputs[3][s_idxs], 
                      'journal_input': inputs[4][s_idxs],  }
        return formatted

    def _format_outputs(self, outputs, s_idxs):
        formatted =  { 'labels': outputs[s_idxs], }
        return formatted

    def _get_x_data(self, db_conn, db_ids):
        sql = SELECT_CITATION_DATA_SQL_TEMPLATE.format(','.join([str(db_id) for db_id in db_ids]))
        inputs_lookup = {}
        with closing(db_conn.cursor()) as cursor:
            cursor.execute(sql)              #pylint: disable=E1101
            for row in cursor.fetchall():    #pylint: disable=E1101
                id, title, abstract, pub_year, date_completed, journal_id = row
                year_completed = date_completed.year
                if not journal_id:
                    journal_id = 0
                inputs_lookup[id] = (title, abstract, pub_year, year_completed, journal_id)
        ordered_inputs = [inputs_lookup[db_id] for db_id in db_ids]
        return zip(*ordered_inputs)

    def _get_sample_indices(self, citation_count, example_count):
         s_idxs = list(range(example_count))
         return s_idxs

    def _get_y_data(self, db_conn, db_ids):
        y_data = None
        return y_data

    def _get_y_data_sql_template(self, db_conn, db_ids, sql_template):
        sql = sql_template.format(','.join([str(db_id) for db_id in db_ids]))
        label_lookup = {}
        with closing(db_conn.cursor()) as cursor:
            cursor.execute(sql)              #pylint: disable=E1101
            for row in cursor.fetchall():    #pylint: disable=E1101
                citation_id, label_id = row
                if citation_id not in label_lookup:
                    label_lookup[citation_id] = set()
                label_lookup[citation_id].add(label_id)
        ordered_label_ids = [list(label_lookup[db_id]) for db_id in db_ids]
        return ordered_label_ids

    def _to_categorical(self, ids, num_labels):
        count = ids.shape[0]
        max_labels = ids.shape[1]
        batch_indices = np.zeros([max_labels, count], np.int32)
        batch_indices[np.arange(max_labels)] = np.arange(count)
        batch_indices = batch_indices.T
        one_hot = np.zeros([count, num_labels + 1], np.int32)
        one_hot[batch_indices, ids] = 1
        one_hot = one_hot[:, 1:]
        return one_hot

    def _to_time_period_input(self, year_indices, num_time_periods):
        batch_size = year_indices.shape[0]
        batch_indices = np.zeros([batch_size, num_time_periods], np.int32)
        batch_indices[np.arange(batch_size)] = np.arange(num_time_periods)
        year_indices_rep = np.repeat(year_indices, num_time_periods, axis=1)
        time_period_input = batch_indices <= year_indices_rep
        time_period_input = time_period_input.astype(np.int32)
        return time_period_input

    def _vectorize_batch_text(self, batch_text, max_words):
        batch_word_indices = [self._sp_processor.EncodeAsIds(text) for text in batch_text]
        vectorized_text = pad_sequences(batch_word_indices, 
                                        maxlen=max_words, 
                                        dtype='int32', 
                                        padding='post', 
                                        truncating='post', 
                                        value=self._pp_config.padding_index)
        return vectorized_text


class MeshPairDatabaseGenerator(DatabaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_y_data(self, db_conn, db_ids):
        y_data = self._get_y_data_sql_template(db_conn, db_ids, SELECT_TOPIC_ID_SQL_TEMPLATE)
        return y_data


class MainheadingDatabaseGenerator(DatabaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_y_data(self, db_conn, db_ids):
        y_data = self._get_y_data_sql_template(db_conn, db_ids, SELECT_DESC_ID_SQL_TEMPLATE)
        return y_data


class SubheadingDatabaseGenerator(DatabaseGenerator):
    
    def __init__(self, max_avg_desc_per_citation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_max_avg_desc_per_citation = max_avg_desc_per_citation is not None
        self._max_avg_desc_per_citation = max_avg_desc_per_citation
      
    def _create_inputs(self, x_data):
        title_input, abstract_input, pub_year_input, year_completed_input, journal_input = super()._create_inputs(x_data)
        desc_ids = np.array(x_data[5], dtype=np.int32).reshape(-1, 1)
        desc_input = self._to_categorical(desc_ids, self._pp_config.num_desc)
        return title_input, abstract_input, pub_year_input, year_completed_input, journal_input, desc_input

    def _format_inputs(self, inputs, s_idxs):
        formatted = super()._format_inputs(inputs, s_idxs)
        formatted['desc_input'] = inputs[5][s_idxs]
        return formatted

    def _get_x_data(self, db_conn, db_ids):
        sql = SELECT_CITATION_DATA_AND_DESC_ID_SQL_TEMPLATE.format(','.join([str(db_id) for db_id in db_ids]))
        data_lookup = {}
        with closing(db_conn.cursor()) as cursor:
            cursor.execute(sql)              #pylint: disable=E1101
            for row in cursor.fetchall():    #pylint: disable=E1101
                citation_id, title, abstract, pub_year, date_completed, journal_id, desc_id = row
                year_completed = date_completed.year
                if not journal_id:
                    journal_id = 0
                if citation_id not in data_lookup:
                    data_lookup[citation_id] = { 'title': title, 
                                                 'abstract': abstract, 
                                                 'pub_year': pub_year, 
                                                 'year_completed': year_completed, 
                                                 'journal_id': journal_id, 
                                                 'desc_ids' : set()}
                data_lookup[citation_id]['desc_ids'].add(desc_id)
        ordered_inputs = []
        for db_id in db_ids:
            data = data_lookup[db_id]
            sorted_desc_ids = list(sorted(data['desc_ids']))
            for desc_id in sorted_desc_ids:
                ordered_inputs.append(( data['title'], 
                                        data['abstract'], 
                                        data['pub_year'], 
                                        data['year_completed'], 
                                        data['journal_id'], 
                                        desc_id))
        return zip(*ordered_inputs)

    def _get_sample_indices(self, citation_count, example_count):
        s_idxs = super()._get_sample_indices(citation_count, example_count)
        if self._apply_max_avg_desc_per_citation:
            max_examples = citation_count*self._max_avg_desc_per_citation
            k = min(example_count, max_examples)
            s_idxs = random.sample(s_idxs, k)
        return s_idxs

    def _get_y_data(self, db_conn, db_ids):
        sql = SELECT_DESC_AND_QUAL_ID_SQL_TEMPLATE.format(','.join([str(db_id) for db_id in db_ids]))
        label_lookup = {}
        with closing(db_conn.cursor()) as cursor:
            cursor.execute(sql)              #pylint: disable=E1101
            for row in cursor.fetchall():    #pylint: disable=E1101
                citation_id, desc_id, qual_id = row
                if citation_id not in label_lookup:
                    label_lookup[citation_id] = {}
                if desc_id not in label_lookup[citation_id]:
                    label_lookup[citation_id][desc_id] = []
                if qual_id not in label_lookup[citation_id][desc_id]:
                    label_lookup[citation_id][desc_id].append(qual_id)
        ordered_label_ids = []
        for db_id in db_ids:
            for desc_id in sorted(label_lookup[db_id]):
                ordered_label_ids.append(label_lookup[db_id][desc_id])
        return ordered_label_ids 