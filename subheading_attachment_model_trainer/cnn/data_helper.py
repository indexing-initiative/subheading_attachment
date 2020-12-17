import json
import math
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils


NULL_QUI = str(None)


class GeneratorBase(tensorflow.keras.utils.Sequence):

    def __init__(self, pp_config, training_resources, dataset, batch_size, max_examples):
        self._pp_config = pp_config
        self._sp_processor = training_resources["sp_processor"]
        self._journal_id_lookup = training_resources["journal_id_lookup"]
        self._main_heading_id_lookup = training_resources["main_heading_id_lookup"]
        self._dataset = dataset
        self._batch_size = batch_size
        self._label_id_mapping = None
        self._num_examples = min(len(dataset), max_examples)

    def __len__(self):
        length = int(math.ceil(self._num_examples/self._batch_size))
        return length

    def __getitem__(self, idx):
        batch_start = idx * self._batch_size
        batch_end = (idx + 1) * self._batch_size
        batch_data = self._dataset[batch_start:batch_end]
        batch_data = [json.loads(citation) for citation in batch_data]
        
        x_data = self._get_x_data(batch_data)
        y_data = self._get_y_data(batch_data)

        inputs = self._create_inputs(x_data)
        outputs = self._create_outputs(y_data)

        citation_count = len(batch_data)
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
            y_data = [[self._label_id_mapping[label_id] for label_id in label_ids if label_id in self._label_id_mapping] 
                         for label_ids in y_data]
        padded_label_ids = pad_sequences(y_data, 
                                         maxlen=self._pp_config.max_labels, 
                                         dtype="int32", 
                                         padding="post", 
                                         truncating="post", 
                                         value=self._pp_config.padding_index)
        labels = self._to_categorical(padded_label_ids, self._pp_config.num_labels)
        return labels

    def _format_inputs(self, inputs, s_idxs):
        formatted = { "title_input": inputs[0][s_idxs], 
                      "abstract_input": inputs[1][s_idxs], 
                      "pub_year_input": inputs[2][s_idxs], 
                      "year_completed_input": inputs[3][s_idxs], 
                      "journal_input": inputs[4][s_idxs],  }
        return formatted

    def _format_outputs(self, outputs, s_idxs):
        formatted =  { "labels": outputs[s_idxs], }
        return formatted

    def _get_citation_data(self, citation):
        title = citation["title"]
        abstract = citation["abstract"]
        pub_year = citation["pub_year"]
        year_completed = citation["year_completed"]
        nlmid = citation["journal_nlmid"]
        journal_id = self._journal_id_lookup[nlmid] if nlmid in self._journal_id_lookup else 0
        return (title, abstract, pub_year, year_completed, journal_id)

    def _get_x_data(self, batch_data):
        ordered_inputs = [self._get_citation_data(citation) for citation in batch_data]
        return list(zip(*ordered_inputs))

    def _get_sample_indices(self, citation_count, example_count):
         s_idxs = list(range(example_count))
         return s_idxs

    def _get_y_data(self, batch_data):
        y_data = None
        return y_data

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
                                        dtype="int32", 
                                        padding="post", 
                                        truncating="post", 
                                        value=self._pp_config.padding_index)
        return vectorized_text


class EndToEndGenerator(GeneratorBase):
    
    def __init__(self, pp_config, training_resources, dataset, batch_size, max_examples):
        super().__init__(pp_config, training_resources, dataset, batch_size, max_examples)
        self._mesh_pair_id_lookup = training_resources["mesh_pair_id_lookup"]
        self._label_id_mapping = training_resources["critical_mesh_pair_id_mapping"]

    def _get_y_data(self, batch_data):
        y_data = []
        for citation in batch_data:
            citation_data = []
            y_data.append(citation_data)
            for dui, quis in citation["mesh_headings"]:
                if len(quis) == 0:
                    citation_data.append(self._mesh_pair_id_lookup[(dui, NULL_QUI)])
                else:
                    citation_data.extend([self._mesh_pair_id_lookup[(dui, qui)] for qui in quis])
        return y_data


class MainHeadingGenerator(GeneratorBase):

    def __init__(self, pp_config, training_resources, dataset, batch_size, max_examples):
        super().__init__(pp_config, training_resources, dataset, batch_size, max_examples)

    def _get_y_data(self, batch_data):  
        y_data  = [[self._main_heading_id_lookup[dui] for dui, _ in citation["mesh_headings"]] for citation in batch_data]
        return y_data


class SubheadingGenerator(GeneratorBase):
    
    def __init__(self, pp_config, training_resources, dataset, batch_size, max_examples):
        super().__init__(pp_config, training_resources, dataset, batch_size, max_examples)
        self._subheading_id_lookup = training_resources["subheading_id_lookup"]
        self._label_id_mapping = training_resources["critical_subheading_id_mapping"]

    def _create_inputs(self, x_data):
        title_input, abstract_input, pub_year_input, year_completed_input, journal_input = super()._create_inputs(x_data[:5])
        main_heading_ids = np.array(x_data[5], dtype=np.int32).reshape(-1, 1)
        main_heading_input = self._to_categorical(main_heading_ids, self._pp_config.num_main_headings)
        return title_input, abstract_input, pub_year_input, year_completed_input, journal_input, main_heading_input

    def _format_inputs(self, inputs, s_idxs):
        formatted = super()._format_inputs(inputs[:5], s_idxs)
        formatted["main_heading_input"] = inputs[5][s_idxs]
        return formatted

    def _get_x_data(self, batch_data):
        ordered_inputs = []
        for citation in batch_data:
            citation_data = self._get_citation_data(citation)
            main_heading_ids = [ self._main_heading_id_lookup[descriptor_ui] for descriptor_ui, _ in citation["mesh_headings"]]
            for main_heading_id in sorted(main_heading_ids):
                _input = citation_data + (main_heading_id,) 
                ordered_inputs.append(_input)
        return list(zip(*ordered_inputs))

    def _get_sample_indices(self, citation_count, example_count):
        s_idxs = super()._get_sample_indices(citation_count, example_count)
        max_examples = citation_count*self._pp_config.max_avg_main_headings_per_citation
        k = min(example_count, max_examples)
        s_idxs = random.sample(s_idxs, k)
        return s_idxs

    def _get_y_data(self, batch_data):
        y_data = []
        for citation in batch_data:
            for _, quis in sorted(citation["mesh_headings"], key=lambda x: self._main_heading_id_lookup[x[0]]):
                if len(quis) == 0:
                    y_data.append([self._subheading_id_lookup[NULL_QUI]])
                else:
                    y_data.append([self._subheading_id_lookup[qui] for qui in quis])
        return y_data