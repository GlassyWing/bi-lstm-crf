import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dense, Dropout, GRU, LSTM
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_preprocessing.text import Tokenizer

from dl_segmenter.utils import load_dictionary, create_embedding_matrix, get_embedding_index


class DLSegmenter:

    def __init__(self,
                 vocab_size,
                 chunk_size,
                 embed_dim=300,
                 bi_lstm_units=256,
                 dropout_rate=0.1,
                 num_gpu=0,
                 optimizer=Adam(),
                 sparse_target=False,
                 emb_matrix=None,
                 weights_path=None,
                 rule_fn=None,
                 src_tokenizer: Tokenizer = None,
                 tgt_tokenizer: Tokenizer = None):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.bi_lstm_units = bi_lstm_units
        self.dropout_rate = dropout_rate
        self.sparse_target = sparse_target

        self.rule_fn = rule_fn
        self.num_gpu = num_gpu
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.model, self.parallel_model = self.__build_model(emb_matrix)
        if weights_path is not None:
            try:
                self.model.load_weights(weights_path)
                logging.info("weights loaded!")
            except:
                logging.error("No weights found, create a new model.")

    def __build_model(self, emb_matrix=None):
        word_input = Input(shape=(None,), dtype='int32', name="word_input")

        word_emb = Embedding(self.vocab_size + 1, self.embed_dim,
                             weights=[emb_matrix] if emb_matrix is not None else None,
                             trainable=True if emb_matrix is None else False,
                             name='word_emb')(word_input)

        bilstm_output = Bidirectional(LSTM(self.bi_lstm_units // 2,
                                           return_sequences=True))(word_emb)

        bilstm_output = Dropout(self.dropout_rate)(bilstm_output)

        output = Dense(self.chunk_size + 1, kernel_initializer="he_normal")(bilstm_output)
        output = CRF(self.chunk_size + 1, sparse_target=self.sparse_target)(output)

        model = Model([word_input], [output])
        parallel_model = model
        if self.num_gpu > 1:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpu)

        parallel_model.compile(optimizer=self.optimizer, loss=crf_loss, metrics=[crf_accuracy])
        return model, parallel_model

    def decode_sequences(self, sequences):
        sequences = self._seq_to_matrix(sequences)
        output = self.model.predict_on_batch(sequences)  # [N, -1, chunk_size + 1]
        output = np.argmax(output, axis=2)
        return self.tgt_tokenizer.sequences_to_texts(output)

    def _single_decode(self, args):
        sent, tag = args
        cur_sent, cur_tag = [], []
        tag = tag.split(' ')
        t1, pre_pos = [], None
        for i in range(len(sent)):
            tokens = tag[i].split('-')
            if len(tokens) == 2:
                c, pos = tokens
            else:
                c = 'i'
                pos = "<UNK>"

            word = sent[i]
            if c in 'sb':
                if len(t1) != 0:
                    cur_sent.append(''.join(t1))
                    cur_tag.append(pre_pos)
                t1 = [word]
                pre_pos = pos
            elif c in 'ie':
                t1.append(word)
                pre_pos = pos

        if len(t1) != 0:
            cur_sent.append(''.join(t1))
            cur_tag.append(pre_pos)

        if self.rule_fn is not None:
            return self.rule_fn(cur_sent, cur_tag)
        return cur_sent, cur_tag

    def decode_texts(self, texts):
        sents = []
        with ThreadPoolExecutor() as executor:
            for text in executor.map(lambda x: list(re.subn("\s+", "", x)[0]), texts):
                sents.append(text)
        sequences = self.src_tokenizer.texts_to_sequences(sents)
        tags = self.decode_sequences(sequences)

        ret = []
        with ThreadPoolExecutor() as executor:
            for cur_sent, cur_tag in executor.map(self._single_decode, zip(sents, tags)):
                ret.append((cur_sent, cur_tag))

        return ret

    def _seq_to_matrix(self, sequences):
        max_len = len(max(sequences, key=len))
        return pad_sequences(sequences, maxlen=max_len, padding="post")

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "chunk_size": self.chunk_size,
            "embed_dim": self.embed_dim,
            "sparse_target": self.sparse_target,
            "bi_lstm_units": self.bi_lstm_units,
            "dropout_rate": self.dropout_rate,
        }

    __singleton = None
    __lock = Lock()

    @staticmethod
    def get_or_create(config, src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None,
                      embedding_file=None,
                      optimizer=Adam(),
                      rule_fn=None,
                      encoding="utf-8"):
        DLSegmenter.__lock.acquire()
        try:
            if DLSegmenter.__singleton is None:
                if type(config) == str:
                    with open(config, encoding=encoding) as file:
                        config = dict(json.load(file))
                elif type(config) == dict:
                    config = config
                else:
                    raise ValueError("Unexpect config type!")

                if src_dict_path is not None:
                    src_tokenizer = load_dictionary(src_dict_path, encoding)
                    config['src_tokenizer'] = src_tokenizer
                    if embedding_file is not None:
                        emb_matrix = create_embedding_matrix(get_embedding_index(embedding_file),
                                                             src_tokenizer.word_index,
                                                             min(config['vocab_size'] + 1, config['max_num_words']),
                                                             config['embed_dim'])
                        config['emb_matrix'] = emb_matrix
                if tgt_dict_path is not None:
                    config['tgt_tokenizer'] = load_dictionary(tgt_dict_path, encoding)

                config['rule_fn'] = rule_fn
                config['weights_path'] = weights_path
                config['optimizer'] = optimizer
                DLSegmenter.__singleton = DLSegmenter(**config)
        except Exception:
            traceback.print_exc()
        finally:
            DLSegmenter.__lock.release()
        return DLSegmenter.__singleton
