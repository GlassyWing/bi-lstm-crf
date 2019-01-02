import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, Dense, LSTM
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_preprocessing.text import Tokenizer

from dl_segmenter.utils import load_dictionary, create_embedding_matrix, get_embedding_index


class DLSegmenter:
    __singleton = None

    def __init__(self,
                 vocab_size,
                 chunk_size,
                 embed_dim=300,
                 bi_lstm_units=256,
                 max_num_words=20000,
                 dropout_rate=0.1,
                 optimizer=Adam(),
                 emb_matrix=None,
                 weights_path=None,
                 src_tokenizer: Tokenizer = None,
                 tgt_tokenizer: Tokenizer = None):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.max_num_words = max_num_words
        self.bi_lstm_units = bi_lstm_units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.model = self.__build_model(emb_matrix)
        if weights_path is not None:
            try:
                self.model.load_weights(weights_path)
            except:
                print("Not weights found, create a new model.")

    def __build_model(self, emb_matrix=None):
        num_words = min(self.max_num_words, self.vocab_size + 1)
        word_input = Input(shape=(None,), dtype='int32', name="word_input")

        if emb_matrix is not None:
            word_embedding = Embedding(num_words, self.embed_dim,
                                       weights=[emb_matrix],
                                       trainable=True,
                                       name='word_emb')(word_input)
            print("Found emb matrix, applied.")
        else:
            word_embedding = Embedding(num_words, self.embed_dim, name="word_emb") \
                (word_input)
        bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2, return_sequences=True, bias_initializer="ones"))(
            word_embedding)
        x = Dropout(self.dropout_rate)(bilstm)
        dense = Dense(self.chunk_size + 1, kernel_initializer="he_normal")(x)
        crf = CRF(self.chunk_size + 1, sparse_target=False)
        crf_output = crf(dense)

        model = Model([word_input], [crf_output])

        model.compile(optimizer=self.optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def decode_sequences(self, sequences):
        sequences = self._seq_to_matrix(sequences)
        output = self.model.predict_on_batch(sequences)  # [N, -1, chunk_size + 1]
        output = np.argmax(output, axis=2)
        return self.tgt_tokenizer.sequences_to_texts(output)

    def _single_decode(self, args, noun_conjoin=True):
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
            if c == 's':
                if len(t1) != 0:
                    cur_sent.append(''.join(t1))
                    cur_tag.append(pre_pos)
                t1 = [word]
                pre_pos = pos
            elif c == 'i':
                t1.append(word)
                pre_pos = pos
            elif c == 'b':
                if len(t1) != 0:
                    cur_sent.append(''.join(t1))
                    cur_tag.append(pre_pos)
                t1 = [word]
                pre_pos = pos

        if len(t1) != 0:
            cur_sent.append(''.join(t1))
            cur_tag.append(pre_pos)

        if noun_conjoin:
            return self.__noun_conjoin(cur_sent, cur_tag)
        return cur_sent, cur_tag

    def __noun_conjoin(self, sent, tags):
        ret_sent = []
        ret_tags = []
        pre_word = None
        pre_tag = None
        for word, tag in zip(sent, tags):
            if tag == 'vn':
                if pre_word is not None:
                    ret_sent.append(pre_word)
                    ret_tags.append(pre_tag)
                pre_word = word
                pre_tag = tag
                continue
            elif tag == 'n':
                if pre_word is not None:
                    pre_word += word
                    pre_tag = tag
                else:
                    ret_sent.append(word)
                    ret_tags.append(tag)
            else:
                if pre_word is not None:
                    ret_sent.append(pre_word)
                    ret_tags.append(pre_tag)
                    pre_word = None
                    pre_tag = None
                ret_sent.append(word)
                ret_tags.append(tag)

        return ret_sent, ret_tags

    def decode_texts(self, texts, noun_conjoin=True):
        sents = []
        with ThreadPoolExecutor() as executor:
            for text in executor.map(lambda x: list(re.subn("\s+", "", x)[0]), texts):
                sents.append(text)
        sequences = self.src_tokenizer.texts_to_sequences(sents)
        tags = self.decode_sequences(sequences)

        ret = []
        with ThreadPoolExecutor() as executor:
            for cur_sent, cur_tag in executor.map(lambda x: self._single_decode(x, noun_conjoin),
                                                  zip(sents, tags)):
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
            "bi_lstm_units": self.bi_lstm_units,
            "max_num_words": self.max_num_words,
            "dropout_rate": self.dropout_rate
        }

    @staticmethod
    def get_or_create(config, src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None,
                      embedding_file=None,
                      optimizer=Adam(),
                      encoding="utf-8"):
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
                    emb_matrix = create_embedding_matrix(get_embedding_index(embedding_file), src_tokenizer.word_index,
                                                         min(config['vocab_size'] + 1, config['max_num_words']),
                                                         config['embed_dim'])
                    config['emb_matrix'] = emb_matrix
            if tgt_dict_path is not None:
                config['tgt_tokenizer'] = load_dictionary(tgt_dict_path, encoding)

            config['weights_path'] = weights_path
            config['optimizer'] = optimizer
            DLSegmenter.__singleton = DLSegmenter(**config)
        return DLSegmenter.__singleton
