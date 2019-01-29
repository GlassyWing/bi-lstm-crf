import os

import h5py
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from dl_segmenter.utils import load_dictionary


class DataLoader:

    def __init__(self,
                 src_dict_path,
                 tgt_dict_path,
                 batch_size=64,
                 max_len=999,
                 fix_len=True,
                 word_delimiter=' ',
                 sent_delimiter='\t',
                 shuffle_batch=10,
                 encoding="utf-8",
                 sparse_target=False):
        self.src_tokenizer = load_dictionary(src_dict_path, encoding)
        self.tgt_tokenizer = load_dictionary(tgt_dict_path, encoding)
        self.batch_size = batch_size
        self.max_len = max_len
        self.fix_len = fix_len
        self.word_delimiter = word_delimiter
        self.sent_delimiter = sent_delimiter
        self.src_vocab_size = self.src_tokenizer.num_words
        self.tgt_vocab_size = self.tgt_tokenizer.num_words
        self.shuffle_batch = shuffle_batch
        self.sparse_target = sparse_target

    def generator(self, file_path, encoding="utf-8"):
        if os.path.isdir(file_path):
            while True:
                for sent, chunk in self.load_sents_from_dir(file_path):
                    yield sent, chunk
        while True:
            for sent, chunk in self.load_sents_from_file(file_path, encoding):
                yield sent, chunk

    def load_sents_from_dir(self, source_dir, encoding="utf-8"):
        for root, dirs, files in os.walk(source_dir):
            for name in files:
                file = os.path.join(root, name)
                for sent, chunk in self.load_sents_from_file(file, encoding=encoding):
                    yield sent, chunk

    def load_sents_from_file(self, file_path, encoding):
        with open(file_path, encoding=encoding) as f:
            sent, chunk = [], []
            for line in f:
                line = line[:-1]
                chars, tags = line.split(self.sent_delimiter)
                sent.append(chars.split(self.word_delimiter))
                chunk.append(tags.split(self.word_delimiter))
                if len(sent) >= self.batch_size:
                    sent = self.src_tokenizer.texts_to_sequences(sent)
                    chunk = self.tgt_tokenizer.texts_to_sequences(chunk)
                    sent, chunk = self._pad_seq(sent, chunk)
                    if not self.sparse_target:
                        chunk = to_categorical(chunk, num_classes=self.tgt_vocab_size + 1)
                    yield sent, chunk
                    sent, chunk = [], []

    @staticmethod
    def load_data(h5_file_path, frac=None):
        with h5py.File(h5_file_path, 'r') as dfile:
            X, Y = dfile['X'][:], dfile['Y'][:]

            if frac is not None:
                assert 0 < frac < 1
                split_point = int(X.shape[0] * frac)
                X_train = X[:split_point]
                Y_train = Y[:split_point]
                X_valid = X[split_point:]
                Y_valid = Y[split_point:]
                return X_train, Y_train, X_valid, Y_valid
            return X, Y

    def generator_from_data(self, X, Y):
        steps = 0
        total_size = X.shape[0]
        while True:
            if steps >= self.shuffle_batch:
                indicates = list(range(total_size))
                np.random.shuffle(indicates)
                X = X[indicates]
                Y = Y[indicates]
                steps = 0
            sample_index = np.random.randint(0, total_size - self.batch_size)
            ret_x = X[sample_index:sample_index + self.batch_size]
            ret_y = Y[sample_index:sample_index + self.batch_size]

            if not self.sparse_target:
                ret_y = to_categorical(ret_y, num_classes=self.tgt_vocab_size + 1)
            else:
                ret_y = np.expand_dims(ret_y, 2)
            yield ret_x, ret_y
            steps += 1

    def load_and_dump_to_h5(self, file_path, output_path, encoding):
        with open(file_path, encoding=encoding) as f:
            sent, chunk = [], []
            for line in f:
                line = line[:-1]
                chars, tags = line.split(self.sent_delimiter)
                sent.append(chars.split(self.word_delimiter))
                chunk.append(tags.split(self.word_delimiter))

        sent = self.src_tokenizer.texts_to_sequences(sent)
        chunk = self.tgt_tokenizer.texts_to_sequences(chunk)
        sent, chunk = self._pad_seq(sent, chunk)

        indicates = list(range(sent.shape[0]))
        np.random.shuffle(indicates)
        sent = sent[indicates]
        chunk = chunk[indicates]

        with h5py.File(output_path, 'w') as dfile:
            dfile.create_dataset('X', data=sent)
            dfile.create_dataset('Y', data=chunk)

    def _pad_seq(self, sent, chunk):
        if not self.fix_len:
            len_sent = min(len(max(sent, key=len)), self.max_len)
            len_chunk = min(len(max(chunk, key=len)), self.max_len)
            sent = pad_sequences(sent, maxlen=len_sent, padding='post')
            chunk = pad_sequences(chunk, maxlen=len_chunk, padding='post')
        else:
            sent = pad_sequences(sent, maxlen=self.max_len, padding='post')
            chunk = pad_sequences(chunk, maxlen=self.max_len, padding='post')
        return sent, chunk
