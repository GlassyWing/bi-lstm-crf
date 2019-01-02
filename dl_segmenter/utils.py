import json
import os
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json


def _parse_data(fh, word_delimiter=' ', sent_delimiter='\t'):
    text = fh.readlines()
    sent, chunk = [], [],
    for line in text:
        line = line[0:-1]
        chars, tags = line.split(sent_delimiter)
        sent.append(chars.split(word_delimiter))
        chunk.append(tags.split(word_delimiter))
    return sent, chunk


def _parse_data_from_dir(file_dir, word_delimiter=' ', sent_delimiter='\t'):
    all_sent, all_chunk = [], []
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            file = os.path.join(root, name)
            sent, chunk = _parse_data(open(file, encoding="utf-8"), word_delimiter, sent_delimiter)
            all_sent.extend(sent)
            all_chunk.extend(chunk)
    return all_sent, all_chunk


def save_dictionary(tokenizer, dict_path, encoding="utf-8"):
    with open(dict_path, mode="w+", encoding=encoding) as file:
        json.dump(tokenizer.to_json(), file)


def load_dictionary(dict_path, encoding="utf-8"):
    with open(dict_path, mode="r", encoding=encoding) as file:
        return tokenizer_from_json(json.load(file))


def load_dictionaries(src_dict_path, tgt_dict_path, encoding="utf-8"):
    return load_dictionary(src_dict_path, encoding), load_dictionary(tgt_dict_path, encoding)


def make_dictionaries(file_path,
                      src_dict_path=None,
                      tgt_dict_path=None,
                      encoding="utf-8",
                      min_freq=5,
                      **kwargs):
    if not os.path.isdir(file_path):

        sents, chunks = _parse_data(open(file_path, 'r', encoding=encoding))
    else:
        sents, chunks = _parse_data_from_dir(file_path)

    src_tokenizer = Tokenizer(**kwargs)
    tgt_tokenizer = Tokenizer(**kwargs)

    src_tokenizer.fit_on_texts(sents)
    tgt_tokenizer.fit_on_texts(chunks)

    src_sub = sum(map(lambda x: x[1] < min_freq, src_tokenizer.word_counts.items()))
    tgt_sub = sum(map(lambda x: x[1] < min_freq, tgt_tokenizer.word_counts.items()))

    src_tokenizer.num_words = len(src_tokenizer.word_index) - src_sub
    tgt_tokenizer.num_words = len(tgt_tokenizer.word_index) - tgt_sub

    if src_dict_path is not None:
        save_dictionary(src_tokenizer, src_dict_path, encoding=encoding)
    if tgt_dict_path is not None:
        save_dictionary(tgt_tokenizer, tgt_dict_path, encoding=encoding)

    return src_tokenizer, tgt_tokenizer


def get_embedding_index(embedding_file):
    embedding_index = {}
    with open(os.path.join(embedding_file), encoding='UTF-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embedding_index[word] = coefs
    return embedding_index


def create_embedding_matrix(embeddings_index, word_index, vocab_size, embed_dim):
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
