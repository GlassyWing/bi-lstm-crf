from dl_segmenter.utils import make_dictionaries

if __name__ == '__main__':
    make_dictionaries("../data/2014_processed",
                          src_dict_path="../data/src_dict.json",
                          tgt_dict_path="../data/tgt_dict.json",
                          filters="\t\n",
                          oov_token="<UNK>",
                          min_freq=1)
