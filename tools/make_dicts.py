import argparse

from dl_segmenter.utils import make_dictionaries

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成字典。")
    parser.add_argument("file_path", type=str, help="用于生成字典的标注文件")
    parser.add_argument("-s", "--src_dict_path", type=str, help="源字典保存路径")
    parser.add_argument("-t", "--tgt_dict_path", type=str, help="目标字典保存路径")
    parser.add_argument("--min_freq", type=int, default=1, help="词频数阈值，小于该阈值的词将被忽略")

    args = parser.parse_args()

    make_dictionaries(args.file_path,
                      src_dict_path=args.src_dict_path,
                      tgt_dict_path=args.tgt_dict_path,
                      filters="\t\n",
                      oov_token="<UNK>",
                      min_freq=args.min_freq)
