import argparse

from dl_segmenter.data_loader import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="转换为hdf5格式")
    parser.add_argument("txt_path", type=str, help="BIS标注文本文件路径")
    parser.add_argument("h5_path", type=str, help="转换后的hdf5文件保存路径")
    parser.add_argument("-s", "--src_dict_path", type=str, help="源字典保存路径", required=True)
    parser.add_argument("-t", "--tgt_dict_path", type=str, help="目标字典保存路径", required=True)
    parser.add_argument("--seq_len", help="语句长度", default=150, type=int)

    args = parser.parse_args()

    data_loader = DataLoader(args.src_dict_path, args.tgt_dict_path,
                             batch_size=1,
                             max_len=args.seq_len,
                             sparse_target=False)

    data_loader.load_and_dump_to_h5(args.txt_path, args.h5_path, encoding='utf-8')
