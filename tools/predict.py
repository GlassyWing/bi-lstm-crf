import argparse

from dl_segmenter import get_or_create

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="执行命令行分词")
    parser.add_argument("-s", "--text", help="要进行分割的语句")
    parser.add_argument("-f", "--file", help="要进行分割的文件。", default="../data/restore.utf8")
    parser.add_argument("-o", "--out_file", help="分割完成后输出的文件。", default="../data/pred_text.utf8")

    args = parser.parse_args()

    tokenizer = get_or_create("../data/default-config.json",
                              src_dict_path="../data/src_dict.json",
                              tgt_dict_path="../data/tgt_dict.json",
                              weights_path="../models/weights.32--0.18.h5")

    text = args.text
    file = args.file
    out_file = args.out_file

    texts = []
    if text is not None:
        texts = text.split(' ')
        results = tokenizer.decode_texts(texts)
        print(results)

    elif file is not None:
        with open(file, encoding='utf-8') as f:
            texts = list(map(lambda x: x[0:-1], f.readlines()))

        if out_file is not None:
            with open(out_file, mode="w+", encoding="utf-8") as f:
                for text in texts:
                    seq, tag = tokenizer.decode_texts([text])[0]
                    f.write(' '.join(seq) + '\n')

