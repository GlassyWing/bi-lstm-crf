import argparse
import os
import re


def process_file(file):
    with open(file, 'r', encoding='UTF-8') as f:
        text = f.readlines()
        bises = _parse_text(text)
    return bises


def _parse_text(text: list):
    bises = []
    for line in text:
        line, _ = re.subn('\n', '', line)
        if line == '' or line == '\n':
            continue
        bises.append(_tag(line))
    return bises


def _tag(line):
    """
    给指定的一行文本打上BIS标签
    :param line: 文本行
    :return:
    """
    bis = []
    words = re.split('\s+', line)
    pre_word = None
    pos_t = None
    for word in words:
        tokens = word.split('/')

        if len(tokens) == 2:
            word, pos = tokens
        elif len(tokens) == 3:
            word, pos_t, pos = tokens
        else:
            continue

        if len(word) == 0 or word.strip() == '':
            continue

        if word[0] == '[':
            pre_word = word
            continue
        if pre_word is not None:
            pre_word += word
            if pos_t is None:
                continue
            elif pos_t[-1] != ']':
                pos_t = None
                continue
            else:
                word = pre_word[1:]
                pre_word = None
        pos_t = None
        bis.append((word, pos))

    return bis


def remove_pos(source_dir, target_path):
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            file = os.path.join(root, name)
            bises = process_file(file)

            with open(target_path, encoding="utf-8", mode="a") as f:
                for bis in bises:
                    sent, tags = [], []
                    for char, tag in bis:
                        sent.append(char)
                        tags.append(tag)
                    sent = ' '.join(sent)
                    f.write(sent + "\n")


def restore(source_dir, target_path):
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            file = os.path.join(root, name)
            bises = process_file(file)
            with open(target_path, encoding="utf-8", mode="a") as f:
                for bis in bises:
                    sent, tags = [], []
                    for char, tag in bis:
                        sent.append(char)
                        tags.append(tag)
                    sent = ''.join(sent)
                    f.write(sent + "\n")


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="根据指定的语料生成黄金标准文件与其相应的无分词标记的原始文件")
    parse.add_argument("--corups_dir", help="语料文件夹", default="../data/2014/")
    parse.add_argument("--gold_file_path", help="生成的黄金标准文件路径", default="../data/gold.utf8")
    parse.add_argument("--restore_file_path", help="生成无标记的原始文件路径", default="../data/restore.utf8")

    args = parse.parse_args()
    corups_dir = args.corups_dir
    gold_file_path = args.gold_file_path
    restore_file_path = args.restore_file_path

    print("Processing...")
    remove_pos(corups_dir, gold_file_path)
    restore(corups_dir, restore_file_path)
    print("Process done.")
