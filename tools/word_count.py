from rx import Observable


def word_counts(lines):
    word_count = {}
    for line in lines:
        words = line.split()
        for word in words:
            if word_count.get(word) is None:
                word_count[word] = 1
            else:
                word_count[word] += 1
    return word_count


def save_to_file(source_file, target_file):
    with open(source_file, "r", encoding="UTF-8") as f:
        lines = f.readlines()

    word_count = word_counts(lines)
    with open(target_file, "a", encoding="UTF-8") as f:
        for w, c in word_count.items():
            f.write("{} {}\n".format(w, c))
        f.flush()


save_to_file("./score/gold_full.utf8", "./score/jieba.dict")
