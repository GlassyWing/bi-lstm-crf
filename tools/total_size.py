import os

if __name__ == '__main__':
    count = 0
    for root, dirs, files in os.walk("../data/2014/valid"):
        for name in files:
            file = os.path.join(root, name)
            with open(file, encoding='utf-8') as f:
                count += len(f.readlines())

    print(count)