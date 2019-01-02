from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from dl_segmenter import get_or_create, save_config
from dl_segmenter.custom.callbacks import LRFinder, SGDRScheduler, WatchScheduler
from dl_segmenter.data_loader import DataLoader

if __name__ == '__main__':
    h5_dataset_path = "../data/2014_processed.h5"  # 转换为hdf5格式的数据集
    config_save_path = "../data/default-config.json"  # 模型配置路径
    weights_save_path = "../models/weights.{epoch:02d}-{val_loss:.2f}.h5"  # 模型权重保存路径
    init_weights_path = "../models/weights.23-0.02.sgdr.h5"  # 预训练模型权重文件路径
    embedding_file_path = "G:\data\word-vectors\word.embdding.iter5"  # 词向量文件路径，若不使用设为None
    embedding_file_path = None  # 词向量文件路径，若不使用设为None

    src_dict_path = "../data/src_dict.json"  # 源字典路径
    tgt_dict_path = "../data/tgt_dict.json"  # 目标字典路径
    batch_size = 32
    epochs = 32

    import os

    # GPU 下用于选择训练的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    data_loader = DataLoader(src_dict_path=src_dict_path,
                             tgt_dict_path=tgt_dict_path,
                             batch_size=batch_size)

    # steps_per_epoch = 415030 // data_loader.batch_size
    # validation_steps = 20379 // data_loader.batch_size

    steps_per_epoch = 2000
    validation_steps = 20

    config = {
        "vocab_size": data_loader.src_vocab_size,
        "chunk_size": data_loader.tgt_vocab_size,
        "embed_dim": 300,
        "bi_lstm_units": 256,
        "max_num_words": 20000,
        "dropout_rate": 0.1
    }

    tokenizer = get_or_create(config,
                              optimizer=Adam(),
                              embedding_file=embedding_file_path,
                              src_dict_path=src_dict_path,
                              weights_path=init_weights_path)

    save_config(tokenizer, config_save_path)

    # tokenizer.model.summary()

    ck = ModelCheckpoint(weights_save_path,
                         save_best_only=True,
                         save_weights_only=True,
                         monitor='val_loss',
                         verbose=0)
    log = TensorBoard(log_dir='../logs',
                      histogram_freq=0,
                      batch_size=data_loader.batch_size,
                      write_graph=True,
                      write_grads=False)

    # Use LRFinder to find effective learning rate
    lr_finder = LRFinder(1e-6, 1e-2, steps_per_epoch, epochs=1)  # => (2e-4, 3e-4)
    lr_scheduler = WatchScheduler(lambda _, lr: lr / 2, min_lr=2e-4, max_lr=4e-4, watch="val_loss", watch_his_len=2)
    lr_scheduler = SGDRScheduler(min_lr=4e-5, max_lr=1e-3, steps_per_epoch=steps_per_epoch,
                                 cycle_length=15,
                                 lr_decay=0.9,
                                 mult_factor=1.2)

    X_train, Y_train, X_valid, Y_valid = DataLoader.load_data(h5_dataset_path, frac=0.8)

    tokenizer.model.fit_generator(data_loader.generator_from_data(X_train, Y_train),
                                  epochs=1,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=data_loader.generator_from_data(X_valid, Y_valid),
                                  validation_steps=validation_steps,
                                  callbacks=[ck, log, lr_finder])

    lr_finder.plot_loss()
