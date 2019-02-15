from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from dl_segmenter import get_or_create, save_config
from dl_segmenter.custom.callbacks import LRFinder, SGDRScheduler, WatchScheduler, SingleModelCK, LRSchedulerPerStep
from dl_segmenter.data_loader import DataLoader
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    h5_dataset_path = "data/2014_processed.h5"  # 转换为hdf5格式的数据集
    config_save_path = "config/default-config.json"  # 模型配置路径
    weights_save_path = "models/weights.{epoch:02d}-{val_loss:.2f}.h5"  # 模型权重保存路径
    init_weights_path = "models/weights.23-0.02.sgdr.h5"  # 预训练模型权重文件路径
    embedding_file_path = "G:\data\word-vectors\word.embdding.iter5"  # 词向量文件路径，若不使用设为None
    embedding_file_path = None  # 词向量文件路径，若不使用设为None

    src_dict_path = "config/src_dict.json"  # 源字典路径
    tgt_dict_path = "config/tgt_dict.json"  # 目标字典路径
    batch_size = 32
    epochs = 128
    num_gpu = 1
    max_seq_len = 150
    initial_epoch = 0

    # GPU 下用于选择训练的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    steps_per_epoch = 2000
    validation_steps = 20

    data_loader = DataLoader(src_dict_path=src_dict_path,
                             tgt_dict_path=tgt_dict_path,
                             batch_size=batch_size,
                             max_len=max_seq_len,
                             shuffle_batch=steps_per_epoch,
                             sparse_target=False)

    config = {
        "vocab_size": data_loader.src_vocab_size,
        "chunk_size": data_loader.tgt_vocab_size,
        "sparse_target": data_loader.sparse_target,
        "embed_dim": 300,
        "bi_lstm_units": 256,
    }

    os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)

    segmenter = get_or_create(config,
                              optimizer=Adam(),
                              embedding_file=embedding_file_path,
                              src_dict_path=src_dict_path,
                              weights_path=init_weights_path)

    save_config(segmenter, config_save_path)

    segmenter.model.summary()

    ck = SingleModelCK(weights_save_path,
                       segmenter.model,
                       save_best_only=True,
                       save_weights_only=True,
                       monitor='val_loss',
                       verbose=0)
    log = TensorBoard(log_dir='logs',
                      histogram_freq=0,
                      batch_size=data_loader.batch_size,
                      write_graph=True,
                      write_grads=False)

    # Use LRFinder to find effective learning rate
    lr_finder = LRFinder(1e-6, 1e-2, steps_per_epoch, epochs=1)  # => (1e-4, 1e-3)
    lr_scheduler = SGDRScheduler(min_lr=1e-4, max_lr=1e-3,
                                 initial_epoch=initial_epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 cycle_length=10,
                                 lr_decay=0.9,
                                 mult_factor=1.2)

    X_train, Y_train, X_valid, Y_valid = DataLoader.load_data(h5_dataset_path, frac=0.9)

    segmenter.parallel_model.fit_generator(data_loader.generator_from_data(X_train, Y_train),
                                           epochs=epochs,
                                           steps_per_epoch=steps_per_epoch,
                                           validation_data=data_loader.generator_from_data(X_valid, Y_valid),
                                           validation_steps=validation_steps,
                                           callbacks=[ck, log, lr_scheduler],
                                           initial_epoch=initial_epoch)

    # lr_finder.plot_loss()
    # plt.savefig("loss.png")
