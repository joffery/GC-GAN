import pickle

import numpy as np
import tensorflow as tf

from utils import mkdir_p
from vaegan import vaegan

flags = tf.app.flags

# ../../ssd/img_align_celeba
flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_epoch" , 100, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 128, "the dim of latent code")
flags.DEFINE_integer("learn_rate_init" , 0.0003, "the init of learn rate")
flags.DEFINE_string("path" , 'D:\workspace\common_database\multipie_processed', "the dataset directory")
flags.DEFINE_integer("operation", 0, "the init of learn rate")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./vaeganlogs/logs/celeba_test"
    vaegan_checkpoint_dir = "./model_vaegan/model.ckpt"
    sample_path = "./vaeganSample/sample"
    exp_transfer_path = "./vaeganSample/exp_transfer"
    lm_interpolation_path = "./vaeganSample/lm_interpolation"

    mkdir_p(root_log_dir)
    mkdir_p(vaegan_checkpoint_dir)
    mkdir_p(sample_path)
    mkdir_p(exp_transfer_path)
    mkdir_p(lm_interpolation_path)

    model_path = vaegan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_epoch = FLAGS.max_epoch
    latent_dim = FLAGS.latent_dim

    learn_rate_init = FLAGS.learn_rate_init

    with open('multipie/train_pairs.pickle', 'rb') as f:
        train_pairs = np.asarray(pickle.load(f, encoding="bytes"))
    with open('multipie/test_pairs.pickle', 'rb') as f:
        test_pairs = np.asarray(pickle.load(f, encoding="bytes"))

    vaeGan = vaegan(batch_size = batch_size, max_epoch = max_epoch,
                      model_path = model_path, train_data = train_pairs, test_data= test_pairs,
                      latent_dim = latent_dim,
                      sample_path = sample_path ,
                      exp_transfer_path = exp_transfer_path,
                      lm_interpolation_path = lm_interpolation_path,
                      log_dir = root_log_dir , learnrate_init=learn_rate_init)

    if FLAGS.operation == 0:

        vaeGan.build_model_vaegan()
        vaeGan.train()

    else:

        vaeGan.build_model_vaegan()
        vaeGan.test_landmark_interpolation()