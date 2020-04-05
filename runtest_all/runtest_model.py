
import sys
import os
import tensorflow as tf
import numpy as np
import copy

from sklearn.metrics import f1_score
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# parameters
flags.DEFINE_string("mean", None, "mean, max or max_mean.")
flags.DEFINE_bool("train", False, "train enable.")
flags.DEFINE_bool("test", False,  "test enable.")
flags.DEFINE_bool("save_np_data", False, "save np data.")
flags.DEFINE_bool("add_hide_seq", True, "add hide seq.")


# path
curPath  = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
dataPath = os.path.join(rootPath, "data")
modelPath = os.path.join(dataPath, "model")
csvPath  = os.path.join(dataPath, "csv")

# file
h5_file      = os.path.join(modelPath, "atec_nlp_calc_{}.h5")

# cfg
sys.path.append(rootPath)

# Custom Class
from _loader import LoadData, save_pred_result
#from _token import TokenizerChg
from _losses import f1, expand_dims_f1, cross_entropy_loss, accuracy
from _model import create_model


def main(_):
    print ("  train: %s" % (FLAGS.train))
    print ("  test: %s" % (FLAGS.test))
    print ("  mean: %s" % (FLAGS.mean))              # 'mean', 'max', 'max_mean'
    print ("  save_data: %s" % (FLAGS.save_np_data)) # 输出比较值(np)
    print ("  add_hide_seq: %s" % (FLAGS.add_hide_seq)) # 输出比较值(np)
    
    data = LoadData(sample_size=None, train_enable=FLAGS.train, test_enable=FLAGS.test, add_hide_seq=FLAGS.add_hide_seq)
    data.show_data_shape()
    
    model = create_model(data.max_vocab_len, data.max_seq_len, data.max_modes_len, h5_file=h5_file, debug=FLAGS.save_np_data, mean=FLAGS.mean, save_data=FLAGS.save_np_data)

    if data.train_enable:
        if FLAGS.save_np_data:
            # 用这个方法, 代码会走model.call(), 这样才能输出详细logging
            pred = model([data.get_train_data()])
        else:
            # 速度快
            pred = model.predict(data.get_train_data())
        loss = cross_entropy_loss(pred, data.train_y)
        print ("  pred--loss: %s" % (loss))
        acc = accuracy(data.train_y, pred)
        print ("  pred--acc:  %s" % (acc))
        f1_score = expand_dims_f1(data.train_y, pred)
        print ("  pred--f1:   %s" % (f1_score))
        res_string = 'loss=%s,   acc=%s,   f1_score=%s'%(np.array(loss), np.array(acc), np.array(f1_score))
        save_pred_result(data.train_t1, data.train_t2, pred, data.train_y, res_string, name='train_calc')
        
    if data.test_enable:
        if FLAGS.save_np_data:
            # 用这个方法, 代码会走model.call(), 这样才能输出详细logging
            pred = model.predict(data.get_test_data())
        else:
            # 速度快
            pred = model([data.get_test_data()])
        loss = cross_entropy_loss(pred, data.test_y)
        print ("  pred--loss: %s" % (loss))
        acc = accuracy(data.test_y, pred)
        print ("  pred--acc:  %s" % (acc))
        f1_score = expand_dims_f1(data.test_y, pred)  # 用f1有问题
        print ("  pred--f1:   %s" % (f1_score))
        res_string = 'loss=%s,   acc=%s,   f1_score=%s'%(np.array(loss), np.array(acc), np.array(f1_score))
        save_pred_result(data.test_t1, data.test_t2, pred, data.test_y, res_string, name='test_calc')


if __name__ == "__main__":
	flags.mark_flag_as_required("mean")
	tf.compat.v1.app.run()

