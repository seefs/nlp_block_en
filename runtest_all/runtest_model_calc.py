
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from keras import backend as K
import numpy as np
import copy
import random
import logging


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# parameters
flags.DEFINE_string("mean", None, "mean, max or max_mean.")
#flags.DEFINE_string("h5_file", None, "h5 file.")

flags.DEFINE_bool("train", False, "train enable.")
flags.DEFINE_bool("test", False,  "test enable.")

flags.DEFINE_integer("batch_size", 40, "batch size.")
flags.DEFINE_integer("training_steps", 10, "training_steps.")
flags.DEFINE_integer("start_check_step", 4, "training_steps.")

flags.DEFINE_float("learning_rate", 5e-5, "learning rate.")


# In[1]:

# path
curPath  = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
dataPath = os.path.join(rootPath, "data")
log_dir   = os.path.join(dataPath,  "log")
modelPath = os.path.join(dataPath, "model")
csvPath  = os.path.join(dataPath, "csv")

# file
h5_file      = os.path.join(modelPath, "atec_nlp_calc_{}.h5")

# log
if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(modelPath) or not os.path.isdir(modelPath):
    os.mkdir(modelPath)

# cfg
sys.path.append(rootPath)

# Custom Class
from _loader import LoadData, save_pred_result
from _token import TokenizerChg
from _losses import f1, expand_dims_f1, cross_entropy_loss, accuracy
from _model import create_model


class RunData:
    def __init__(self, h5_file=None, mean=None, train_enable=True, test_enable=False):
        self.h5_file      = h5_file
        self.mean         = mean
        self.train_enable = train_enable
        self.test_enable  = test_enable
        
        # Training Parameters
        self.learning_rate = 0.0001
        self.training_steps = 10
        self.batch_size = 40
        self.display_step = 1
        
        # Dynamic step size
        # Master switch, recommended to open
        self.is_use_check_stop = True
        # Temporary variables, do not change
        self.cur_save_cnt  = 0
        # Initial exit condition:
        self.check_stop_acc   = 0.6
        self.check_stop_f1    = 0.7
        # Start verification:
        #   Start from 0, 0 means check every step.
        self.start_check_step = 4
        # Dynamic exit step spacing
        #   Set to about 6 to 10 steps; only affect exit conditions, not normal save.
        self.diff_stop_step = 8
        # Temporary variables, do not change
        self.cur_stop_step  = 0

        # data
        self.data  = None
        self.ds_train  = None
        self.ds_test   = None
        self.model  = None
        self.optimizer  = None

    def init_param(self, batch_size=40, training_steps=10, display_step=1, learning_rate=0.0001):
        self.batch_size     = batch_size
        self.training_steps = training_steps
        self.display_step   = display_step
        self.learning_rate  = learning_rate
        # init_data
        self.init_data()
        self.init_model()
        
    def init_stop_step(self, is_use_check_stop=True, init_stop_acc=0.6, init_stop_f1=0.7, start_check_step=4, diff_stop_step=8):
        # The training results are unstable, using variable step sizes, 
        # such as exiting without improvement within 8 steps.
        self.is_use_check_stop  = is_use_check_stop
        self.check_stop_acc  = init_stop_acc
        self.check_stop_f1  = init_stop_f1
        self.start_check_step  = start_check_step
        self.diff_stop_step  = diff_stop_step

    def init_data(self):
        self.data = LoadData(sample_size=None, train_enable=self.train_enable, test_enable=self.test_enable)
        self.data.show_data_shape()

        if self.data.train_enable:
            self.ds_train = tf.data.Dataset.from_tensor_slices(self.data.get_train_slices())
            self.ds_train  = self.ds_train.repeat().shuffle(1).batch(self.batch_size).prefetch(1)

        if self.data.test_enable:
            self.ds_test = tf.data.Dataset.from_tensor_slices(self.data.get_test_slices())
            self.ds_test  = self.ds_test.repeat().shuffle(1).batch(self.batch_size).prefetch(1)

    def init_model(self):
        # model
        self.model = create_model(self.data.max_vocab_len, self.data.max_seq_len, self.data.max_modes_len, h5_file=self.h5_file, 
            debug=True, mean=self.mean, save_data=False)
        # Adam optimizer.
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        #self.optimizer = keras.optimizers.SGD(self.learning_rate)

    def show_param(self):
        print ("  ================== ")
        print ("  train_enable: %s" % (self.train_enable))
        print ("  test_enable: %s" % (self.test_enable))
        print ("  mean: %s" % (self.mean))
        print ("  batch_size: %s" % (self.batch_size))
        print ("  training_steps: %s" % (self.training_steps))
        print ("  display_step: %s" % (self.display_step))
        print ("  learning_rate: %s" % (self.learning_rate))
        print ("  h5_file: %s" % (self.h5_file.format(self.data.max_vocab_len)))
        print ("  data_file: %s" % (self.data.data_file))
        print ("  ================== ")


# Optimization process. 
def run_optimization(mRun, x1, x2, m1, m2, mi1, mi2, n1, n2, cnt1, cnt2, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = mRun.model([x1, x2, m1, m2, mi1, mi2, n1, n2, cnt1, cnt2])
        #print("---pred %s" % (pred))
        # Compute loss.
        loss = cross_entropy_loss(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = mRun.model.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update weights following gradients.
    mRun.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return pred,loss


def check_run_stop(step, data, model, mRun):
    # 单步训练与总体评估，批大小不一样, 这里单独计算总体评估
    if data.test_enable:
        pred = model.predict(data.get_test_data())    # 感觉这个要快，相对于pred = model([])方式
        loss = cross_entropy_loss(pred, data.train_y)
        acc  = accuracy(data.train_y, pred)
        f1_score = expand_dims_f1(data.train_y, pred)  # 如果直接用f1，参数格式不对
    
        if step<0:
            # 训练结束, 传参数-1, 显示总指标:
            print(">>>>> Save %i times. predict: accuracy: %f, f1: %f" % (mRun.cur_save_cnt, acc, f1_score))
        elif acc>mRun.check_stop_acc and f1_score>mRun.check_stop_f1:
            # 更新f1, 但不更新acc
            # 检查f1  , 判断是否保存:
            mRun.check_stop_f1   = f1_score
            # 更新weights, 动态改变结束步长
            if mRun.cur_stop_step>0:
                print(">>>>> useful data. update stop step: %s, min f1 = %f"%(step+mRun.diff_stop_step, mRun.check_stop_f1))
                model.save_weights(h5_file.format(data.max_vocab_len), overwrite=True)
                mRun.cur_save_cnt = mRun.cur_save_cnt + 1
                # 第N步后退出:
                mRun.cur_stop_step = step + mRun.diff_stop_step
            else:
                # 第1步不保存, 效果未知, 避免覆盖上次训练的结果
                print(">>>>> init min f1 = %f"%(mRun.check_stop_f1))
                print(">>>>> init stop step: %s"%(step+mRun.diff_stop_step))
                # 第N步后退出:
                mRun.cur_stop_step = mRun.start_check_step + mRun.diff_stop_step
        else:
            # 退出条件
            if mRun.cur_stop_step>0 and step>mRun.cur_stop_step:
                print(">>>>> Stop running, No new effect within %s steps. cur step: %s > %s" % (mRun.diff_stop_step, step, mRun.cur_stop_step))
                return 1
            print(">>>>> %i. useless data. acc = %f, f1 = %f < %f" % (step, acc, f1_score, mRun.check_stop_f1))
    elif data.train_enable:
        pred = model.predict(data.get_train_data())    # 感觉这个要快，相对于pred = model([])方式
        loss = cross_entropy_loss(pred, data.train_y)
        acc  = accuracy(data.train_y, pred)
        f1_score = expand_dims_f1(data.train_y, pred)  # 如果直接用f1，参数格式不对
    
        if step<0:
            # 训练结束, 传参数-1, 显示总指标:
            print(">>>>> Save %i times. predict: accuracy: %f, f1: %f" % (mRun.cur_save_cnt, acc, f1_score))
        elif acc>mRun.check_stop_acc and f1_score>mRun.check_stop_f1:
            # 更新f1, 但不更新acc
            # 检查f1  , 判断是否保存:
            mRun.check_stop_f1   = f1_score
            # 更新weights, 动态改变结束步长
            if mRun.cur_stop_step>0:
                print(">>>>> useful data. update stop step: %s, min f1 = %f"%(step+mRun.diff_stop_step, mRun.check_stop_f1))
                model.save_weights(h5_file.format(data.max_vocab_len), overwrite=True)
                mRun.cur_save_cnt = mRun.cur_save_cnt + 1
                # 第N步后退出:
                mRun.cur_stop_step = step + mRun.diff_stop_step
            else:
                # 第1步不保存, 效果未知, 避免覆盖上次训练的结果
                print(">>>>> init min f1 = %f"%(mRun.check_stop_f1))
                print(">>>>> init stop step: %s"%(step+mRun.diff_stop_step))
                # 第N步后退出:
                mRun.cur_stop_step = mRun.start_check_step + mRun.diff_stop_step
        else:
            # 退出条件
            if mRun.cur_stop_step>0 and step>mRun.cur_stop_step:
                print(">>>>> Stop running, No new effect within %s steps. cur step: %s > %s" % (mRun.diff_stop_step, step, mRun.cur_stop_step))
                return 1
            print(">>>>> %i. useless data. acc = %f, f1 = %f < %f" % (step, acc, f1_score, mRun.check_stop_f1))
    return 0
    
def save_weights_logger(model):
    w = model.get_weights()
    logging.getLogger().info("--weights\n  %s" % (tf.shape(w)))
    logging.getLogger().info("--weights\n  %s" % (w))


def start_train(mRun):
    # Run training for the given number of steps.
    for step, (x1, x2, m1, m2, mi1, mi2, n1, n2, cnt1, cnt2, y) in enumerate(mRun.ds_train.take(mRun.training_steps), 1):
        if random.random() < .3:
            # 其实丢掉第1步，对初始化f1没有影响
            if step % mRun.display_step == 0 or step == 1:
                print("step: %i,  give up." % (step))
            continue
            
        check_ret = 0
        if mRun.is_use_check_stop and step>mRun.start_check_step or mRun.cur_stop_step == 0:
            # 一定会输出log，保存或不保存.
            check_ret = check_run_stop(step, mRun.data, mRun.model, mRun)
            
        if check_ret>0:
            break
        else:
            # Run the optimization to update W and b values.
            pred,loss = run_optimization(mRun, x1, x2, m1, m2, mi1, mi2, n1, n2, cnt1, cnt2, y)
        
            if step % mRun.display_step == 0 or step == 1:
                #loss = cross_entropy_loss(pred, y)
                acc   = accuracy(y, pred)
                f1_score  = expand_dims_f1(y, pred)   # 用f1有问题
                print("step: %i,  loss: %f, accuracy: %f, f1: %f" % (step, loss, acc, f1_score))


    if mRun.is_use_check_stop==False:
        print("  save: %s" % (mRun.h5_file.format(mRun.data.max_vocab_len)))
        mRun.model.save_weights(mRun.h5_file.format(mRun.data.max_vocab_len), overwrite=True)
        mRun.cur_save_cnt = mRun.cur_save_cnt + 1


    # 总的acc,f1
    check_run_stop(-1, mRun.data, mRun.model, mRun)


def main(_):
    # init
    mRun = RunData(h5_file=h5_file, mean=FLAGS.mean, train_enable=FLAGS.train, test_enable=FLAGS.test)
    
    # init param
    mRun.init_param(batch_size=FLAGS.batch_size, training_steps=FLAGS.training_steps, display_step=1, learning_rate=FLAGS.learning_rate)
    
    # init stop step
    mRun.init_stop_step(is_use_check_stop=True, init_stop_acc=0.6, init_stop_f1=0.7, start_check_step=FLAGS.start_check_step, diff_stop_step=8)
    
    # show
    mRun.show_param()
    
    # train
    start_train(mRun)
    
#    save_weights_logger(mRun.model)


if __name__ == "__main__":
	flags.mark_flag_as_required("mean")
	tf.compat.v1.app.run()


