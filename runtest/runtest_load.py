

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
flags.DEFINE_string("test_type", None, "test type.")
flags.DEFINE_integer("tcnt", 0, "tcnt.")
flags.DEFINE_integer("fcnt", 0, "fcnt.")
flags.DEFINE_integer("first_cnt", 0, "first_cnt.")
flags.DEFINE_integer("allcnt", 0, "allcnt.")


# path
curPath  = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
dataPath = os.path.join(rootPath, "data")
#csvPath  = os.path.join(dataPath, "csv")

# cfg
sys.path.append(rootPath)

# Custom Class
from _loader import LoadData, save_re_data, load_directory_data
from _loader import train_csv_file, test_csv_file, debugPath
from _token  import TokenizerChg

if not os.path.exists(debugPath):
    os.mkdir(debugPath)


# test-1:
def data_length_check(data):
# Display x1:
    for i in range(len(data.train_t1)):
        _t = data.train_t1[i]
        print("   text[%s]--%s--%s"%(i, len(_t), _t))
        
# Check whether the lengths of x1 are equal (spaces will not convert id, and finally have to be filled once):
    for i in range(len(data.train_x1)):
        _x = data.train_x1[i]
        print("   text[%s]--%s%s"%(i, len(_x), '--err' if len(_x)!=data.max_seq_len else ''))
    
# Display x2:
    for i in range(len(data.train_t2)):
        _t = data.train_t2[i]
        print("   text[%s]--%s--%s"%(i, len(_t), _t))

# Check if the lengths of x2 are equal (spaces will not convert id, it will be filled once at the end):
    for i in range(len(data.train_x2)):
        _x = data.train_x2[i]
        print("   text[%s]--%s%s"%(i, len(_x), '--err' if len(_x)!=data.max_seq_len else ''))

# test-2:
def data_order_take_out(_t1, _t2, _m1, _m2, _cnt1, _cnt2, _y):
    # The data order is changed, positive and negative
    t1, t2, m1, m2, cnt1, cnt2, y = [], [], [], [], [], [], []
    clen = len(_y)
    curb,i,j = 0,0,0
    while i<clen and j<clen:
        if curb:
            b = _y[i]
            while b!=curb:
                i+=1
                b = _y[i]
            t1.append(_t1[i])
            t2.append(_t2[i])
            m1.append(_m1[i])
            m2.append(_m2[i])
            cnt1.append(_cnt1[i])
            cnt2.append(_cnt2[i])
            y.append(_y[i])
            curb = 0
            i+=1
        else:
            b = _y[j]
            while b!=curb:
                j+=1
                b = _y[j]
            t1.append(_t1[j])
            t2.append(_t2[j])
            m1.append(_m1[j])
            m2.append(_m2[j])
            cnt1.append(_cnt1[j])
            cnt2.append(_cnt2[j])
            y.append(_y[j])
            curb = 1
            j+=1
    return np.array(t1), np.array(t2), np.array(m1), np.array(m2), np.array(cnt1), np.array(cnt2), np.array(y)

def data_take_out(data, tcnt=100, fcnt=100, allcnt=0, first_cnt=0):
    if data.train_enable:
        tcur,fcur,allcur=0,0,0
        t1, t2, m1, m2, cnt1, cnt2, y = [], [], [], [], [], [], []
        for i in range(data.sents_len):
            if i<first_cnt:
                continue
            b = data.train_y[i]
            tcur += b
            fcur += (1-b)
            allcur += 1
            # allcnt=0, Means unlimited total
            if (b>0 and tcur<=tcnt or b==0 and fcur<=fcnt) and (allcur<=allcnt or allcnt==0):
                t1.append(data.train_t1[i])
                t2.append(data.train_t2[i])
                m1.append(data.train_m1[i])
                m2.append(data.train_m2[i])
                cnt1.append(data.train_cnt1[i])
                cnt2.append(data.train_cnt2[i])
                y.append(data.train_y[i])
        t1, t2, m1, m2, cnt1, cnt2, y = np.array(t1), np.array(t2), np.array(m1), np.array(m2), np.array(cnt1), np.array(cnt2), np.array(y)
        t1, t2, m1, m2, cnt1, cnt2, y = data_order_take_out(t1, t2, m1, m2, cnt1, cnt2, y)
        ###
        cnt_info = str(first_cnt+1)+'_to_'+str(first_cnt+len(y))
        save_re_data(t1,   cnt_info=cnt_info, name='train_t1')
        save_re_data(t2,   cnt_info=cnt_info, name='train_t2')
        save_re_data(m1,   cnt_info=cnt_info, name='train_m1')
        save_re_data(m2,   cnt_info=cnt_info, name='train_m2')
        save_re_data(cnt1, cnt_info=cnt_info, name='train_cnt1')
        save_re_data(cnt2, cnt_info=cnt_info, name='train_cnt2')
        save_re_data(y,    cnt_info=cnt_info, name='train_y')
        print("---max cnt, true=%s, false=%s, all=%s"%(tcur,fcur,allcur))
        print("   save train take out--%s"%(len(y)))
        
    if data.test_enable:
        tcur,fcur,allcur=0,0,0
        t1, t2, m1, m2, cnt1, cnt2, y = [], [], [], [], [], [], []
        for i in range(data.sents_len):
            if i<first_cnt:
                continue
            b = data.test_y[i]
            tcur += b
            fcur += (1-b)
            allcur += 1
            if (b>0 and tcur<=tcnt or b==0 and fcur<=fcnt) and (allcur<=allcnt or allcnt==0):
                t1.append(data.test_t1[i])
                t2.append(data.test_t2[i])
                m1.append(data.test_m1[i])
                m2.append(data.test_m2[i])
                cnt1.append(data.test_cnt1[i])
                cnt2.append(data.test_cnt2[i])
                y.append(data.test_y[i])
        t1, t2, m1, m2, cnt1, cnt2, y = np.array(t1), np.array(t2), np.array(m1), np.array(m2), np.array(cnt1), np.array(cnt2), np.array(y)
        t1, t2, m1, m2, cnt1, cnt2, y = data_order_take_out(t1, t2, m1, m2, cnt1, cnt2, y)
        ###
        cnt_info = str(first_cnt+1)+'_to_'+str(first_cnt+len(y))
        save_re_data(t1,   cnt_info=cnt_info, name='test_t1')
        save_re_data(t2,   cnt_info=cnt_info, name='test_t2')
        save_re_data(m1,   cnt_info=cnt_info, name='test_m1')
        save_re_data(m2,   cnt_info=cnt_info, name='test_m2')
        save_re_data(cnt1, cnt_info=cnt_info, name='test_cnt1')
        save_re_data(cnt2, cnt_info=cnt_info, name='test_cnt2')
        save_re_data(y,    cnt_info=cnt_info, name='test_y')
        print("---max cnt, true=%s, false=%s, all=%s"%(tcur,fcur,allcur))
        print("   save test take out--%s"%(len(y)))


# test-3:
def save_tokens_parsing_info(data, df, fname=None):
    assert fname is not None
    text_split_file = os.path.join(debugPath, 'text_split_{}.txt')
    file_name  = os.path.join(debugPath, text_split_file.format(fname))
    
    # Save the word segmentation process:
    cnt = 0
    if data.sample_size is None:
        sample_size = 1
        sample_step = 0
    else:
        sample_size = data.sample_size
        sample_step = 1
    with open(file_name,'w') as f:
        for ndx, row in df.iterrows():
            text1, text2 = row[LoadData.DATA_COLUMN1], row[LoadData.DATA_COLUMN2]
            for text in [text1, text2]:
                if cnt>sample_size:
                    break
                unique_list = data.token_chg.get_jieba_cut_list(text);
                f.write("sentence %2d: %s\n" % (cnt, text))
                f.write("jieba  : %s\n" % (unique_list))
                mode0_list = data.token_chg.tokens_mode0(unique_list)
                mode1_list = data.token_chg.tokens_mode1(mode0_list)
                mode2_list = data.token_chg.tokens_mode2(mode1_list)
                mode3_list = data.token_chg.tokens_mode3(mode2_list)
                f.write("mode0_list : %s\n" % (mode0_list))
                f.write("mode1_list : %s\n" % (mode1_list))
                f.write("mode2_list : %s\n\n" % (mode2_list))
                f.write("mode3_list : %s\n\n" % (mode3_list))
                cnt=cnt+sample_step
        f.flush()
    return 0
    
def test_save_tokens(data, sub_path=""):
    # Save the word segmentation process:
    if data.train_enable:
        train = load_directory_data(train_csv_file.format(sub_path))
        save_tokens_parsing_info(data, train, fname="train")
    if data.test_enable:
        test  = load_directory_data(test_csv_file.format(sub_path))
        save_tokens_parsing_info(data, test, fname="test")


def main(_):
    print ("  test_type: %s" % (FLAGS.test_type))
    print ("  tcnt: %s" % (FLAGS.tcnt))
    print ("  fcnt: %s" % (FLAGS.fcnt))
    print ("  first_cnt: %s" % (FLAGS.first_cnt))
    print ("  allcnt: %s" % (FLAGS.allcnt))
    
    data = LoadData(sample_size=None, train_enable=True, test_enable=False)
    # show shape
    data.show_data_shape()
    
# test-1:
#   Check if the length is equal (when the preprocessing error, output the length of each data)
    if FLAGS.test_type == "data_length_check":
        data_length_check(data)

    
# test-2:
#   Parameter Description:
#      1. Take 100 ----- positive examples 100, negative examples 100, a total of 200
#        data_take_out (data, tcnt = 100, fcnt = 100)
#      Take 500 ----- positive example 500, negative example 500, a total of 1000
#        data_take_out (data, tcnt = 500, fcnt = 500)
#      2. Take 100 each, starting from 101-positive example 100, negative example 100, starting from 200 + 1
#        data_take_out (data, tcnt = 100, fcnt = 100, first_cnt = 200)
#      3. A total of 200, the number of positive and negative examples is uncertain
#        data_take_out (data, tcnt = 999, fcnt = 999, allcnt = 200)
#
    if FLAGS.test_type == "data_take_out":
        data_take_out(data, tcnt=FLAGS.tcnt, fcnt=FLAGS.fcnt, allcnt=FLAGS.allcnt, first_cnt=FLAGS.first_cnt)


# test-3:
#   Save the word segmentation process
    if FLAGS.test_type == "save_tokens":
        test_save_tokens(data, sub_path="")
        #test_save_tokens(data, sub_path="_tf200_2_f1_0529")


if __name__ == "__main__":
	flags.mark_flag_as_required("test_type")
	tf.compat.v1.app.run()


