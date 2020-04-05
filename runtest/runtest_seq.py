

import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import backend as K


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# parameters
flags.DEFINE_string("test_type", None, "test type.")


# path
curPath  = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
dataPath = os.path.join(rootPath, "data")
csvPath  = os.path.join(dataPath, "csv")

# file
sqlite3_file = os.path.join(dataPath, "sqlite3", "data.db3")

# cfg
sys.path.append(rootPath)

# Block
from _block import m2n                    # Block
from _token import TokenizerChg



    
# test-1:
#   Encoding-> Fill
def m2n_test():
#    m   = [[ 6, 7, 3, 2, 0]]
    m   = [[ 10, 3, 5, 11]]
#    m   = [[7, 3, 11, 9, 10, 10]]
    cnt = [4]
    n1, mi1  = m2n(m, cnt, max_space=5, debug=True)
    print("--m--", m)
    print("--n1--", n1)
    print("--mi1--", mi1)



# test-2:
#   Word-> Coding-> Fill
def run_block_parsing(text):
    # No need to print the word segmentation process
    token_chg = TokenizerChg(db_path=sqlite3_file, debug_log=False)
    m1, cnt1, tokens = [], [], []
    for _t in text:
        # One more bracket
        for t in _t:
            tokens1 = token_chg.tokens_mode2(text=t)
#            tokens1 = token_chg.tokens_mode5(text=t)
            modes1  = [int(mode) for item in tokens1 for mode in item[1:]]
            m1.append(modes1)
            cnt1.append(len(modes1))
            tokens.append(tokens1)
    m1   = np.array(m1)
    cnt1 = np.array(cnt1)
    # Type of printing not set
    n1, mi1 = m2n(m1, cnt1, max_space=4, debug=False)
    for i in range(len(text)):
        print("-----------------")
        t = text[i]
        print ("sentence   : %s" % (t))
        print("  tokens1:", tokens[i])
        print("  m1:", m1[i])
        print("  cnt1:", cnt1[i])
        print("  n1:", n1[i])
    max_nlen = max([len(i) for i in n1])
    print("max_nlen:", max_nlen)


def tokens_parsing_main():
# test
    text = [
#            ["花呗都用在哪里"], 
#            ["花呗都能在哪里购物"],

            ["花呗分期付款的需要付手续费吗"], 
            ["花呗分期怎么才能免手续费"],
            
            
#            ["花呗支付支持农信银行卡不"], 
#            ["支持农信银行卡还花呗吗"],

           ]

# Single sentence test
    run_block_parsing(text);


def main(_):
    print ("  test_type: %s" % (FLAGS.test_type))
    
# test-1:
#   Coding-> Fill in hidden parts of speech
    if FLAGS.test_type == "m2n":
        m2n_test()

# test-2:
#   Word-> Code-> Fill hidden part of speech
    if FLAGS.test_type == "tokens_parsing":
        tokens_parsing_main()


if __name__ == "__main__":
    flags.mark_flag_as_required("test_type")
    tf.compat.v1.app.run()



