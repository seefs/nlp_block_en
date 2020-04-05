# Super part of speech


## 1.file
* Common module
```
_loader.py -------------Preprocessed data
_block.py --------------Preprocessed data
_token.py --------------Participle
_losses.py
_layer.py
_model.py
_tool.py ---------------Other, coding and log initialization
```
* training
```
run_block.py
runtest_all\runtest_model_calc.py
runtest_all\runtest_model.py
```
* test
```
runtest_all\runtest_param.py
runtest_all\runtest_merge.py  
runtest\runtest_load.py  
runtest\runtest_seq.py  
runtest\runtest_split.py  
```

## 2.Training and evaluation instructions

### 2.1.Training (not recommended, very card)
* Tested on an 8G memory computer, 1000 cards are very bad; 100 cards are fine  
```
python run_block.py
```


### 2.2.Training (available)
```
1) Only show parameters, no training
python_w runtest_all\runtest_model_calc.py --mean=mean

2) 10 steps of training, can run on ordinary computer  
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=mean --batch_size=40 --training_steps=10

3) 10 steps of training, in order to maintain the maximum value of f1, change the index to max_mean; it seems not to converge  
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=max_mean --batch_size=40 --training_steps=10 --learning_rate=5e-5

4) 60 steps of training
python_w runtest_all\runtest_model_calc.py --train=true --test=false --mean=mean --batch_size=120 --training_steps=30 --start_check_step=10 --learning_rate=2e-5
```
* Operation result: [view large image](/images/screenshot/train/train2.jpg)<br>
![train1](/images/screenshot/train/train1.jpg)

### 2.3. Forecast (available)
```
1) Mixed index (max_mean), dual index is better
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean

2) Single indicator (mean)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=mean

3) Single index (max)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max

4) Output intermediate data (slightly slow, used for debugging the best ordinal number, see 3.1)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean --save_np_data=true

5) Remove hidden mode (add_hide_seq)
python_w runtest_all\runtest_model.py --train=true --test=false --mean=max_mean --add_hide_seq=false

```

* View detailed results:
[data\debug\result_train_calc.txt](data/debug/result_train_calc.txt)
* View intermediate output: (used for debugging the best ordinal number, see 3.1)
[data\debug\brd_sum.txt](data/debug/brd_sum.txt)

* Running result (max_mean): [view large image](/images/screenshot/long/3.4.5_max_mean.jpg)<br>
![train1](/images/screenshot/short/3.4.5_max_mean.jpg)

* Operation result (mean): [view large image](/images/screenshot/long/3.4.5_mean.jpg)<br>
![train1](/images/screenshot/short/3.4.5_mean.jpg)

* Running result (max): [view large image](/images/screenshot/long/3.4.5_max.jpg)<br>
![train1](/images/screenshot/short/3.4.5_max.jpg)

* Operation result (add_hide_seq): [view large image](/images/screenshot/long/4_delete_seq.jpg)<br>
![train1](/images/screenshot/short/4_delete_seq.jpg)


### 3. Parameter analysis
### 3.1 Calculate the best coefficient
* Prerequisite: Output intermediate data (see 2.3)

```
1) Display basic parameters
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path=

2) Check the data before p = 0.49 ~ 0.501 to see if the prediction is accurate, this section is more difficult to control
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=mid_range

3) After updating the above two sets of coefficients and baseline, calculate the best baseline
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=max_base

4) After fixing the max group coefficient, reconfirm the best coefficient (a bit slow)
  bit = 0 ~ 16, determine the coefficient separately in 17 times
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=max_coefficient --bit=2

5) After fixing the mean group coefficient, reconfirm the best coefficient (a bit slow)
  bit = 0 ~ 16, determine the coefficient separately in 17 times
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=mean_coefficient --bit=9

6) Display the final mixed indicator
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=merge_show

7) Mixed indicators, reconfirm the best range parameters:
  bit = 0 ~ 9, and manually comment out the corresponding line of the range array (partB) (duplicates the range corresponding to bit)
python_w runtest_all\runtest_param.py --data_file=brd_sum --sub_model_path= --test_type=merge_coefficient
```

### 3.2 Parameter mixing
* Because the parameters of the same type are relatively close, you can use the trained data to initialize new parameters to speed up the training.
* Don't consider using it for now.
```
python runtest_all\runtest_merge.py
```


### 4. Preprocessing
### 4.1. Intercepting data length and word segmentation

```
1) Check whether the length is equal (when the preprocessing error, output the length of each data)
python_w runtest\runtest_load.py	--test_type=data_length_check

2) Length of intercepted data
# Positive example 100, negative example 100, a total of 200
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=100 --fcnt=100
# Positive example 500, negative example 500, a total of 1000
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=500 --fcnt=500
# Positive example 100, negative example 100, starting from the 200 + 1
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=500 --fcnt=500 --first_cnt=200
# A total of 200, the number of positive and negative examples is uncertain
python runtest\runtest_load.py	--test_type=data_take_out --tcnt=999 --fcnt=999 --allcnt=200

3) Save the word segmentation process
python_w runtest\runtest_load.py	--test_type=save_tokens
```

* View intercepted data:
data\csv\redata_1_to_200\train_xxx.csv

* View detailed word segmentation process:
[data\debug\text_split_train.txt](data/debug/text_split_train.txt)


### 4.2. Preprocessing--Part of Speech Coding

```
1) Coding-> Fill hidden part of speech
python_w runtest\runtest_seq.py  --test_type=m2n

2) Words-> Coding-> Fill hidden parts of speech
python_w runtest\runtest_seq.py  --test_type=tokens_parsing
```
* Operation result: [view large image](/images/screenshot/long/3.4.1_tokens_parsing.jpg)<br>
![train1](/images/screenshot/short/3.4.1_tokens_parsing.jpg)

### 4.3. Word Segmentation
* Word segmentation of one or several sentences, please add sentence content in the code
```
python runtest\runtest_split.py
```


