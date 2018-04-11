# TRF toolkits based on tensorflow

This toolkits includes the source code of the trans-dimensional random field (TRF) language models (LMs), which is developed based on the tensorflow.
This toolkits also includes the baseline LMs, such as the ngram LMs and LSTM LMs.
LMs are evaulated by rescoring the n-best list to evaulate the performance in speech recognition. 

For the details of the TRF LMs, see:
[1] Bin Wang, Zhijian Ou, Zhiqiang Tan, “Learning Trans-dimensional Random Fields with Applications to Language Modeling”, IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2017.
[2] Bin Wang, and Zhijian Ou. "Language modeling with neural trans-dimensional random fields." ASRU, 2017.

Usage:
1. Python 3.0+ is need. We suggest the Anaconda 3.6 distribution in https://www.anaconda.com/download/
2. Tensorflow is need. See https://www.tensorflow.org/install
3. Install SRILM toolkits to apply the n-gram LM
```
cd tools
./install_srilm.sh
```
4. Install word2vec tools
```
cd tools/word2vec
make
```
5. Install the source code of Cython
```
cd tfcode/base
python setup.py build_ext --inplace
```

The experiments are in the folder egs/, which are originazed as follows:

- 'egs/word': the word morphology experiment
- 'egs/ptb_wsj0': train on PTB and rescore on WSJ'92 
- 'egs/ptb_chime4test': train on PTB and rescore on CHiME4 developing and test set
- 'egs/CHiME4': perform CHiME4 challange
- 'google1B': traing on Google 1-billion data set
- 'hkust': experiment on hkust dataset

Typical Experiments:

- For "Language modeling with neural trans-dimensional random fields." ASRU, 2017, see 'egs/ptb_wsj0/run_trf_neural_sa.py'
- For "Learning neural trans-dimensional random field language models with noise-contrastive estimation" ICASSP, 2018, see 'egs/CHiME4/run_trf_neural_nce.py'
