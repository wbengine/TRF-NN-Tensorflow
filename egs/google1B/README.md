# Google one-billion benchmark experiments

LMs are trained on Google 1-billion benchmark dataset, including about 0.8 billion words and a vocabulary of more than 500 K words.
For more details of the experiments configuration, see 
_Bin Wang, Zhijian Ou, "Improved training of neural trans-dimensional random field language models with dynamic noise-contrastive estimation", Submitted to SLT, 2018_. 

1. run ``corpus.py`` to prepare the vocabulary. The dataset path should be assigned in the script.
2. run ``exp/run_baseline_ngrams.py`` to train a baseline ngram LMs.
3. run ``exp/run_lstm.py`` to train a LSTM LM with adapeive softmax on multiple GPUs.
4. run ``exp/run_trf_nce.py`` to train a neural TRF LMs with DNCE on multiple GPUs.