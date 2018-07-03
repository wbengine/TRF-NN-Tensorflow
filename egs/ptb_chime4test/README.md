# Training neural TRF LMs on CHiME 4 challenge

For the details of the experiments, see 
_Bin Wang, Zhijian Ou, "Learning neural trans-dimensional random field language models with noise-contrastive estimation." ICASSP, 2018_.

1. run ``local/task.py`` to prepare the dataset
2. run ``local/run_baseline_ngrams.py`` to train the ngram LMs
3. run ``local/run_lstmlm.py`` to train the LSTM LMs.
4. run ``local/run_trf_nce_pretrain.py`` to train the neural TRF LMs.

