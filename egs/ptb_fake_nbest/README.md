# LMs trained on PTB and evaulated by rescoring the 1000-best list of WSJ'92 test set

Evaulating the performance of DNCE training.
For the details of experiment configuration, see 
_Bin Wang, Zhijian Ou, "Improved training of neural trans-dimensional random field language models with dynamic noise-contrastive estimation", Submitted to SLT, 2018_.

1. run 'gen_data.py' to prepare the dataset.
2. run 'run_ngrams.py' to train the baseline ngram LMs.
3. run 'run_lstm.py' to train the baseline LSTM LMs.
4. run 'run_trf_neural_nce.py' to train the neural TRF LMs with DNCE training.

