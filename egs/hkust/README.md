# Experiments on HKUST Chinese dataset

- ``exp_char``: character-based LMs. See 
_Bin Wang, Zhijian Ou, "Improved training of neural trans-dimensional random field language models with dynamic noise-contrastive estimation", Submitted to SLT, 2018_ for more details.
    1. run ``exp_chr/task.py`` to prepare the dataset. 
	2. run ``exp_chr/run_ngrams_chr.py`` to train a char-based ngram LM, and run ``exp_chr/run_ngram.py`` to train a word-based ngram LM.
	3. run ``exp_chr/run_lstm_chr.py`` to train a char-based LSTM LM, and run ``exp_chr/run_lstm.py`` to train a word-based LSTM LM.
	4. run ``exp_chr/run_trf_neural_chr_nce.py`` to train a char-based neural TRF LM.

- ``local``: word-based LMs.

- ``pku_train``: train LMs on pku_rmrb dataset.
