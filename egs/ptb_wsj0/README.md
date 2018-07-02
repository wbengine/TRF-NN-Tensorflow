# Understand this toolkit

* `run_baseline_ngram.py`: train ngram LMs and used to rescore n-best list. This need install SRILM toolkit first by 
```
cd tools
./install_srilm.sh
```
* `run_lstm.py`: train LSTM LM with standard softmax. The source code of LSTM is in `tfcode/lm/lstmlm.py`.
* `run_lstmlm_nce.py`: train LSTM LM using BNCE.