# TRF semi-supervised learning on PTB

This is semi-supervised learning on PTB dataset.
TRFs are used to model the joint probabilities of the word sequences and the POS sequences.
TRFs are compared with CRFs in POS tagging task, and compared with ngram/LSTM in language modeling task.

### Folders

- full: supervised training on all the training data
- semi: semi-supervised training on part of training data


### Scripts

full/

|scripts         |description                         |
|----------------|------------------------------------|
|data_process.py | (run first) prepare the training data |
|eval.py         | including the code to compute the error rate |
|plot.py         | plot the results |
|plot_thesis.py  | plot the results used in thesis |
|run_baseline_ngrams.py | run ngram LMs |
|run_lsmtlm.py       | run lstm LMs |
|run_crf_discrete.py | run CRF with discrete features |
|run_crf_neural.py   | run neural CRF |
|run_crf_neural2.py  | run neural CRF, a better realization than `run_crf_neural.py` |
|run_hrf_\*.py | the old version of training TRFs. The new scripts are in `semi/` |

semi/

|scripts         |description                         |
|----------------|------------------------------------|
|task.py  | (run first) prepare the training data, i.e. selecting part of the training data including tags|
|eval.py         | including the code to compute the error rate |
|plot.py         | plot the results |
|run_lsmtlm.py     | run lstm LMs |
|run_crf_neural.py | train neural CRFs on part data set |
|run_hrf_neural.py | train neural TRFs by AugSA method (very slow) |
|run_hrf_neural_nce.py | train neural TRFs by NCE/DNCE (fast) |



