

workdir='../../con-TRF/egs/ptb_wsj0/data/ptb/'

time ./word2vec -train ${workdir}ptb.train.txt -output ${workdir}vectors.txt -cbow 0 -size 200 -window 7 -negative 1 -hs 1 -sample 1e-3 -threads 12 -binary 0 -save-vocab ${workdir}voc -classes 200