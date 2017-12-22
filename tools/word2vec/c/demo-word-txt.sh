make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
time ./word2vec -train text8 -output vectors.txt -cbow 0 -size 200 -window 7 -negative 1 -hs 1 -sample 1e-3 -threads 12 -binary 0 -save-vocab voc
./distance vectors.bin