
# use the tensroflow
try:
    from base import layers
except:
    print('[%s] no tensorflow.' % __name__)

# do not use the tensorflow
from base import ngram
from base import parser
from base import wblib as wb
from base import matlib as mlib
from base import reader
from base import vocab
from base import sampling as sp
from base import word2vec
from base import trie
from base import learningrate as lr
from base import log
from base import seq

import numpy as np
from scipy.misc import logsumexp
from collections import OrderedDict
