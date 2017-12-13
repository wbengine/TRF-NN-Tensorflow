import os
import sys
import numpy as np
import corpus

sys.path.insert(0, os.getcwd() + '/../../tfcode/')
from model import reader
from model import wblib as wb
from model import ngram


def main():
    print(sys.argv)
    if len(sys.argv) == 1:
        print('\"python run_ngram.py -train\" train \n',
              '\"python run_ngram.py -rescore\" rescore nbest\n',
              '\"python run_ngram.py -wer\" compute WER'
              )
    if wb.is_window():
        bindir = 'd:\\wangbin\\tools'
    else:
        bindir = '../../tools/srilm'
    fres = wb.FRes('result.txt')  # the result file
    datadir = corpus.word_raw_dir()
    nbestdir = reader.wsj0_nbest()
    # print(nbestdir)
    workdir = 'ngramlm/'
    model = ngram.model(bindir, workdir)

    order_reg = [5]
    for order in order_reg:
        write_model = os.path.join(workdir, '{}gram.lm'.format(order))
        write_name = 'KN{}'.format(order)

        print(write_model)

        if '-train' in sys.argv or '-all' in sys.argv:
            if order_reg.index(order) == 0:
                model.prepare(*datadir)
            model.train(order, write_model)

        if '-test' in sys.argv or '-all' in sys.argv:
            PPL = [0]*3
            PPL[0] = model.ppl(write_model, order, datadir[0])
            PPL[1] = model.ppl(write_model, order, datadir[1])
            PPL[2] = model.ppl(write_model, order, datadir[2])
            fres.AddPPL(write_name, PPL, datadir[0:3])

        if '-rescore' in sys.argv or '-all' in sys.argv:
            model.rescore(write_model, order, nbestdir[3], write_model + '.lmscore')

        if '-wer' in sys.argv or '-all' in sys.argv:
            nbest = reader.NBest(*nbestdir)
            nbest.lmscore = wb.LoadScore(write_model + '.lmscore')

            wer = nbest.wer()
            print('wer={} lmscale={} acscale={}'.format(wer, nbest.lmscale, nbest.acscale))
            fres.AddWER(write_name, wer)

            trans_txt = workdir + 'nbest_transcripts.txt'
            nbest.get_trans_txt(trans_txt)
            PPL_trans = model.ppl(write_model, order, trans_txt)
            LL_trans = -wb.PPL2LL(PPL_trans, trans_txt)
            fres.Add(write_name, ['LL-wsj', 'PPL-wsj'], [LL_trans, PPL_trans])


if __name__ == '__main__':
    main()
