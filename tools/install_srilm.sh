#!/bin/bash

srilm_dir='./srilm'
root_dir=$(pwd)
package_name='srilm-1.7.1.tar.gz'

#install srilm
cd ${root_dir}/${srilm_dir}
tar -xzf ${package_name}
mv Makefile Makefile.sv
cat Makefile.sv | sed "/SRILM =/i\SRILM = $(pwd)/" > Makefile
make World MAKE_PIC=-fPIC > build_srilm.log 2>&1
cp $(find bin -name ngram-count) . --force
cp $(find bin -name ngram-merge) . --force
cp $(find bin -name ngram) . --force
