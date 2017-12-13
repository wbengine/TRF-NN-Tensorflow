#!/bin/bash

srilm_dir="./srilm"
package_name="srilm-1.7.1.tar.gz"
root_dir=$(pwd)
stage=0

if [ task_$1 = task_clean ]; then
	echo "delete all files in ${srilm_dir}"
	ls ${srilm_dir} | grep -v ${package_name} | xargs rm -rf
fi

if [ task_$1 = task_ ]; then
	#install srilm
	if [ ! -x srilm/ngram-count ]; then
		echo '$0: please run install_srilm.lm first'
	fi
	#if [ $stage -le 1 ]; then
	#	echo "stage 1: install srilm"
	#	cd ${root_dir}/${srilm_dir}
	#	tar -xzf ${package_name}
	#	mv Makefile Makefile.sv
	#	cat Makefile.sv | sed "/SRILM =/i\SRILM = $(pwd)/" > Makefile
	#	make World MAKE_PIC=-fPIC > build_srilm.log 2>&1
	#	cp $(find bin -name ngram-count) . --force
	#	cp $(find bin -name ngram) . --force
	#fi

	# install liblbfgs
	if [ $stage -le 2 ]; then
		echo "stage 2: install liblbfgs, used for MaxEnt model"
		git clone https://github.com/chokkan/liblbfgs.git ${root_dir}/liblbfgs
		cd ${root_dir}/liblbfgs/
		./autogen.sh && ./configure && make && sudo make install
		echo "stage 2: finished, please add /usr/local/lib to LD_LIBRARY_PATH"
	fi

	# install 
	if [ $stage -le 3 ]; then
		echo "stage 3: install srilm-python"
		echo "stage 3.1: clone the srilm-python"
		cd ${root_dir}/${srilm_dir}
		git clone https://github.com/nuance1979/srilm-python
		cd srilm-python/
		patch ${root_dir}/${srilm_dir}/lm/src/MEModel.cc < srilm/MEModel.cc.patch
		
		echo "stage 3.2: rebuild SRILM"
		cd ..
		make cleanest > /dev/null 2>&1
		make World MAKE_PIC=-fPIC HAVE_LIBLBFGS=1 > rebuild_srilm.log 2>&1
		cp $(find bin -name ngram-count) . --force
		cp $(find bin -name ngram) . --force
		
		echo "stage 3.3: buile python"
		export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
		cd srilm-python/
		make
	fi

	echo "Build finished, please: "
	echo "  1. add the /usr/local/lib to LD_LIBRARY_PATH"
	echo "  2. add the ${root_dir}/${srilm_dir}/srilm-python/ to PYTHONPATH or sys.path"
fi

