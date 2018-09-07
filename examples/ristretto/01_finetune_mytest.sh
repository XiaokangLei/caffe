#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=models/mytest/RistrettoDemo/solver_finetune.prototxt \
	--weights=models/mytest/solver_iter_500.caffemodel
