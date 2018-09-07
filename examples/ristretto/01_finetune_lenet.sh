#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=models/LeNet/RistrettoDemo/solver_finetune.prototxt \
	--weights=models/LeNet/lenet_iter_10000.caffemodel
