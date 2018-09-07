#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/LeNet/RistrettoDemo/quantized.prototxt \
	--weights=models/LeNet/RistrettoDemo/lenet_iter_10000.caffemodel \
	--gpu=0 --iterations=2000
