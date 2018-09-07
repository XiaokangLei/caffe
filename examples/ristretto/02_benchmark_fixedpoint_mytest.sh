#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/mytest/RistrettoDemo/quantized.prototxt \
	--weights=models/mytest/RistrettoDemo/mytest_iter_500.caffemodel \
	--gpu=0 --iterations=2000
