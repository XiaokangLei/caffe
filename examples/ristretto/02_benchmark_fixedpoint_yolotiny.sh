#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/yolotiny/RistrettoDemo/quantized.prototxt \
	--weights=models/yolotiny/RistrettoDemo/yolo_tiny.caffemodel \
	--gpu=0 --iterations=2000
