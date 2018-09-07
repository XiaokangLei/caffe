#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=models/yolotiny/RistrettoDemo/solver_finetune.prototxt \
	--weights=models/yolotiny/yolo_tiny.caffemodel
