#!/usr/bin/env sh

./build/tools/ristretto quantize --model=models/LeNet/lenet_train_test.prototxt \
      --weights=models/LeNet/lenet_iter_10000.caffemodel \
      --model_quantized=models/LeNet/RistrettoDemo/quantized.prototxt \
      --iterations=1  --trimming_mode=dynamic_fixed_point --error_margin=1
