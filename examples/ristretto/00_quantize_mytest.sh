#!/usr/bin/env sh

./build/tools/ristretto quantize --model=models/mytest/train_val.prototxt \
      --weights=models/mytest/solver_iter_500.caffemodel \
      --model_quantized=models/mytest/RistrettoDemo/quantized.prototxt \
      --iterations=100 --gpu=0 --trimming_mode=dynamic_fixed_point --error_margin=1
