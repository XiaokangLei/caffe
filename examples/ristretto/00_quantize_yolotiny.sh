#!/usr/bin/env sh

./build/tools/ristretto quantize --model=models/yolotiny/yolo_tiny_deploy.prototxt \
      --weights=models/yolotiny/yolo_tiny.caffemodel \
      --model_quantized=models/yolotiny/RistrettoDemo/quantized.prototxt \
      --iterations=100 --gpu=0 --trimming_mode=dynamic_fixed_point --error_margin=1
