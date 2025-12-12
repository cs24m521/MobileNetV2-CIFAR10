<<<<<<< HEAD

# CS6886 Assignment 3 – MobileNetV2 Compression

## Train Baseline

python train/train.py


## Quantize Weights

python compression/weights_to_quant.py --bits 8


## Evaluate Quantized Model

python eval/eval_quant.py --weight_bits 8 --act_bits 8
>>>>>>> f06edc7 (Initial commit - assignment code)
