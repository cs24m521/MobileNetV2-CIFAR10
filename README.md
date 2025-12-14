# MobileNet-V2 CIFAR-10 Compression
This project implements **post-training quantization** of **MobileNet-V2** on **CIFAR-10**, including configurable **weight and activation quantization**, evaluation, and **WandB sweep analysis**.

## Environment
* Platform: **Google Colab (Ubuntu 22.04)**
* GPU: **NVIDIA T4**
* Python: **3.10**

## Dependencies
```bash
pip install torch==2.1.0 torchvision==0.16.0
pip install wandb==0.23.1
pip install numpy==1.26.4
pip install matplotlib==3.8.0
pip install pyyaml==6.0.1
```
## Dataset
* CIFAR-10 (auto-downloaded via `torchvision`)

## Project Structure
```
models/        MobileNet-V2 definition
train/         Baseline training + evaluation
eval/          Quantized evaluation
compression/   Weight & activation quantization
data/          CIFAR-10 loader
checkpoints/   Baseline model (.pth)
sweep.yaml     WandB sweep config
sweep_run.py   Sweep execution
main.py        Entry point
```

## Baseline Training
```bash
python main.py --train
```
Output:
```
checkpoints/mobilenetv2_baseline.pth
```
(Baseline accuracy ≈ 85.6%)

## Quantized Evaluation (Single Run)
```bash
python main.py --eval_quant --weight_bits 4 --act_bits 8
```

## WandB Sweep Configuration
`sweep.yaml`
```yaml
method: grid
parameters:
  weight_bits: [4, 8]
  act_bits: [4, 8]
  seed: [42, 123]
```
Total runs: 8 

## Running the Sweep
```bash
wandb login
wandb sweep sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

## Seed Configuration
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)
```
Seeds used: **42, 123**


## Logged Metrics
* `weight_bits`
* `act_bits`
* `compression_ratio`
* `model_size_mb`
* `val_acc`

Used to generate the Parallel Coordinates plot in WandB.




