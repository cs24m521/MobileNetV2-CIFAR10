import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--act_bits", type=int, default=8)
    args = parser.parse_args()

    if args.train:
        subprocess.run(["python", "train/train.py"])

    if args.eval_quant:
        subprocess.run([
            "python",
            "eval/eval_quant.py",
            "--weight_bits", str(args.weight_bits),
            "--act_bits", str(args.act_bits)
        ])

