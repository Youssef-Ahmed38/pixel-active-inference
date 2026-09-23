"""Train the PixelAI decoder. Resumes automatically from <train.out_dir>/ckpt_last.pt.

    python scripts/train_decoder.py                                  # laptop / 1 GPU
    torchrun --nproc_per_node=2 scripts/train_decoder.py             # Kaggle 2x T4
    python scripts/train_decoder.py --resume-from /kaggle/input/prev-run/runs/pixelai_decoder
"""

from _cli import parse

from pai.train.decoder import train

if __name__ == "__main__":
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--resume-from", default=None))
    print(train(cfg, resume_from=args.resume_from))
