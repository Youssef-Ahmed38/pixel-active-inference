"""Train the entity world-model ensemble on slice transitions.

    python scripts/slice_train_wm.py --config configs/tabletop.yaml
"""

from _cli import parse

from pai.world.train import train_world_model

if __name__ == "__main__":
    _, cfg = parse(__doc__)
    print(train_world_model(cfg))
