"""Publish a hash-checked agent manifest only after both statistical gates pass."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pai.chess.ladder import file_hash,pair_statistics
from pai.chess.provenance import metadata

def main():
    positions = json.loads(Path("results/chess_strength_final_positions.json").read_text())
    games = json.loads(Path("results/chess_strength_final_promotion.json").read_text())
    group = games["groups"]["t1.0_incumbent"]
    stats = pair_statistics(games["raw_games"])
    if not positions["position_gate_passed"] or not stats["promotion_gate_passed"]:
        raise ValueError("position and fresh paired-game gates must both pass")
    hashes = games["checkpoint_hashes"]
    if any(file_hash(path) != expected for path,expected in hashes.items()):
        raise ValueError("tested model identity has changed")
    manifest = {**metadata([30400,30401,64400],{"both_sides_movetime_seconds":1.0}),
                "approved":True,"config":group["players"]["candidate"],"checkpoint_hashes":hashes,
                "position_report":"results/chess_strength_final_positions.json",
                "game_report":"results/chess_strength_final_promotion.json",
                "position_gate":{"positions":positions["fresh_positions"],"sign_p":positions["paired_cluster_sign_p_one_sided"]},
                "game_gate":{k:stats[k] for k in ("games","wins","draws","losses","promotion_gate_passed")},
                "preserved_incumbent":"runs/chess_rules/value.json"}
    Path("runs/chess_strength_data/promoted_agent.json").write_text(json.dumps(manifest,indent=2)+"\n")
    Path("results/chess_strength_approved_config.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Published the independently validated agent configuration")

if __name__ == "__main__": main()
