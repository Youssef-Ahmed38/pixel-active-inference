"""Video of one delivery: chase view, the rider's head camera, its map and its decisions.

    python scripts/bike_g1_delivery_video.py --town 4520 --house 3 --out docs/media/bike/delivery

Panels: the chase camera (top left); what the G1's head camera sees (top right; it turns toward the
street it is checking); the rider's map (bottom right: streets, its planned route, the streets it
believes blocked or clear, parked cars it has seen, its position and the house); the decision panel
(bottom left: speed, roll, target speed, the route, and every look with what was seen and what it
decided). The map shows only the obstacles the rider has seen.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.delivery import Delivery, blocked_prior
from pai.bike.g1_rider import G1BikeEnv
from pai.bike.navigation import HeadCamera
from pai.bike.town import Town, town_scene

W, H = 1280, 720
FPS = 25


def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_map(town, house, route, events, xy, yaw, size=(480, 360)):
    w, h = size
    img = Image.new("RGB", size, (245, 245, 240))
    d = ImageDraw.Draw(img)
    span = (town.n - 1) * town.block
    sc = min(w, h - 30) / (span + 16)
    ox, oy = (w - span * sc) / 2, 30 + (h - 30 - span * sc) / 2

    def P(p):
        return (ox + p[0] * sc, oy + (span - p[1]) * sc)

    seen, cars = {}, []
    for e in events:
        seen[tuple(sorted(e["street"]))] = e["seen"]
        if e["seen"] == "partial" and e["where"]:
            cars.append(e["where"])
    colour = {"blocking": (215, 40, 40), "clear": (60, 160, 70), "partial": (230, 150, 30)}
    for a, b in town.edges:
        s = seen.get((a, b))
        d.line([P(town.nodes[a]), P(town.nodes[b])], fill=colour.get(s, (175, 175, 175)), width=5 if s else 3)
        if s == "blocking":
            x, y = P(np.add(town.nodes[a], town.nodes[b]) / 2)
            d.line([x - 7, y - 7, x + 7, y + 7], fill=(215, 40, 40), width=3)
            d.line([x - 7, y + 7, x + 7, y - 7], fill=(215, 40, 40), width=3)
    for c in cars:
        x, y = P(c)
        d.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(40, 70, 160))
    nodes = dict(town.nodes)
    (a, b), pad, _, t = town.houses[house]
    nodes["pad"] = tuple(np.add(nodes[a], t * np.subtract(nodes[b], nodes[a])))
    d.line([P(nodes[n]) for n in route], fill=(30, 90, 220), width=2)
    x, y = P(town.depot[1])
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(30, 70, 200))
    d.text((x + 8, y - 18), "depot", fill=(30, 50, 140), font=font(13))
    x, y = P(pad)
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(240, 140, 20))
    d.text((x + 8, y - 8), f"house {house}", fill=(120, 60, 0), font=font(13))
    x, y = P(xy)
    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 0, 0))
    d.line([x, y, x + 12 * math.cos(yaw), y - 12 * math.sin(yaw)], fill=(0, 0, 0), width=3)
    d.text((8, 6), "rider's map: route blue, seen clear green, barrier red X, car orange",
           fill=(30, 30, 30), font=font(12))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--town", type=int, default=4520)
    ap.add_argument("--house", type=int, default=3)
    ap.add_argument("--gains-from", type=Path, default=ROOT / "results/bike_g1_turns_v3.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm-steer", action="store_true", help="the G1's arms turn the handlebar (no fork servo)")
    args = ap.parse_args()
    gains = np.array(json.loads(args.gains_from.read_text())["practised"])
    town = Town(n=4, seed=args.town)
    env = G1BikeEnv(scenery=town_scene(town), max_seconds=1e9, arm_kp=5.0, steer_kp=300.0, arm_steer=args.arm_steer)
    cam = HeadCamera(env.model, width=480, height=360)
    dv = Delivery(town, env, gains, blocked_prior(range(4000, 4004)), HeadCamera(env.model))
    res = dv.run(args.house, seed=args.town, record=True)
    print(res["outcome"], res["time_s"], "s,", res["replans"], "replans,", res["looks"], "looks")
    steps = [f for f in res["frames"] if "qpos" in f]
    events = res["events"]
    m = env.model
    data = mujoco.MjData(m)
    chase = mujoco.Renderer(m, 450, 800)
    ccam = mujoco.MjvCamera()
    ccam.type = mujoco.mjtCamera.mjCAMERA_FREE
    every = max(1, round(1 / (FPS * env.control_dt)))
    frames_dir = args.out.parent / (args.out.name + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    f_big, f_mid, f_small = font(22), font(17), font(15)
    yaw_s, n, canvas = None, 0, None
    for k in range(0, len(steps), every):
        st = steps[k]
        t = st["t"]
        data.qpos[:] = st["qpos"]
        mujoco.mj_forward(m, data)
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        yaw_s = yaw if yaw_s is None else yaw_s + 0.15 * ((yaw - yaw_s + math.pi) % (2 * math.pi) - math.pi)
        ccam.lookat[:] = data.qpos[:3] + np.array([0, 0, 0.6])
        ccam.distance, ccam.azimuth, ccam.elevation = 4.5, math.degrees(yaw_s), -16
        chase.update_scene(data, ccam)
        big = chase.render()
        past = [e for e in events if e["t"] <= t]
        looking = [e for e in past if t - e["t"] < 1.2]
        pov, _, _ = cam.look(data, looking[-1]["look_yaw"] if looking else yaw)
        canvas = Image.new("RGB", (W, H), (22, 24, 28))
        canvas.paste(Image.fromarray(big), (0, 0))
        canvas.paste(Image.fromarray(pov), (W - 480, 0))
        canvas.paste(Image.fromarray(draw_map(town, args.house, st["route"], past, data.qpos[:2], yaw)), (W - 480, 360))
        d = ImageDraw.Draw(canvas)
        d.rectangle([W - 480, 0, W, 24], fill=(0, 0, 0))
        d.text((W - 472, 4), "G1 head camera" + ("  - checking the next street" if looking else ""),
               fill=(255, 255, 255), font=f_small)
        d.rectangle([0, 0, 800, 30], fill=(0, 0, 0))
        d.text((10, 6), f"G1 parcel delivery: depot -> house {args.house}   (town {args.town}, never seen before)",
               fill=(255, 255, 255), font=f_mid)
        d.text((16, 458), f"t = {t:5.1f} s   speed {st['speed']:3.1f} m/s (target {st['target_speed']:.1f})   "
               f"roll {math.degrees(st['roll']):+5.1f} deg", fill=(235, 235, 235), font=f_big)
        d.text((16, 490), f"target lean {math.degrees(st['lean_ref']):+5.1f} deg   steering {math.degrees(st['steer']):+5.1f} deg"
               f"   route: " + " > ".join(map(str, st["route"])), fill=(200, 200, 200), font=f_mid)
        d.text((16, 528), "what the camera saw, and the decision:", fill=(255, 210, 120), font=f_mid)
        yy = 554
        for e in past[-5:]:
            seen = {"clear": "clear", "blocking": "BARRIER", "partial": "parked car"}[e["seen"]]
            line = f"{e['t']:5.1f}s  street {e['street'][0]}-{e['street'][1]}: {seen} ({e['points']} depth points)"
            if e.get("decision"):
                line += f"  ->  {e['decision']}"
            col = {"blocking": (255, 120, 110), "partial": (255, 190, 90)}.get(e["seen"], (150, 220, 150))
            d.text((16, yy), line, fill=col, font=f_small)
            yy += 26
        if k + every >= len(steps):
            d.text((16, 690), f"outcome: {res['outcome']} after {res['time_s']:.0f} s", fill=(255, 255, 255), font=f_mid)
        canvas.save(frames_dir / f"{n:05d}.png")
        n += 1
    for j in range(2 * FPS):           # hold the last frame
        canvas.save(frames_dir / f"{n + j:05d}.png")
    mp4 = str(args.out) + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "32", "-preset", "slow", mp4], check=True)
    print("wrote", mp4, round(Path(mp4).stat().st_size / 1e6, 2), "MB,", n, "frames")
    (args.out.parent / (args.out.name + "_events.json")).write_text(
        json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))
    for p in frames_dir.glob("*.png"):
        p.unlink()
    frames_dir.rmdir()


if __name__ == "__main__":
    main()
