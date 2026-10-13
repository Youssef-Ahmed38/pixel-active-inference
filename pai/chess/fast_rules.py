"""Optional compilation of existing generic features; no rule oracle."""
import numpy as np
from numba import njit

@njit(cache=True)
def tree_predict(x, feature, threshold, left, right, value):
    out = np.empty(len(x), dtype=value.dtype)
    for row in range(len(x)):
        node = 0
        while left[node] >= 0:
            node = left[node] if x[row, feature[node]] <= threshold[node] else right[node]
        out[row] = value[node]
    return out

@njit(cache=True)
def fields(board, paths, table):
    blockers = np.zeros((64, 64), dtype=np.int16)
    white, black = np.zeros(64, dtype=np.int16), np.zeros(64, dtype=np.int16)
    attack = np.zeros((64, 64), dtype=np.int8)
    for src in range(64):
        for dst in range(64):
            for k in range(6):
                cell = paths[src, dst, k]
                if cell < 0: break
                blockers[src, dst] += int(board[cell] != 0)
            a = table[board[src], src, dst, int(blockers[src, dst] > 0)]
            attack[src, dst] = a
            if 0 < board[src] <= 6: white[dst] += a
            elif board[src] > 6: black[dst] += a
    return blockers, attack, white, black

@njit(cache=True)
def spatial(board, turn, castle, ep, actions, paths, table):
    blockers, attack, white, black = fields(board, paths, table)
    pw, pb = np.zeros((64, 64), dtype=np.int16), np.zeros((64, 64), dtype=np.int16)
    for src in range(64):
        for dst in range(64):
            for k in range(6):
                cell = paths[src, dst, k]
                if cell < 0: break
                pw[src, dst] += white[cell]
                pb[src, dst] += black[cell]
    x = np.empty((len(actions), 47), dtype=np.float32)
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for row in range(len(actions)):
        src, dst, promo = actions[row]
        sx, sy, tx, ty = src % 8, src // 8, dst % 8, dst // 8
        dx, dy = tx - sx, ty - sy
        a, b = board[src], board[dst]
        base = [a, b, turn, promo, sx, sy, tx, ty, dx, dy, abs(dx), abs(dy), blockers[src, dst],
                int(a > 6), int(b > 6), ep, int(dst == ep), white[src], black[src], white[dst], black[dst], pw[src, dst], pb[src, dst]]
        for col in range(23): x[row, col] = base[col]
        for col in range(4): x[row, 23 + col] = castle[col]
        for col in range(8): x[row, 27 + col] = board[sy * 8 + col]
        for col in range(4):
            ox, oy = offsets[col]
            xx, yy = tx + ox, ty + oy
            x[row, 35 + col] = board[xx + 8 * yy] if 0 <= xx < 8 and 0 <= yy < 8 else -1
        x[row, 39] = attack[src, dst]
        x[row, 40] = int(a > 0 and (a <= 6) == bool(turn))
        x[row, 41] = int(b > 0 and (b <= 6) == bool(turn))
        other, other_path = (black, pb) if turn else (white, pw)
        x[row, 42], x[row, 43], x[row, 44] = other[src], other[dst], other_path[src, dst]
        signx, signy = int(np.sign(dx)), int(np.sign(dy))
        distancex = 7 - sx if signx > 0 else sx if signx < 0 else 8
        distancey = 7 - sy if signy > 0 else sy if signy < 0 else 8
        distance = min(distancex, distancey)
        edge = min(63, max(0, sx + signx * distance + 8 * (sy + signy * distance)))
        x[row, 45] = blockers[src, edge] if dx == 0 or dy == 0 or abs(dx) == abs(dy) else 0
        x[row, 46] = int(b != 0)
    return x

@njit(cache=True)
def safety_batch(boards, turn, paths, table):
    x = np.zeros((len(boards), 25), dtype=np.int16)
    x[:, 0] = turn
    for row in range(len(boards)):
        _, _, white, black = fields(boards[row], paths, table)
        for square in range(64):
            code = boards[row, square]
            if code > 0:
                x[row, 2 * code - 1] = max(x[row, 2 * code - 1], white[square])
                x[row, 2 * code] = max(x[row, 2 * code], black[square])
    return x
