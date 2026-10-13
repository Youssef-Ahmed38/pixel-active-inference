"""Public board observations and syntactic actions; no chess legality rules."""
from dataclasses import dataclass

from pai.cognition.contracts import SkillCommand, StateBelief


@dataclass(frozen=True)
class BoardObservation:
    # 0 empty, 1..6 white pawn..king, 7..12 black pawn..king; a1 first.
    pieces: tuple[int, ...]
    turn: int
    castling: tuple[int, ...]
    ep_square: int  # 64 means absent.

    def __post_init__(self):
        if len(self.pieces) != 64 or any(not 0 <= x <= 12 for x in self.pieces):
            raise ValueError("expected 64 piece codes in 0..12")
        if self.turn not in (0, 1) or len(self.castling) != 4 or any(x not in (0, 1) for x in self.castling):
            raise ValueError("invalid public turn or castling state")
        if not 0 <= self.ep_square <= 64:
            raise ValueError("invalid en passant square")

    @classmethod
    def from_belief(cls, belief: StateBelief):
        if belief.domain != "chess":
            raise ValueError("expected chess belief")
        squares = sorted((e for e in belief.entities if e.kind == "square"),
                         key=lambda e: int(e.features["index"]))
        board = next(e for e in belief.entities if e.kind == "board")
        return cls(tuple(int(e.features["piece"]) for e in squares),
                   int(board.features["turn"]),
                   tuple(int(board.features[f"castle_{i}"]) for i in range(4)),
                   int(board.features["ep_square"]))


@dataclass(frozen=True)
class BoardAction:
    source: int
    destination: int
    promotion: int = 0  # 0 none, 1 knight, 2 bishop, 3 rook, 4 queen.

    def __post_init__(self):
        if not (0 <= self.source < 64 and 0 <= self.destination < 64 and 0 <= self.promotion < 5):
            raise ValueError("action out of range")

    def uci(self):
        def square(i):
            return "abcdefgh"[i % 8] + str(i // 8 + 1)
        return square(self.source) + square(self.destination) + (" nbrq"[self.promotion].strip())

    def command(self):
        return SkillCommand("move_piece", arguments=(self.uci(),))

    @classmethod
    def from_uci(cls, text):
        if len(text) not in (4, 5):
            raise ValueError("invalid UCI action")
        def square(s):
            if s[0] not in "abcdefgh" or s[1] not in "12345678":
                raise ValueError("invalid square")
            return "abcdefgh".index(s[0]) + 8 * (int(s[1]) - 1)
        p = 0
        if len(text) == 5:
            if text[4] not in "nbrq":
                raise ValueError("invalid promotion")
            p = "nbrq".index(text[4]) + 1
        return cls(square(text[:2]), square(text[2:4]), p)


def candidate_actions():
    """Enumerate syntax only, without asking a teacher for a legal-move mask."""
    return tuple(BoardAction(a, b, p) for a in range(64) for b in range(64)
                 if a != b for p in range(5))
