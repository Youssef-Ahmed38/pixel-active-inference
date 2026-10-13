"""Rules live here, in the teaching environment, never in agent inference."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace

import chess

from pai.chess.representation import BoardAction, BoardObservation, candidate_actions
from pai.chess.state import GameState
from pai.cognition.contracts import (CognitiveEpisode, GoalRelation, SkillCommand, SkillOutcome)


LESSONS = {
    "opening": chess.STARTING_FEN,
    "castling": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "en_passant": "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "promotion": "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "check": "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    "mate_in_one": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "checkmate": "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
    "stalemate": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
}
for _piece, _symbol in (("pawn", "P"), ("knight", "N"), ("bishop", "B"),
                         ("rook", "R"), ("queen", "Q")):
    LESSONS[_piece] = f"4k3/8/8/8/3{_symbol}4/8/8/4K3 w - - 0 1"
LESSONS["king"] = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
LESSONS["capture"] = "4k3/8/8/2p1p3/3P4/8/8/4K3 w - - 0 1"


def observation(board: chess.Board) -> BoardObservation:
    pieces = tuple(0 if board.piece_at(i) is None else
                   board.piece_at(i).piece_type + (0 if board.piece_at(i).color else 6)
                   for i in range(64))
    rights = (board.has_kingside_castling_rights(chess.WHITE),
              board.has_queenside_castling_rights(chess.WHITE),
              board.has_kingside_castling_rights(chess.BLACK),
              board.has_queenside_castling_rights(chess.BLACK))
    return BoardObservation(pieces, int(board.turn), tuple(map(int, rights)),
                            64 if board.ep_square is None else board.ep_square)


class ChessTeacher:
    domain = "chess"

    def __init__(self, fen=chess.STARTING_FEN, episode=0):
        self.board = chess.Board(fen)
        if not self.board.is_valid():
            raise ValueError("invalid teaching position")
        self.step = 0
        self.public_history = [observation(self.board)]
        self.claimed_outcome = None
        self.record = CognitiveEpisode("chess", episode, GoalRelation("legal_move", "learner"), False, 0)
        self.record.beliefs.append(self.observe())

    @classmethod
    def from_board(cls, board, episode=0):
        """Resume a teacher game while preserving its public repetition history."""
        teacher = cls(board.root().fen(en_passant="fen"), episode)
        replay = board.root()
        history = [observation(replay)]
        for move in board.move_stack:
            if not replay.is_legal(move) or replay.outcome() is not None:
                raise ValueError("invalid game history")
            replay.push(move)
            history.append(observation(replay))
        if replay.fen(en_passant="fen") != board.fen(en_passant="fen"):
            raise ValueError("history does not reproduce the supplied board")
        teacher.board = board.copy(stack=True)
        teacher.public_history = history
        teacher.record.beliefs = [teacher.observe()]
        return teacher

    def observe(self):
        belief = GameState(observation(self.board), self.board.halfmove_clock,
                           self.board.fullmove_number, tuple(self.public_history)).to_belief(self.step)
        if self.game_outcome() is not None:
            belief = replace(belief, relations=frozenset({("game_over", "board", None)}))
        return belief

    def game_outcome(self):
        return self.claimed_outcome or self.board.copy(stack=True).outcome()

    def available_skills(self):
        return tuple(a.command() for a in candidate_actions()) + (SkillCommand("claim_draw"),)

    def demonstrate(self):
        if self.game_outcome() is not None:
            return ()
        return tuple(BoardAction.from_uci(m.uci()) for m in self.board.legal_moves)

    def execute(self, command: SkillCommand):
        before = observation(self.board)
        legal = False
        moved = False
        try:
            if self.game_outcome() is not None:
                raise ValueError("game is over")
            if command.name == "claim_draw" and len(command.arguments) <= 1:
                candidate = self.board.copy(stack=True)
                if command.arguments:
                    intended = chess.Move.from_uci(command.arguments[0])
                    if not candidate.is_legal(intended):
                        raise ValueError("invalid intended move")
                    candidate.push(intended)
                if candidate.is_fifty_moves():
                    self.claimed_outcome = chess.Outcome(chess.Termination.FIFTY_MOVES, None)
                elif candidate.is_repetition(3):
                    self.claimed_outcome = chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
                legal = self.claimed_outcome is not None
            elif command.name != "move_piece" or len(command.arguments) != 1:
                raise ValueError("expected one UCI move")
            else:
                move = chess.Move.from_uci(command.arguments[0])
                legal = self.board.is_legal(move)
                if legal:
                    self.board.push(move)
                    moved = True
        except ValueError:
            pass
        after = observation(self.board)
        if moved:
            self.public_history.append(after)
        changed = tuple((f"square_{i}", "piece") for i, (a, b) in
                        enumerate(zip(before.pieces, after.pieces)) if a != b)
        if legal and command.name == "claim_draw":
            changed += (("board", "game_over"),)
        outcome = SkillOutcome(command, legal, not legal, 1.0, changed)
        self.step += 1
        self.record.commands.append(command)
        self.record.outcomes.append(outcome)
        self.record.beliefs.append(self.observe())
        self.record.events.append({"legal": legal, "check": self.board.is_check(),
                                   "checkmate": self.board.is_checkmate(),
                                   "termination": None if self.game_outcome() is None else self.game_outcome().termination.name,
                                   "correction": None if legal else [a.uci() for a in self.demonstrate()]})
        self.record.steps = self.step
        self.record.success = legal
        return outcome


@dataclass(frozen=True)
class TeachingExample:
    before: BoardObservation
    action: BoardAction
    legal: bool
    after: BoardObservation
    topic: str


def position_key(obs):
    return hashlib.sha256(repr(obs).encode()).hexdigest()


def curriculum(seed=0, games=40, plies=60, negatives=8):
    """Random-game lessons plus motifs, deduplicated by agent-visible position.

    Every position includes all legal demonstrations and negatives from both
    occupied and arbitrary source squares. Labels exist only in this dataset.
    """
    if games < 1 or plies < 1 or negatives < 1:
        raise ValueError("games, plies and negatives must be positive")
    rng = random.Random(seed)
    positions = {}
    for topic, fen in LESSONS.items():
        b = chess.Board(fen)
        positions[position_key(observation(b))] = (b, topic)
    for _ in range(games):
        b = chess.Board()
        for _ in range(plies):
            legal = list(b.legal_moves)
            if not legal:
                break
            positions.setdefault(position_key(observation(b)), (b.copy(stack=False), "game"))
            b.push(rng.choice(legal))
    rows = []
    for b, topic in positions.values():
        before = observation(b)
        legal = list(b.legal_moves)
        actions = [(m, True) for m in legal]
        occupied = list(b.piece_map())
        rejected = set()
        target = max(negatives, len(legal))
        while len(rejected) < target:
            source = rng.choice(occupied) if rng.random() < .5 else rng.randrange(64)
            a = BoardAction(source, rng.randrange(64), rng.randrange(5) if rng.random() < .25 else 0)
            if a.source == a.destination:
                continue
            m = chess.Move.from_uci(a.uci())
            if not b.is_legal(m):
                rejected.add(m)
        actions += [(m, False) for m in sorted(rejected, key=lambda x: x.uci())]
        for move, label in actions:
            following = b.copy(stack=False)
            if label:
                following.push(move)
            rows.append(TeachingExample(before, BoardAction.from_uci(move.uci()), label,
                                        observation(following), topic))
    return rows


def split_examples(rows):
    """Hash complete visible positions, keeping all their actions in one split."""
    train, test = [], []
    for row in rows:
        (test if int(position_key(row.before)[:8], 16) % 5 == 0 else train).append(row)
    if not train or not test:
        raise ValueError("curriculum needs nonempty training and held-out positions")
    return train, test
