"""Learn legality and board effects from teaching; no rule-library dependency."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pai.chess.representation import BoardObservation, candidate_actions


def tensors(observations, actions, device="cpu"):
    boards = torch.tensor([o.pieces for o in observations], dtype=torch.long, device=device)
    context = torch.tensor([(o.turn, *o.castling) for o in observations], dtype=torch.float32, device=device)
    ep = torch.tensor([o.ep_square for o in observations], dtype=torch.long, device=device)
    moves = torch.tensor([(a.source, a.destination, a.promotion) for a in actions], dtype=torch.long, device=device)
    return boards, context, ep, moves


class ChessWorldModel(nn.Module):
    """Generic board/action network. Rule supervision comes entirely from teacher labels."""
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.piece = nn.Embedding(13, 8)
        self.square = nn.Embedding(65, 8)
        self.promotion = nn.Embedding(5, 4)
        self.trunk = nn.Sequential(nn.Linear(64 * 8 + 5 + 8 + 16 + 4 + 16, hidden),
                                   nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.legality = nn.Linear(hidden, 1)
        self.effects = nn.Linear(hidden, 64 * 13)

    def forward(self, boards, context, ep, moves):
        pieces = self.piece(boards)
        index = torch.arange(len(boards), device=boards.device)
        selected = torch.cat((pieces[index, moves[:, 0]], pieces[index, moves[:, 1]]), dim=1)
        features = torch.cat((pieces.flatten(1), context, self.square(ep),
                              self.square(moves[:, :2]).flatten(1),
                              self.promotion(moves[:, 2]), selected), dim=1)
        latent = self.trunk(features)
        # Generic persistence prior: unchanged cells retain their observed token.
        # The learned residual must still account for every changed cell.
        prior = 4.0 * F.one_hot(boards, num_classes=13).float()
        return self.legality(latent).squeeze(1), prior + self.effects(latent).reshape(-1, 64, 13)

    @torch.no_grad()
    def choose(self, observation: BoardObservation, batch_size=512):
        """Rank every syntactic action. No teacher mask or legality fallback."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.eval()
        device = next(self.parameters()).device
        actions = candidate_actions()
        best_action, best_score = None, -float("inf")
        for start in range(0, len(actions), batch_size):
            chunk = actions[start:start + batch_size]
            logits, _ = self(*tensors([observation] * len(chunk), chunk, device))
            score, index = logits.max(0)
            if score.item() > best_score:
                best_action, best_score = chunk[index.item()], score.item()
        return best_action, float(torch.sigmoid(torch.tensor(best_score)))


def encode_examples(rows, device="cpu"):
    inputs = tensors([r.before for r in rows], [r.action for r in rows], device)
    labels = torch.tensor([r.legal for r in rows], dtype=torch.float32, device=device)
    after = torch.tensor([r.after.pieces for r in rows], dtype=torch.long, device=device)
    return inputs, labels, after


def fit(model, rows, epochs=10, batch_size=256, seed=0, learning_rate=.001):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    device = next(model.parameters()).device
    inputs, labels, after = encode_examples(rows, device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        order = torch.randperm(len(rows), generator=generator)
        for start in range(0, len(rows), batch_size):
            ix = order[start:start + batch_size].to(device)
            batch = tuple(x[ix] for x in inputs)
            legality, effects = model(*batch)
            loss = F.binary_cross_entropy_with_logits(legality, labels[ix])
            # Changed squares matter much more than copying the many empty squares.
            legal = labels[ix].bool()
            if legal.any():
                target = after[ix][legal]
                changed = target != batch[0][legal]
                weights = 1.0 + 15.0 * changed.float()
                error = F.cross_entropy(effects[legal].transpose(1, 2), target, reduction="none")
                loss = loss + (error * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ix)
        history.append({"epoch": epoch + 1, "loss": total / len(rows)})
    return history


@torch.no_grad()
def evaluate(model, rows, batch_size=512):
    if not rows or batch_size < 1:
        raise ValueError("nonempty examples and positive batch size required")
    model.eval()
    device = next(model.parameters()).device
    tp = tn = fp = fn = exact = changed_correct = changed_total = legal_count = 0
    topics = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        inputs, labels, after = encode_examples(chunk, device)
        logits, effects = model(*inputs)
        prediction, truth = logits >= 0, labels.bool()
        tp += int((prediction & truth).sum())
        tn += int((~prediction & ~truth).sum())
        fp += int((prediction & ~truth).sum())
        fn += int((~prediction & truth).sum())
        boards = effects.argmax(-1)
        changed = (after != inputs[0]) & truth[:, None]
        changed_correct += int(((boards == after) & changed).sum())
        changed_total += int(changed.sum())
        exact += int(((boards == after).all(1) & truth).sum())
        legal_count += int(truth.sum())
        for row, correct in zip(chunk, (prediction == truth).tolist()):
            tally = topics.setdefault(row.topic, {"correct": 0, "total": 0})
            tally["correct"] += int(correct)
            tally["total"] += 1
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / len(rows)
    # No gate pass for a degenerate split containing only one class.
    passed = bool(tp + fn and tn + fp and min(accuracy, recall, specificity, precision) > .99)
    return {"examples": len(rows), "legal_examples": legal_count,
            "accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2,
            "legal_recall": recall, "illegal_recall": specificity, "legal_precision": precision,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "exact_board_effect_accuracy": exact / max(legal_count, 1),
            "changed_square_accuracy": changed_correct / max(changed_total, 1),
            "topics": topics, "rule_gate_passed": passed}
