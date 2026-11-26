"""Train a FactorioGPT model: float pretraining, then QAT, then integer export and checks.

  .venv/bin/python train/train.py v3t --steps 12000 --qat-steps 3000
Outputs train/out/<name>/{model.pt, int_params.pkl, config.json, report.json}
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
from fgpt import intsim, spec  # noqa: E402
from model import GPT  # noqa: E402

DATA = ROOT / "train" / "data" / "chat.jsonl"


def load_sequences(T: int, tk=None):
    """One sequence per dialogue. Character builds: one token per character. Tokenizer builds: prompts
    stay one character per token, replies are BPE. The same dialogues are kept either way (filtered by
    character length), so validation sets match across builds."""
    seqs = []
    for line in DATA.read_text().splitlines():
        turns = json.loads(line)["turns"]
        s = "".join(spec.USER + t["user"] + spec.BOT + t["bot"] for t in turns) + spec.USER
        if len(s) < 8:
            continue
        ids = tk.encode_dialogue(turns) if tk else spec.encode(s)
        seqs.append(ids[: T + 1])
    return seqs


def batches(seqs, B, T, device, rng):
    while True:
        idx = rng.sample(range(len(seqs)), B)
        Tb = max(len(seqs[j]) - 1 for j in idx)   # pad only to the longest sequence in the batch
        Tb = min(T, (Tb + 15) // 16 * 16)
        x = torch.full((B, Tb), 0, dtype=torch.long)
        y = torch.full((B, Tb), -1, dtype=torch.long)
        for i, j in enumerate(idx):
            s = seqs[j]
            n = len(s) - 1
            x[i, :n] = torch.tensor(s[:-1])
            y[i, :n] = torch.tensor(s[1:])
        yield x.to(device), y.to(device)


def evaluate(model, seqs, T, device, qat, n=400, per_char=False):
    """Mean loss per token. per_char=True also returns loss per character of text predicted, which is
    comparable between character and tokenizer builds."""
    model.eval()
    lens = getattr(model, "tok_lens", None)
    rng = random.Random(1)
    tot, cnt, chars = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(n // 50):
            x, y = next(batches(seqs, 50, T, device, rng))
            logits = model(x, qat=qat)
            loss = Fn.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="sum")
            tot += loss.item()
            cnt += (y >= 0).sum().item()
            yy = y[y >= 0]
            chars += int(lens.to(yy.device)[yy].sum().item()) if lens is not None else yy.numel()
    model.train()
    return (tot / cnt, tot / chars) if per_char else tot / cnt


def train_phase(model, seqs, val, cfg, device, steps, lr, qat, B, log):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = random.Random(0 if not qat else 7)
    it = batches(seqs, B, cfg.T, device, rng)
    t0 = time.time()
    for step in range(steps):
        cur = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        x, y = next(it)
        logits = model(x, qat=qat)
        loss = Fn.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            v = evaluate(model, val, cfg.T, device, qat)
            log(f"{'qat' if qat else 'float'} step {step} loss {loss.item():.3f} val {v:.3f} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--qat-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    cfg = spec.CONFIGS[args.name]
    out = ROOT / "train" / "out" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    torch.manual_seed(0)
    tk = spec.tokenizer(cfg)
    seqs = load_sequences(cfg.T, tk)
    random.Random(0).shuffle(seqs)
    nval = max(200, len(seqs) // 50)
    val, tr = seqs[:nval], seqs[nval:]
    log(f"{cfg}: {len(tr)} train / {len(val)} val sequences")
    model = GPT(cfg).to(args.device)
    model.tok_lens = torch.tensor([len(t) for t in tk.tokens]) if tk else None
    nparam = sum(p.numel() for p in model.parameters())
    log(f"params {nparam}")
    train_phase(model, tr, val, cfg, args.device, args.steps, args.lr, False, args.batch, log)
    model.freeze_shifts()
    log(f"shifts {model.shifts}")
    log(f"qat val before finetune {evaluate(model, val, cfg.T, args.device, True):.3f}")
    train_phase(model, tr, val, cfg, args.device, args.qat_steps, args.lr / 4, True, args.batch, log)
    torch.save(model.state_dict(), out / "model.pt")
    cfg.save(out / "config.json")
    P = model.export_int()
    pickle.dump(P, open(out / "int_params.pkl", "wb"))

    # agreement: QAT (float64, CPU) vs integer simulator on validation prefixes
    model_cpu = GPT(cfg).double()
    model_cpu.load_state_dict({k: v.detach().cpu().double() for k, v in model.state_dict().items()})
    model_cpu.shifts = model.shifts
    agree = total = 0
    for s in val[:40]:
        x = torch.tensor([s[:-1]])
        with torch.no_grad():
            ql = model_cpu(x, qat=True)[0].argmax(-1).tolist()
        st = intsim.new_state(cfg)
        for p, tok in enumerate(s[:-1]):
            il = intsim.argmax(intsim.step(P, cfg, st, tok, p), tok=bool(cfg.tok))
            agree += int(il == ql[p])
            total += 1
    log(f"intsim vs QAT argmax agreement {agree}/{total} = {agree / total:.3%}")
    samples = {}
    for prompt in ["HI", "HELLO HOW ARE YOU?", "WHAT IS YOUR NAME?", "DO YOU LIKE IRON?", "TELL ME A JOKE", "I AM SAD"]:
        samples[prompt] = intsim.generate(P, cfg, prompt)
        log(f"  {samples[prompt]}")
    vt, vc = evaluate(model, val, cfg.T, args.device, True, n=2000, per_char=True)
    log(f"val QAT loss per token {vt:.4f}, per character {vc:.4f}")
    json.dump({"params": nparam, "agreement": agree / total, "samples": samples,
               "val_qat": vt, "val_qat_per_char": vc}, open(out / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
