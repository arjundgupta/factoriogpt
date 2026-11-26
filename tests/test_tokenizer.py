import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgpt import intsim, spec  # noqa: E402
from fgpt.tokenizer import LMAX, Tokenizer  # noqa: E402

TK = Tokenizer.load(Path(__file__).resolve().parents[1] / "fgpt" / "tokens_768.json")


def test_vocab_shape():
    assert TK.V == 768
    assert TK.tokens[:spec.V] == spec.CHARS
    assert max(len(t) for t in TK.tokens) <= LMAX
    assert len(set(TK.tokens)) == TK.V


def test_roundtrip():
    for s in ["HI THERE! WANT TO TALK ABOUT BELTS?", "I AM FACTO.", "  A  B ", "123 GO!!", ""]:
        assert TK.decode(TK.encode(s)) == s
    assert len(TK.encode("I LOVE IRON PLATES.")) < len("I LOVE IRON PLATES.") / 2


def test_dialogue_prompt_is_characters():
    ids = TK.encode_dialogue([{"user": "HI", "bot": "HELLO THERE!"}])
    assert ids[:4] == [spec.VOCAB["<"], spec.VOCAB["H"], spec.VOCAB["I"], spec.VOCAB[">"]]
    assert ids[-1] == spec.VOCAB["<"]


def test_signals_disjoint():
    sig = spec.token_signals(768)
    assert len(set(sig)) == 768
    assert not set(sig[spec.V:]) & set(spec.pos_signals(256))


def test_argmax_ties_lowest_id():
    lg = np.zeros(768, np.int64)
    lg[[5, 300, 700]] = 50
    assert intsim.argmax(lg, tok=True) == 5
    lg[700] = 51
    assert intsim.argmax(lg, tok=True) == 700
