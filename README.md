# FactorioGPT

A GPT made only of vanilla Factorio combinators. No mods and no Space Age.

It has 8,483,616 parameters in 100,093 entities. You type a question on an in-game keyboard, and it writes the answer on a screen of display panels, about three letters every 4.8 seconds. Every reply tested matches an exact Python simulation, letter for letter.

```
TELL ME A JOKE      WHY DID THE BELT STOP? IT WAS TIRED!
WHAT IS YOUR NAME?  I AM FACTO! A ROBOT MADE OF WIRES.
DO YOU LIKE IRON?   YES! IRON PLATES ARE MY FAVORITE THING.
I AM SAD            OH NO. I AM HERE WITH YOU.
```

A second, much smaller build trains itself inside the game: forward pass, backpropagation and SGD, all in combinators.

## Try it in the game

Everything is in [`game/`](game). It needs Factorio 2.1 or later.

- **factoriogpt.zip**: a save with both builds, powered and ready.
- **factoriogpt.txt**: the GPT as a blueprint string.
- **factoriogpt_trainer.txt**: the trainer as a blueprint string.

[`game/README.md`](game/README.md) says how to type a prompt and how to start the trainer.

## How it works

- **A wire is a vector.** A wire carries many signals at once, one per dimension.
- **A combinator is a neuron.** One arithmetic combinator set to `each × each` multiplies two vectors and adds the products. That is a dot product, one neuron. Its weights are stored in a constant combinator beside it.
- **Whole numbers only.** Wires carry 32-bit integers. Weights are 8-bit integers, and activations are stored x256. Softmax's exp and RMSNorm's 1/√x are lookup tables made of decider combinators.
- **Memory.** Each layer's keys and values are stored in memory cells, written once per token.
- **Clock.** A selector combinator reads the game tick. Every 287 ticks, the chosen token is fed back in and the position moves on.
- **Tokens.** A 768-token BPE vocabulary, one signal per token. Single letters are tokens too, so the prompt is typed one letter per key.

## The code

- `fgpt/`: the model, its exact integer simulator (`intsim.py`) and the circuit generator (`gpt_circuit.py`, `circuit.py`). `fixed.py` copies the game's integer math, as measured in the game (`notes/game_facts.md`).
- `fgpt/trainsim.py`, `fgpt/train_circuit.py`: the in-game trainer's integer reference and its circuit.
- `train/`: the chat data, the PyTorch model and the training script, with quantization-aware training that matches the circuit. `train/out/v3t/` holds the trained weights.
- `harness/`: runs a headless Factorio server, builds a circuit in it, and checks the game against the simulator.
- `tests/`: tests for the integer math and the tokenizer.

## Run it

Python 3.12, then:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Ask the simulator, no game needed:

```
.venv/bin/python -m fgpt.intsim "TELL ME A JOKE"
```

Train the model again (about 6 minutes on one H100; it writes `train/out/v3t/`):

```
.venv/bin/python train/train.py v3t --steps 12000 --qat-steps 3000
```

Build it in the game and check it against the simulator. These start a headless server from the Factorio binary, by default the macOS one. Set `FACTORIO` to use another.

```
.venv/bin/python harness/test_gpt.py v3t "TELL ME A JOKE"     # build the GPT, compare its replies
.venv/bin/python harness/paste_test.py v3t "TELL ME A JOKE"   # export the blueprint, paste it, compare again
.venv/bin/python harness/test_trainer.py 100                  # train 100 steps in game, compare every weight
.venv/bin/python harness/make_save.py                         # make game/factoriogpt.zip and check it
```

Tests:

```
.venv/bin/python -m pytest tests
```

`train/data/chat.jsonl` was generated with the Claude API by `train/data/make_chat.py`.
