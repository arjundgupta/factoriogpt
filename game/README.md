# Play it

Factorio 2.1 or later. The base game is enough.

- **factoriogpt.zip**: a save with both builds, powered and ready. Put it in your Factorio `saves` folder and load it.
- **factoriogpt.txt**: the GPT as a blueprint string, for your own world. It is 85 MB, so pasting takes a while.
- **factoriogpt_trainer.txt**: the trainer as a blueprint string.

The blueprints need power. The GPT's blueprint includes an electric energy interface, which works in editor mode or in a sandbox world. Otherwise, power its substations.

## Ask the GPT

You start at its keyboard: a row of 32 constant combinators. The first one on the left is RESET. The other 31 are keys.

1. In the save, RESET is already on. In your own world, turn it on.
2. Type the prompt into the keys, left to right, one character per key. In each key, set one signal to 1: a letter, a digit, or the `.`, `?` or `!` signal. For a space, use the horizontal line from the shapes tab.
3. Turn RESET off.

The prompt and then the reply appear on the screen, a block of display panels about 300 tiles south of the keyboard. Open the map to watch it. A reply comes about three letters every 4.8 seconds.

To ask again, turn RESET on, change the keys, and turn RESET off.

## Train the trainer

The trainer is south of the GPT. Its two switches are constant combinators at its top left corner: INIT, and TRAIN under it.

1. Turn INIT on, wait a second, and turn it off. This loads the starting random weights.
2. Turn TRAIN on. It trains on one word per step, about one step a second.

Its display is at the bottom of the build. The top row is the word being trained, the second row is the model's guess for each next letter, and the lamps fill as it gets more letters right. It gets about 74% of letters right by step 4,000. Turn TRAIN off to pause.
