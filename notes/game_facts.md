# Game facts

Measured in Factorio 2.1.24, base game, with small probe circuits on a test server. The raw readings are in `m0_raw.json`, and `tests/test_fixed.py` checks `fgpt/fixed.py` against them.

## Combinators

- `each[red] × each[green] → one signal` multiplies and adds the products: a whole dot product in one combinator. x = (3, −1, 2) and w = (2, 4, 1) give 4.
- `each[red] × each[green] → each` multiplies elementwise.
- `each[red] × R[green] → each` multiplies a vector by one signal.
- `each ^ 2 → one signal` gives the sum of squares.
- A wire can carry 1,210 different signals, so a vector can be up to 1,210 wide.
- A constant combinator has 1,000 slots per section and can have several sections.
- A selector set to select max, index 0, outputs the largest signal with its value.
- Each combinator adds one tick of delay.

## Integer math

- Every value is a signed 32-bit integer. Overflow wraps: 2147483647 + 1 = −2147483648.
- `/` truncates toward zero: −7 / 2 = −3. Dividing by 0 gives 0.
- `%` takes the sign of the dividend: −7 % 2 = −1.
- `>>` is an arithmetic shift: −8 >> 1 = −4. On negative numbers it differs from `/`.
- Shift amounts are taken mod 32: 1 << 32 = 1.
- `^` wraps on overflow. A negative exponent gives 0.

## Power

Arithmetic, decider and selector combinators draw 1 kW each. Constant combinators and display panels draw nothing. Lamps draw 5 kW.

## Test server

Every console command is sent on to connected players, and big commands are throttled. A 2 MB command takes 0.09 s with nobody connected and about 250 s with one player connected. So paste big blueprints with nobody connected.
