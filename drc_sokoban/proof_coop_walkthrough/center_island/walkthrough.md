# center_island

## Two-agent layout (MABoxoban training)

The coop generator stores **one** `@` in the file; `MABoxobanEnv` strips it and spawns **A** (blue) and **B** (magenta) — see `initial_two_agent.png` / `.txt`. That is the actual two-body starting state for IPPO.

- Agent **A** (row,col) = `(4, 3)`  |  **B** = `(2, 4)`

## Single-agent shortest solution (solvability proof)

The generator checks **single-agent** solvability. The frames below replay one optimal **one-agent** path (not a joint two-agent schedule).

- generator seed: `10372`
- shortest 1-agent solution: **19** actions
- frames: every **3** steps (+ final); see `ascii/` and `png/`.

## Step 0 / 19

```
######
#   .#
#$ # #
# $# #
#. @ #
######
```

## Step 3 / 19

```
######
#   .#
#$$# #
#  # #
#.@  #
######
```

## Step 6 / 19

```
######
#   .#
#$$# #
#  #@#
#.   #
######
```

## Step 9 / 19

```
######
#  @.#
#$$# #
#  # #
#.   #
######
```

## Step 12 / 19

```
######
#   .#
#@$# #
#$ # #
#.   #
######
```

## Step 15 / 19

```
######
# $ .#
# @# #
#  # #
#*   #
######
```

## Step 18 / 19

```
######
# @$.#
#  # #
#  # #
#*   #
######
```

## Step 19 / 19

```
######
#  @*#
#  # #
#  # #
#*   #
######
```

