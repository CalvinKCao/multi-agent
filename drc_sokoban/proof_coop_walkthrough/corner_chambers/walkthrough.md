# corner_chambers

## Two-agent layout (MABoxoban training)

The coop generator stores **one** `@` in the file; `MABoxobanEnv` strips it and spawns **A** (blue) and **B** (magenta) — see `initial_two_agent.png` / `.txt`. That is the actual two-body starting state for IPPO.

- Agent **A** (row,col) = `(4, 4)`  |  **B** = `(1, 4)`

## Single-agent shortest solution (solvability proof)

The generator checks **single-agent** solvability. The frames below replay one optimal **one-agent** path (not a joint two-agent schedule).

- generator seed: `10582`
- shortest 1-agent solution: **10** actions
- frames: every **3** steps (+ final); see `ascii/` and `png/`.

## Step 0 / 10

```
######
#    #
# $. #
### ##
#.$ @#
######
```

## Step 3 / 10

```
######
#    #
# $. #
### ##
#* @ #
######
```

## Step 6 / 10

```
######
#  @ #
# $. #
### ##
#*   #
######
```

## Step 9 / 10

```
######
#    #
#@$. #
### ##
#*   #
######
```

## Step 10 / 10

```
######
#    #
# @* #
### ##
#*   #
######
```

