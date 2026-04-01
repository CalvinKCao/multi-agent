# mutual_block

## Two-agent layout (MABoxoban training)

The coop generator stores **one** `@` in the file; `MABoxobanEnv` strips it and spawns **A** (blue) and **B** (magenta) — see `initial_two_agent.png` / `.txt`. That is the actual two-body starting state for IPPO.

- generator seed: `10288`
- shortest MA solution: **2** joint actions
- frames: every **3** steps (+ final); see `ascii/` and `png/`.

## Step 0 / 2

```
######
#A##B#
#$##$#
# ## #
#.  .#
######
```

## Step 2 / 2

```
######
# ## #
# ## #
#A##B#
#*  *#
######
```

