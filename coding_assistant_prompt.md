# Coding Task: Train a DRC-Style ConvLSTM Agent on Boxoban and Run Planning Probes

## What This Project Is

We are replicating and extending the methodology from Bush et al. (2025) "Interpreting Emergent
Planning in Model-Free Reinforcement Learning" (ICLR 2025). That paper showed that a model-free
DRC (Deep Repeated ConvLSTM) agent playing Sokoban develops internal linear representations of
its future plans — readable by simple logistic regression probes on the hidden state — with clean
spatial correspondence between the ConvLSTM state and the game board.

We are extending this to a **cooperative multi-agent setting** (two agents solving Sokoban-style
puzzles together) to test whether agent A's hidden state encodes agent B's future plans — emergent
Theory of Mind (ToM). But first we need to train a working single-agent DRC on Boxoban from
scratch to validate the architecture and probe pipeline before extending to multi-agent.

This document specifies everything the coding assistant needs to implement.

---

## Repositories and Data — Use Exactly These

### Environment
**Boxoban (official DeepMind repo):**
```
https://github.com/google-deepmind/boxoban-levels
```
This is the dataset only (900,000 Sokoban levels). You also need the Python environment wrapper:
```
https://github.com/google-deepmind/boxoban-environment
```
The environment wrapper includes a C extension for fast stepping. **Compile the C extension** —
do not use pure Python stepping, it is 5-10x slower.

If the boxoban-environment C extension fails to compile, fallback to:
```
pip install gym-sokoban
```
`gym-sokoban` (https://github.com/mpSchrader/gym-sokoban) uses a symbolic representation and
is pure Python but good enough for smoke testing.

### Reference Implementation
**Bush et al. codebase** — the authors have not released code, so we implement from scratch.
Read their paper carefully (attached as context). All architectural details are in Section 2.3 and
Appendix E.3 of the paper.

**Guez et al. 2019 (original DRC paper):**
```
https://github.com/google-deepmind/drc
```
This is the original DRC implementation in TensorFlow/Lua. Use it as a reference for architecture
details only — do NOT use it directly, implement in PyTorch.

### Training Algorithm
Use **IMPALA** (importance-weighted actor-learner) if you want distributed collection, or
**PPO** for simplicity. For a PoC, **PPO with vectorized environments is fine and much easier
to implement correctly.** Start with PPO.

Reference PPO implementation:
```
https://github.com/vwxyzjn/cleanrl
```
CleanRL's `ppo_atari.py` is the cleanest single-file PPO reference. Adapt it for the DRC
architecture replacing the CNN+LSTM with ConvLSTM as specified below.

---

## Project Structure

Create this exact directory layout:

```
drc_sokoban/
├── envs/
│   ├── boxoban_env.py          # Env wrapper with vectorized reset/step
│   └── make_env.py             # Factory for parallel envs
├── models/
│   ├── conv_lstm.py            # ConvLSTM cell and stack — THE CRITICAL FILE
│   └── agent.py                # Full agent (encoder + ConvLSTM + heads)
├── training/
│   ├── ppo.py                  # PPO trainer
│   └── rollout_buffer.py       # Rollout buffer that also stores hidden states
├── probing/
│   ├── hook_manager.py         # Forward hook extraction of hidden states
│   ├── concept_labeler.py      # CA and CB concept labeling (from Bush et al.)
│   ├── train_probes.py         # Logistic regression probe training
│   └── evaluate_probes.py      # F1 evaluation and results table
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── collect_for_probing.py  # Run trained agent and save hidden states
│   └── run_probes.py           # Full probe pipeline
├── checkpoints/                # Saved model weights
├── data/
│   ├── boxoban_levels/         # Symlink or copy of Boxoban dataset
│   └── probe_data/             # Collected trajectories for probing
└── results/                    # Probe metrics and figures
```

---

## CRITICAL: The ConvLSTM Architecture

**This is the most important part. Get this right before anything else.**

The entire reason we use ConvLSTM (not regular LSTM) is spatial correspondence:

```
Sokoban grid:      8 × 8  (game board cells)
ConvLSTM state:    8 × 8 × 32  (hidden state)
                   ↑ MUST MATCH EXACTLY
```

Every position (x, y) in the ConvLSTM hidden state must correspond to grid cell (x, y) on the
board. This is what enables 1×1 spatial probes — the core of Bush et al.'s methodology.

### Observation Encoding

The Sokoban symbolic observation is `(8, 8, 7)` — 7 binary channels encoding:
- Channel 0: wall
- Channel 1: empty floor
- Channel 2: box on floor
- Channel 3: box on target
- Channel 4: target (empty)
- Channel 5: agent on floor
- Channel 6: agent on target

Permute to PyTorch convention: `(7, 8, 8)` — (C, H, W).

The encoder must map `(7, 8, 8)` → `(G_0, 8, 8)` where G_0 is the number of input channels
to the first ConvLSTM layer.

**CRITICAL CONSTRAINT: NO striding, NO pooling, NO operations that change spatial dimensions.**
Use only same-padding convolutions with kernel_size=3, padding=1 to preserve 8×8.

```python
# CORRECT encoder — preserves 8×8
encoder = nn.Sequential(
    nn.Conv2d(7, 32, kernel_size=3, padding=1),  # (32, 8, 8) — preserved
    nn.ReLU(),
)

# WRONG — do NOT do this
encoder = nn.Sequential(
    nn.Conv2d(7, 32, kernel_size=3, stride=2),  # (32, 4, 4) — spatial dims destroyed!
)
```

### ConvLSTM Cell

Implement a standard ConvLSTM cell. Inputs and outputs are 3D tensors (C, H, W), not 1D vectors.

```python
class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.
    
    Operates on spatial tensors of shape (batch, channels, H, W).
    Hidden state (h, c) each have shape (batch, hidden_channels, H, W).
    
    CRITICAL: kernel_size=3, padding=1 ensures H and W are preserved.
    This is what gives spatial correspondence to the game grid.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same padding
        
        # Gates: input, forget, output, cell
        # Combined into one conv for efficiency
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x:      (batch, input_channels, H, W)
            h_prev: (batch, hidden_channels, H, W)
            c_prev: (batch, hidden_channels, H, W)
        Returns:
            h_next: (batch, hidden_channels, H, W)
            c_next: (batch, hidden_channels, H, W)
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        
        # Split gates
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # cell gate
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, H, W, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, H, W, device=device),
            torch.zeros(batch_size, self.hidden_channels, H, W, device=device),
        )
```

### DRC Stack (D layers, N ticks per step)

This is the key DRC innovation: N ticks of recurrent computation per environment step.

```python
class DRCStack(nn.Module):
    """
    Deep Repeated ConvLSTM (DRC) stack.
    
    Performs N ticks of recurrent computation per environment step.
    Each tick passes through all D ConvLSTM layers.
    
    From Guez et al. 2019 and Bush et al. 2025:
    - D = 3 layers (full), D = 2 (PoC minimum)
    - N = 3 ticks per step
    - hidden_channels = 32
    
    The hidden states (h, c) for all D layers are carried across BOTH
    ticks (within a step) AND environment steps (across steps in episode).
    This is what makes it recurrent over time.
    
    IMPORTANT for probing: we expose ALL intermediate tick hidden states,
    not just the final one. Bush et al.'s Fig 6 shows probe F1 improving
    across ticks — you need all of them.
    """
    def __init__(self, input_channels, hidden_channels, num_layers, num_ticks):
        super().__init__()
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.hidden_channels = hidden_channels
        
        # Build D ConvLSTM layers
        # Layer 0: takes encoder output as input
        # Layers 1..D-1: take previous layer's h as input
        self.cells = nn.ModuleList()
        for d in range(num_layers):
            in_ch = input_channels if d == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels))
    
    def forward(self, encoded_obs, hidden_states):
        """
        Args:
            encoded_obs:   (batch, input_channels, 8, 8) — encoder output
            hidden_states: list of D tuples [(h_d, c_d), ...], each (batch, 32, 8, 8)
        
        Returns:
            new_hidden_states: list of D tuples [(h_d, c_d), ...]
            all_tick_hiddens:  list of N lists, each containing D (h, c) tuples
                               Shape: [tick_0_states, tick_1_states, tick_2_states]
                               where tick_k_states = [(h_0, c_0), (h_1, c_1), ..., (h_D-1, c_D-1)]
                               
                               This is what you probe: all_tick_hiddens[tick][layer][0] gives
                               h at that tick and layer, shape (batch, 32, 8, 8).
        """
        h_states = [hs[0] for hs in hidden_states]
        c_states = [hs[1] for hs in hidden_states]
        
        all_tick_hiddens = []
        
        for tick in range(self.num_ticks):
            tick_hiddens = []
            x = encoded_obs  # reset input to encoder output at each tick
            
            for d, cell in enumerate(self.cells):
                h_new, c_new = cell(x, h_states[d], c_states[d])
                h_states[d] = h_new
                c_states[d] = c_new
                tick_hiddens.append((h_new, c_new))
                x = h_new  # next layer's input is this layer's h
            
            all_tick_hiddens.append(tick_hiddens)
        
        new_hidden_states = [(h_states[d], c_states[d]) for d in range(self.num_layers)]
        return new_hidden_states, all_tick_hiddens
    
    def init_hidden(self, batch_size, H=8, W=8, device='cpu'):
        return [
            cell.init_hidden(batch_size, H, W, device)
            for cell in self.cells
        ]
```

### Full Agent

```python
class DRCAgent(nn.Module):
    """
    Full DRC agent for Sokoban/Boxoban.
    
    Architecture:
        obs (7, 8, 8) → encoder → (32, 8, 8) → DRC stack (D layers, N ticks)
        → final h from last layer/last tick → flatten → policy head + value head
    
    Hyperparameters matching Bush et al.:
        encoder_channels = 32
        hidden_channels  = 32  (Gd in the paper)
        num_layers       = 3   (D in the paper), use 2 for PoC
        num_ticks        = 3   (N in the paper)
    """
    def __init__(self, obs_channels=7, hidden_channels=32, num_layers=3,
                 num_ticks=3, num_actions=4, H=8, W=8):
        super().__init__()
        self.H = H
        self.W = W
        self.num_actions = num_actions
        
        # Encoder: obs → spatial features, SAME spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # DRC recurrent stack
        self.drc = DRCStack(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_ticks=num_ticks,
        )
        
        # Output heads: flatten final h then linear
        flat_dim = hidden_channels * H * W  # 32 * 8 * 8 = 2048
        self.policy_head = nn.Linear(flat_dim, num_actions)
        self.value_head = nn.Linear(flat_dim, 1)
    
    def forward(self, obs, hidden_states, return_all_ticks=False):
        """
        Args:
            obs:           (batch, 7, 8, 8) — symbolic Sokoban observation
            hidden_states: list of D tuples [(h, c), ...] carried from previous step
            return_all_ticks: if True, also return all intermediate tick hiddens
                              (needed during probe data collection, not during training)
        
        Returns:
            logits:            (batch, num_actions)
            value:             (batch, 1)
            new_hidden_states: list of D tuples for next step
            all_tick_hiddens:  (only if return_all_ticks=True) list of N tick states
        """
        # Encode observation
        encoded = self.encoder(obs)  # (batch, 32, 8, 8) — spatial dims preserved
        
        # Run DRC ticks
        new_hidden_states, all_tick_hiddens = self.drc(encoded, hidden_states)
        
        # Use final layer's h from final tick for action selection
        final_h = all_tick_hiddens[-1][-1][0]  # [last_tick][last_layer][h]
        flat = final_h.flatten(start_dim=1)   # (batch, 2048)
        
        logits = self.policy_head(flat)
        value = self.value_head(flat)
        
        if return_all_ticks:
            return logits, value, new_hidden_states, all_tick_hiddens
        return logits, value, new_hidden_states
    
    def init_hidden(self, batch_size, device='cpu'):
        return self.drc.init_hidden(batch_size, self.H, self.W, device)
```

---

## Training Setup

### Hyperparameters

Use these exactly. They are tuned for Boxoban on a single GPU.

```python
# Environment
NUM_ENVS        = 32         # parallel environments
HORIZON         = 400        # max steps per episode
OBS_SHAPE       = (7, 8, 8)  # symbolic Sokoban observation
NUM_ACTIONS     = 4          # up, down, left, right

# Architecture  
HIDDEN_CHANNELS = 32
NUM_LAYERS      = 2          # use 2 for PoC speed, 3 for full experiment
NUM_TICKS       = 3

# PPO
LEARNING_RATE   = 3e-4
GAMMA           = 0.97       # Bush et al. value for Sokoban
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.1
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.01
MAX_GRAD_NORM   = 10.0       # higher than typical due to LSTM hidden state gradients
PPO_EPOCHS      = 4
MINIBATCH_SIZE  = 8          # batch of envs per PPO update (keep small for hidden state mgmt)
ROLLOUT_STEPS   = 20         # steps per env before PPO update

# Training budget
TARGET_STEPS    = 50_000_000  # 50M for PoC, 250M for full

# Checkpointing
SAVE_EVERY_STEPS = 5_000_000  # save checkpoint every 5M steps
```

### Critical: Hidden State Management Across Steps

This is the most common implementation mistake with ConvLSTM agents in PPO. The hidden state
must be correctly carried across environment steps AND correctly reset at episode boundaries.

```python
# In your rollout buffer / PPO loop:

# At episode start or after done=True:
hidden = agent.init_hidden(batch_size=NUM_ENVS, device=device)

# At each step:
with torch.no_grad():
    logits, value, new_hidden = agent(obs, hidden)

# CRITICAL: reset hidden state for envs that just finished
# dones shape: (NUM_ENVS,) bool tensor
for d in range(agent.drc.num_layers):
    h, c = new_hidden[d]
    # Zero out hidden state for completed episodes
    mask = (~dones).float().view(-1, 1, 1, 1)  # (NUM_ENVS, 1, 1, 1)
    new_hidden[d] = (h * mask, c * mask)

hidden = new_hidden

# During PPO update: DO NOT backprop through hidden states across rollout boundaries
# Detach hidden states when computing PPO loss:
# hidden_detached = [(h.detach(), c.detach()) for h, c in hidden_states_from_rollout]
```

### Boxoban Dataset Setup

```python
# Download and setup Boxoban levels:
# git clone https://github.com/google-deepmind/boxoban-levels
# Structure:
#   boxoban-levels/unfiltered/train/*.txt   (train set, 900k levels)
#   boxoban-levels/unfiltered/valid/*.txt   (validation set)
#   boxoban-levels/medium/train/*.txt       (harder filtered levels)

# Bush et al. trained on the UNFILTERED training set.
# Use unfiltered/train for training, unfiltered/valid for probe evaluation.

# Each .txt file contains multiple Sokoban levels in the format:
# ; Level N
# ########
# # @ .  #
# #  $   #
# ########
# (standard Sokoban notation: # wall, @ agent, $ box, . target, * box on target, + agent on target)
```

### Training Monitoring

Print these metrics every 1M steps. Stop training early if the solve rate is not improving:

```
Step: 10M | Solve rate: 0.12 | Mean reward: 0.48 | Entropy: 1.21 | VF loss: 0.34
Step: 20M | Solve rate: 0.31 | Mean reward: 0.87 | ...
```

A solve rate of >0.15 at 20M steps indicates the model is learning. If solve rate is <0.05 at
20M steps, something is wrong — check hidden state reset logic first.

---

## Probe Concepts (from Bush et al. — implement exactly these)

After training, collect rollout data and label each timestep with these concepts:

### Concept CA — Agent Approach Direction

For each grid square (x, y) at timestep t, the label is:
- `UP` if the next time the agent steps onto (x, y), it comes from below (agent was at y+1)
- `DOWN` if agent comes from above
- `LEFT` if agent comes from right
- `RIGHT` if agent comes from left
- `NEVER` if the agent never steps onto (x, y) again in this episode

This requires looking ahead through the rest of the episode to find the agent's next visit to (x, y).

### Concept CB — Box Push Direction

For each grid square (x, y) at timestep t, the label is:
- `UP` if the next time a box is pushed off (x, y), it goes up (box was pushed upward)
- `DOWN` / `LEFT` / `RIGHT` similarly
- `NEVER` if no box is pushed off (x, y) again in this episode

### Implementation Notes for Labeling

```python
def label_episode(states, actions):
    """
    states: list of full game states for each timestep (from env.get_state() or similar)
    actions: list of actions taken
    
    Returns:
        ca_labels: (T, 8, 8) array of int labels {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NEVER}
        cb_labels: (T, 8, 8) array of int labels
    """
    T = len(states)
    UP, DOWN, LEFT, RIGHT, NEVER = 0, 1, 2, 3, 4
    
    ca_labels = np.full((T, 8, 8), NEVER, dtype=np.int32)
    cb_labels = np.full((T, 8, 8), NEVER, dtype=np.int32)
    
    # For each timestep t, find the NEXT time each square is visited/pushed
    # Look forward through the episode from t
    
    # Agent position at each step
    agent_positions = [get_agent_pos(s) for s in states]
    
    # Box positions at each step (list of sets)
    box_positions = [get_box_positions(s) for s in states]
    
    for t in range(T):
        for x in range(8):
            for y in range(8):
                # Find next agent visit to (x, y) after timestep t
                for t2 in range(t + 1, T):
                    if agent_positions[t2] == (x, y):
                        # Agent came from the direction of their previous position
                        prev_pos = agent_positions[t2 - 1]
                        direction = get_direction(prev_pos, (x, y))
                        ca_labels[t, y, x] = direction
                        break
                
                # Find next box push off (x, y) after timestep t
                for t2 in range(t + 1, T):
                    if (x, y) in box_positions[t2 - 1] and (x, y) not in box_positions[t2]:
                        # Box was pushed off (x, y) at step t2
                        # Direction = where the box went
                        new_box_pos = find_moved_box(box_positions[t2-1], box_positions[t2], (x,y))
                        direction = get_direction((x, y), new_box_pos)
                        cb_labels[t, y, x] = direction
                        break
    
    return ca_labels, cb_labels
```

**Note:** This labeling is O(T × 64 × T) = O(T²) per episode. For T=400 episodes this is 
manageable (~10M ops per episode). Parallelize across episodes using multiprocessing if slow.

---

## Probe Training (follow Bush et al. exactly)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def train_spatial_probe(hidden_states, labels, x, y, layer, tick,
                         n_seeds=5):
    """
    Trains a 1×1 spatial probe at position (x, y) for a specific concept.
    
    hidden_states: (N_samples, num_ticks, num_layers, 32, 8, 8)
    labels:        (N_samples, 8, 8) — concept labels per grid position
    x, y:          grid position to probe
    layer:         which ConvLSTM layer (0, 1, 2)
    tick:          which tick (0, 1, 2)
    
    The 1×1 probe extracts ONLY the 32-dim activation at position (x, y).
    This is what gives spatial grounding — we're only using local information.
    """
    # Extract 32-dim feature at position (x, y)
    # hidden_states[:, tick, layer, :, y, x] shape: (N_samples, 32)
    X = hidden_states[:, tick, layer, :, y, x]  # (N, 32)
    y_labels = labels[:, y, x]                  # (N,)
    
    # Remove NEVER-class samples for directional probe F1
    # (or keep them — Bush et al. keep them and use macro F1)
    
    best_f1 = -1
    best_probe = None
    
    for seed in range(n_seeds):
        probe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=seed,
                C=1.0,
                class_weight='balanced',
                multi_class='auto',
            ))
        ])
        probe.fit(X_train, y_train)
        preds = probe.predict(X_val)
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_probe = probe
    
    return best_probe, best_f1


def train_all_probes(hidden_states, ca_labels, cb_labels,
                     num_ticks=3, num_layers=3):
    """
    Trains probes for ALL positions, ALL layers, ALL ticks.
    Also trains observation-only baseline probes.
    
    Returns results dict matching Bush et al. Figure 4 structure:
    {
        'CA': {(layer, tick): mean_macro_f1_across_positions},
        'CB': {(layer, tick): mean_macro_f1_across_positions},
        'CA_baseline': float,   # obs-only probe F1
        'CB_baseline': float,
    }
    """
    results = {'CA': {}, 'CB': {}}
    
    for layer in range(num_layers):
        for tick in range(num_ticks):
            ca_f1s, cb_f1s = [], []
            for y in range(8):
                for x in range(8):
                    _, f1_ca = train_spatial_probe(
                        hidden_states, ca_labels, x, y, layer, tick
                    )
                    _, f1_cb = train_spatial_probe(
                        hidden_states, cb_labels, x, y, layer, tick
                    )
                    ca_f1s.append(f1_ca)
                    cb_f1s.append(f1_cb)
            
            results['CA'][(layer, tick)] = np.mean(ca_f1s)
            results['CB'][(layer, tick)] = np.mean(cb_f1s)
            print(f"Layer {layer}, Tick {tick}: CA F1={results['CA'][(layer,tick)]:.3f}, "
                  f"CB F1={results['CB'][(layer,tick)]:.3f}")
    
    return results
```

### Data Split

**Always split by episode, never by timestep.** Adjacent timesteps within an episode are
highly correlated and will cause probe leakage.

```python
# Split: 70% train, 15% val, 15% test
# By episode index, not by timestep
train_eps = episodes[:int(0.7 * N)]
val_eps   = episodes[int(0.7*N):int(0.85*N)]
test_eps  = episodes[int(0.85*N):]
```

---

## Probe Data Collection

Run the trained agent for 3000 training episodes + 1000 validation episodes (matching Bush et al.)
and save:

```python
def collect_probe_data(agent, env, n_episodes, save_path, device):
    """
    Runs agent and saves all data needed for probing.
    
    Saves per episode:
    - observations[t]:     (7, 8, 8) tensor
    - hidden_states[t]:    (num_ticks, num_layers, 32, 8, 8) — ALL ticks, ALL layers
    - actions[t]:          int
    - agent_positions[t]:  (x, y) tuple
    - box_positions[t]:    list of (x, y) tuples
    - reward[t]:           float
    - done[t]:             bool
    - solved:              bool (did agent solve the level)
    """
    all_episodes = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        hidden = agent.init_hidden(batch_size=1, device=device)
        
        episode = {
            'observations': [], 'hidden_states': [],
            'actions': [], 'agent_positions': [],
            'box_positions': [], 'rewards': [], 'dones': [],
        }
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value, new_hidden, all_tick_hiddens = agent(
                    obs_t, hidden, return_all_ticks=True
                )
            
            # Save hidden states from ALL ticks and ALL layers
            # Shape we want: (num_ticks, num_layers, 32, 8, 8)
            tick_layer_h = []
            for tick_hiddens in all_tick_hiddens:
                layer_h = []
                for (h, c) in tick_hiddens:
                    layer_h.append(h.squeeze(0).cpu().numpy())  # (32, 8, 8)
                tick_layer_h.append(layer_h)
            
            episode['hidden_states'].append(np.array(tick_layer_h))  # (T, N, D, 32, 8, 8)
            episode['observations'].append(obs)
            
            # Get game state for labeling
            episode['agent_positions'].append(env.get_agent_pos())
            episode['box_positions'].append(env.get_box_positions())
            
            # Step
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, info = env.step(action)
            
            # Reset hidden if done
            mask = torch.FloatTensor([0.0 if done else 1.0]).view(1,1,1,1).to(device)
            new_hidden = [(h * mask, c * mask) for h, c in new_hidden]
            hidden = new_hidden
            
            episode['actions'].append(action)
            episode['rewards'].append(reward)
            episode['dones'].append(done)
        
        # Convert to arrays
        for key in ['observations', 'hidden_states', 'actions', 'rewards', 'dones']:
            episode[key] = np.array(episode[key])
        
        episode['solved'] = bool(np.sum(episode['rewards']) > 0)
        all_episodes.append(episode)
        
        if ep % 100 == 0:
            solve_rate = np.mean([e['solved'] for e in all_episodes])
            print(f"Episode {ep}/{n_episodes} | Solve rate: {solve_rate:.3f}")
    
    with open(save_path, 'wb') as f:
        pickle.dump(all_episodes, f)
    print(f"Saved {n_episodes} episodes to {save_path}")
```

---

## Expected Results to Verify the Pipeline is Working

Run these checks in order. Stop and debug if any check fails before continuing.

### Check 1: Architecture Sanity (before any training)
```python
agent = DRCAgent(num_layers=2, num_ticks=3, hidden_channels=32)
obs = torch.randn(4, 7, 8, 8)        # batch of 4
hidden = agent.init_hidden(4, 'cpu')
logits, value, new_hidden, ticks = agent(obs, hidden, return_all_ticks=True)

assert logits.shape == (4, 4),                    "Policy head wrong shape"
assert value.shape == (4, 1),                     "Value head wrong shape"
assert len(new_hidden) == 2,                      "Should have D=2 layer states"
assert new_hidden[0][0].shape == (4, 32, 8, 8),  "CRITICAL: spatial dims must be 8x8"
assert len(ticks) == 3,                           "Should have N=3 ticks"
assert ticks[0][0][0].shape == (4, 32, 8, 8),    "Tick hidden state must be 8x8"
print("CHECK 1 PASSED: Architecture is spatially correct")
```

### Check 2: Training Progress (at 10M steps)
- Solve rate should be >0.10 on unfiltered Boxoban
- If solve rate is <0.02, hidden state reset logic is broken

### Check 3: Probe Sanity (before full probe training)
- Train a probe for 10 positions only on 500 episodes
- CA probe on cell states should beat obs-only baseline by >0.05 F1
- If it doesn't, either agent hasn't learned enough (train more) or hidden state
  extraction is wrong (check tick/layer indexing)

### Check 4: Main Result (matching Bush et al. Figure 4)
Expected approximate F1 scores at 250M training steps, 3000 probe training episodes:
```
           Layer 0   Layer 1   Layer 2   Obs-baseline
CA probe:   0.55      0.60      0.62        0.35
CB probe:   0.45      0.52      0.55        0.28
```
At 50M steps (PoC) expect roughly 60-70% of these values. The obs-baseline should be
the same regardless of training — if it changes something is wrong.

---

## Smoke Test Script

Run this first before starting full training. Should complete in under 5 minutes.

```bash
python scripts/train.py \
    --num-envs 4 \
    --target-steps 100000 \
    --num-layers 2 \
    --hidden-channels 16 \
    --save-path checkpoints/smoke_test.pt \
    --smoke-test

# Then verify hidden state extraction:
python scripts/collect_for_probing.py \
    --checkpoint checkpoints/smoke_test.pt \
    --n-episodes 10 \
    --save-path data/probe_data/smoke_test.pkl

# Then run one probe to verify shapes:
python scripts/run_probes.py \
    --data data/probe_data/smoke_test.pkl \
    --n-positions 5 \
    --quick-check
```

---

## Dependency List

```
torch>=2.0.0
numpy<2.0.0          # numpy 2.0 breaks some gym environments
scikit-learn>=1.3.0
gymnasium>=0.29.0
gym-sokoban           # fallback if boxoban-environment C build fails
matplotlib
pandas
tqdm
pickle5               # for large pickle files
```

For the Boxoban C extension:
```bash
cd boxoban-environment
pip install -e .
# If build fails, check that gcc is installed: sudo apt-get install build-essential
```

---

## What NOT to Do

- **DO NOT** use striding or pooling in the encoder. One mistake here destroys the entire
  spatial correspondence that makes the probing methodology work.
- **DO NOT** flatten the hidden state before the output heads. Keep it spatial until the
  very last step (flatten just before the linear layers).
- **DO NOT** split probe train/test by timestep — split by episode only.
- **DO NOT** use accuracy as the probe metric — use macro F1 (class imbalance from NEVER class).
- **DO NOT** reset hidden state between ticks within a step — only between episodes.
- **DO NOT** start probe training until the agent's solve rate is >0.10. Probing an agent
  that hasn't learned anything will give meaningless results.
- **DO NOT** backpropagate through the hidden states across rollout boundaries in PPO.
  Detach the initial hidden state at the start of each PPO update.
