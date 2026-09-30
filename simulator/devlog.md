# CR-Bot Devlog: From Computer Vision to Reinforcement Learning

When Flavortown ended, **CR-Bot** was mainly a computer vision project.

My motivation came partly from playing Clash Royale myself. I thought it was a great game for an AI project because no particularly strong AI seemed to exist for it, most relevant information is visible to the player, and the game requires a mixture of timing, positioning, resource management and strategy.

At the end of Flavortown, the bot could already watch gameplay, detect troops with **YOLO**, read the timer, elixir, cards and tower health, and turn this into a structured game state. I had also built an **18 × 32 arena grid**, legal action masks, enemy card tracking and the beginnings of an expert-gameplay pipeline.

My original plan was:

1. Record strong **2.6 Hog Cycle** players.
2. Detect what happened in every frame.
3. Extract their actions.
4. Train a model to imitate them.
5. Improve it later with reinforcement learning.

After Flavortown, I started working on the missing part: reliably detecting what the player actually did.

## May: figuring out what the player actually played

Detecting the game state was only half of the problem. Training also required an accurate record of every action, for example:

```text
PLAY
Hog Rider
cell (3, 23)
```

I started combining several signals. A card disappearing from the hand was useful evidence, but it did not necessarily mean that the card had been played. A player could drag and cancel it, an emote could cover part of the interface, or the detector could simply miss a frame.

**Elixir changes** became another important signal. If a four-elixir card disappeared and the elixir count also dropped by about four, the evidence became much stronger.

I added checks for hand state, timing and deployment positions, together with special handling for cards such as **The Log** and **Barbarian Barrel**.

Spells were especially difficult. A troop normally remains visible after deployment, while a Fireball or Zap may only appear for a few frames. Their effects can also overlap with attacks already happening in the arena.

This made me question the original training plan. If my extracted actions were wrong, the network would learn from **incorrect labels**. A larger dataset would not necessarily solve that. It could simply make the model better at learning my annotation mistakes.

## June: measuring the extractor instead of trusting it

I started building proper evaluation tools for the action extractor.

I created ground-truth examples by labeling every troop placement from the enemy and the expert frame-by-frame. Additionally I added replay support and debugging tools for action misses and grid calibration.

Replay support was particularly useful because I could run the same match through different versions of the extractor without recording new gameplay each time.

I also experimented with improved spell tracking and other signals such as audio.

The main problem remained that **spells were especially difficult to detect reliably**. Using **KataCR detection**, cards such as **Zap, Arrows, Lightning and Clone** were often missed or detected incorrectly because their effects only appeared briefly and could overlap with other actions in the arena.

Building a large expert dataset from ordinary gameplay therefore required reliably determining:

* which side played a card
* which card was played
* exactly when it happened
* where it was placed
* whether the action was cancelled
* whether a brief effect was actually a spell
* whether overlapping units were separate deployments

Every mistake could become a wrong training example.

## August: pushing the annotation system further

I did not immediately abandon video-based training.

Instead, I divided annotation into smaller tasks. Own card plays could use hand and elixir changes, enemy troops could use deployment markers and movement over several frames, and spells could be analysed through short sequences instead of single images.

I also recorded dedicated footage for spells that the existing detector did not handle reliably, including **Zap, Arrows, Lightning and Clone**.

The system became more sophisticated, but this also revealed a larger problem: I was building a complicated machine-learning system just to generate the dataset required for another machine-learning system.

# The big change: build Clash Royale itself

I knew that rebuilding Clash Royale alone would normally be unrealistic. There are too many mechanics, interactions and edge cases.

However, the recent improvements in LLMs made me wonder whether I could put an AI into an automated development loop and gradually build and test a simulator.

Because of that, the project changed from:

```text
expert video
  -> vision
  -> estimated state
  -> estimated expert action
  -> behavior cloning
```

to:

```text
simulator
  -> exact state
  -> policy chooses action
  -> simulator applies action
  -> reward
  -> reinforcement learning
```

The **vision system still matters**, because the final policy must eventually operate on information that can also be obtained from the real game.

## Building and testing the simulator

I did not want a simulator that merely looked approximately correct.

A reinforcement-learning agent can exploit bugs that a human might never notice. If one placement, collision or tower interaction is slightly wrong, the model may discover that mistake and build its strategy around it.

For that reason, the simulator was designed around **determinism, reproducibility and testing**.

Identical starting states and actions should produce identical results. Generated tests cover movement, targeting, spawning, projectiles, death effects, status effects, transformations and many card-specific mechanics.

I also learned to distinguish between two kinds of correctness.

A simulator can be **internally consistent** and still be wrong compared with Clash Royale.

Passing thousands of deterministic tests proves that the implementation behaves consistently. It does not prove that the real game behaves the same way.

My first idea for solving this was a **physical lab** using two phones. The computer would create controlled situations in the real game and compare movement, timing, damage and targeting against the simulator. In practice, automatically reproducing identical situations on real phones was too unreliable and slow.

Instead, I used an automated AI audit loop. The model repeatedly researched card statistics and mechanics online, compared them against the simulator, found discrepancies and fixed them.

For well-documented values, it could compare against public statistics. For less documented mechanics, it searched gameplay discussions and other sources.

I ran this process repeatedly for roughly a full day.

This still does not prove that the simulator is perfect. Clash Royale contains many edge cases, and online information can be incomplete or wrong. There are almost certainly bugs remaining.

The simulator is therefore still deliberately marked as **provisional**.

# Building the actual policy

Once the simulator became usable, I could focus on the neural network.

I wanted the policy to **learn strategy** instead of having Clash Royale strategy hard-coded into Python.

For example, I did not want:

```text
if Balloon:
    play Musketeer
```

That might improve results quickly, but then I would be writing the strategy myself.

Instead, the actor should learn when Musketeer is useful, when another response is better, and when waiting is the correct decision.

## What the actor actually receives

One thing I recently inspected more carefully was the exact boundary between the simulator and the neural network.

The actor does not simply receive a Python object containing "cards, troops and tower health". It receives several fixed-size numerical tensors.

At every decision, the main public inputs are:

```text
board          21 × 32 × 18
global vector  768 values
entity tokens  up to 128 × 32
legal plays    4 × 32 × 18
GRU memory     previous hidden state
```

The **21-channel board raster** contains arena geometry such as the river, bridges and tower locations, together with spatial maps for visible troops, air units, HP, threat and living towers.

The **768-value global vector** contains more general information. This includes:

* own elixir
* a **publicly estimated opponent elixir**
* remaining time and overtime
* tower HP
* destroyed towers
* activated King Towers
* the exact card in each of the four hand slots
* the player's next card
* which enemy cards have already been seen

The opponent elixir is important because it is **not taken directly from the simulator's hidden state**. It is estimated from public information such as elapsed time and observed card plays.

The actor also receives up to **128 public entity tokens**. Each visible entity can contain information such as:

* card identity
* ally or enemy
* X and Y position
* HP fraction
* air, building, tower or spell type
* lane
* distance from friendly and enemy towers

The entity format reserves additional fields for information such as velocity, attacking state, stun state, recent damage and age. However, many of these are currently **zero-filled** because the public observation system does not yet expose them safely.

This is useful because it prevents the simulator from accidentally leaking hidden information into the actor.

Finally, the actor receives an exact **legal-action mask** for the four hand slots and every possible placement cell. This prevents impossible actions such as playing a card without enough elixir or deploying a building somewhere illegal.

The mask only removes actions that are physically impossible. It does **not** say which legal action is strategically good.

## Memory

The actor does not receive a separate list of recent cards, actions or troop positions. Instead, it uses a **GRU** to maintain a hidden state that is updated after every observation, allowing information from earlier decisions to remain available later in the match. A **Transformer** processes the visible entity tokens, while the GRU carries information through time.

The architecture is roughly:

```text
current public state
        |
        v
CNN + global features + entity Transformer
        |
        v
       GRU
        |
        v
action heads
```

This gives the policy access to both the **current battlefield** and information it has learned to remember from earlier states.

## Choosing an action

The action is factorised into three stages. First, the actor chooses between:

```text
WAIT or PLAY
```

If it chooses PLAY, it selects one of the four hand slots and then chooses a placement cell:

```text
choose one of 4 hand slots
        |
        v
choose a placement cell
```

This is much smaller than representing every card and position as one enormous action space. It also makes debugging easier because I can inspect separately whether a mistake came from **timing, card choice or placement**.

# A privileged critic

Training also uses a separate **critic**. The actor represents the real deployed bot and receives only public information, while the critic is used during training and can access additional exact simulator state. Its job is to estimate how good or bad the current situation is, giving PPO a stronger learning signal.

The important rule is that **private simulator information never enters actor inference**. After training, the critic is not needed to choose actions.

# Starting reinforcement learning

I implemented **PPO**, or Proximal Policy Optimization, to train the policy. The complete loop is:

```text
simulator
-> public observation
-> neural network
-> action
-> simulator
-> reward
-> PPO update
-> improved policy
```

The initial goal was not to create a strong Clash Royale AI immediately, but to prove that the entire learning loop worked correctly. A curriculum gradually increases the difficulty, moving from basic mechanics to scripted opponents, real deck archetypes, historical self-play and eventually a small league of policies.

A random neural network playing complete Clash Royale matches receives very weak feedback because most early games are long sequences of bad decisions. Shorter, targeted situations make the connection between **action and consequence** much clearer.

# Making training fast enough

Training speed became another major problem because the simulator eventually became faster than neural-network inference and had to wait for the policy. I worked on batching, GPU inference, CPU threading, entity processing and parallel rollout environments.

One high-throughput trainer reached roughly **590 decisions per second** using many parallel lanes. The current strength-tested configuration is slower because it uses safer memory handling and more checks, but it still reaches around **350 decisions per second** on my GPU setup.

This matters because reinforcement learning may eventually require **hundreds of millions of decisions**. At 10 decisions per second, that would be extremely slow; at hundreds or thousands per second, experiments become much more practical.

# My first major training problem: PPO made the bot worse

One of the most useful failures happened when I continued training an already working policy with PPO. Instead of improving, the policy became worse, and a checkpoint that handled several fixed evaluation situations eventually dropped to **zero successful matches**.

Simply changing the learning rate or trying random PPO parameters would not explain why, so I added more detailed diagnostics. For identical game states, I can now compare the good and bad checkpoint actions, alternative actions, card and placement probabilities, advantages, PPO ratios, entropy, KL divergence, gradient sizes and resulting tower damage.

This produced much more useful evidence. In one concrete comparison, the older actor selected:

```text
Skeletons at (3, 22)
```

while the changed actor selected:

```text
Hog Rider at (3, 17)
```

Across the comparison states, dozens of decisions changed, and those changes were followed by much more damage to my own towers. Interestingly, the overall PPO KL divergence was still small, showing that a global metric could hide a large behavioral change inside one part of the policy.

Looking at **actual Clash Royale decisions** was much more useful than looking only at average loss curves.

# Fixing the card-selection problem

The next issue was inside the card-selection architecture. The recurrent network already contained old card preferences, and although I added current public hand information, it initially acted only as a correction to those existing logits. The new information could therefore be useful without actually controlling the final decision.

I changed the architecture so that the **current public card context can become the primary card-selection path**. This significantly improved the provisional actor.

On one exact six-match evaluation, it achieved:

**4 wins and 2 losses**

In a larger paired evaluation with shuffled starting hands:

**current actor: 16 / 24 wins**

**previous baseline: 2 / 24 wins**

This does not mean the bot is already strong, but it does suggest that the policy is becoming more dependent on the **current game state** instead of stale card preferences.

# The newest change: real basic-mechanics scenarios

The newest curriculum improvement changes how the early training situations are generated. Previously, the curriculum contained categories such as:

```text
25% offense
25% ground defense
20% air defense
15% spells
15% kiting, cycling and elixir
```

However, these were mostly labels attached to ordinary matches, while the arena still started empty. That was not really a basic-mechanics curriculum.

The simulator can now generate **short, targeted starting states**. An air-defense scenario can begin with an air troop already approaching, a ground-defense scenario can begin with a push already in progress, a spell scenario can begin with several potential spell targets, and an offense scenario can begin in a position where tower pressure matters.

The simulator randomizes the lane, hand order, elixir, tower HP, threat timing and placement. Most importantly, the environment does **not tell the actor what card to play**. It creates the problem, and the neural network chooses the solution. Rewards depend on the resulting state, such as eliminating a threat or damaging a tower.

Instead of programming:

```text
air threat -> Musketeer
```

I can generate many different air-threat situations and allow the AI to discover which responses work.

# Where the project is now

CR-Bot is now very different from the version I shipped for Flavortown.

The project has three main systems:

```text
real Clash Royale
        |
        v
vision / public observation

simulator
        |
        v
fast training environment

policy
        |
        v
learned decisions
```

The simulator is still provisional and needs more comparison against the real game.

The policy is also still at an early stage. Tens of thousands of decisions are tiny compared with the millions or hundreds of millions that the full training plan may eventually require.

However, the complete real-game path now works.

A phone streams Clash Royale gameplay into the visual extractor. The extractor converts the screen into the public observation expected by the neural network. The policy chooses a card and placement, and the command is executed on the phone using **ADB**.

```text
phone screen
   |
   v
live stream
   |
   v
visual extractor
   |
   v
public observation
   |
   v
neural network
   |
   v
card + placement
   |
   v
ADB
   |
   v
real Clash Royale
```

It is not yet a reliable or competitive bot.

The vision system can still make mistakes, action timing needs more work, and the neural network has only started learning useful behavior.

But for the first time, the whole system works outside the simulator.

The next goal is to make the **basic-mechanics curriculum** reliably teach offense, defense, spell use, cycling, timing and placement. After that, I can gradually expose the policy to more complete matches, more decks and eventually self-play.

The biggest lesson from the past few months is that the hardest part is not choosing a neural-network architecture or starting a training run.

The difficult part is building an environment where the network receives the **right information**, gets a **useful learning signal**, cannot exploit simulator mistakes, and can eventually transfer what it learns back into the real game.
# What's next?

My long-term goal is not only to create a bot that can play Clash Royale. If the policy eventually becomes **significantly stronger than ordinary players**, I want to turn it into a tool that helps people analyse and improve their own gameplay.

The idea would be similar to **Stockfish for chess**.

A user could upload an **MP4 recording of a Clash Royale match**. The vision system would reconstruct the game state throughout the replay and detect what the player actually did. At every important moment, the trained policy could then calculate what **it would have played instead**.

For example:

```text
You played:
Hog Rider at (4, 17)

AI recommendation:
Wait

Alternative:
Musketeer at (12, 23)
