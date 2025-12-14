# Hierarchical Reasoning Model (HRM) for builders: recursion, deep supervision, and why ACT is annoying

#### TLDR
##### If you remember 3 things
  1. TRM replaces “fixed-point theory + 1-step grad” with “unroll recursion (final pass) + no-grad refinement passes.”
  2. TRM replaces ACT/Q-learning halting with a simple BCE halting head (no extra forward pass).
  3. The whole model reduces to **two states**: current solution (y) + latent reasoning (z).

##### What to implement first
The core engine of TRM to grok is `latent_recursion` and `deep_recursion` (See Figure 3). Make sure to pay close attention to where `torch.no_grad()` and `.detach()` are called.

##### Where folks might mess up
Mixing up detach points (accidentally backprop through deep supervision), mis-implementing the no-grad passes, or treating the halting logit incorrectly (thresholding the wrong thing).

Main problem: LLMs struggle on hard question-answer problems like puzzles (Sudoku)
Approach: heirarchical reasoning models (hrm) [Wang et al]

The genius of trm really lands when it's juxtaposed against HRM. simple is better than complex, complex is better than complicated. Don't get me wrong, hrm is still an incredible invention, but it's a bit complicated. 

## Hierarchical Reasoning Model (HRM): Overview and issues

### Summary
- Recursive refinement + deep supervision is the real juice
- HRM’s gradient story is DEQ-ish / 1-step-ish and arguably shaky
- ACT adds complexity + extra forward pass

### Recursive Refinement + Deep supervision

This is where all the magic in HRM happens.

*Recursive refinement* is shown in Figure 2 and executes when the `hrm` function is called. "Recursion" just means iterating over the reasoning latents, zH and zL, passing them through the same network repeatedly. The gradients are only computed on the last two refinement steps of the recursion. The reason for the word 'hierarchical' in HRM come from zH and zL. In HRM, zL is the fast-updating / high-frequency latent (updated many times per cycle), while zH is slow-updating / low-frequency (updated less often). Lastly, `hrm` calls two transformer based networks, `L_net` and `H_net`, the high and low frequency networks, respectively. The TRM uses a simpler single-network based architecture.

*Deep supervision* is iterating over a single batch sample within the training loop $N_{sup}$ number of times, repeatedly supervising the prediction, updating the latent $z$, and passing the updated latent to `hrm` on the next go around. See Figure 2 in the appendix.

### Deep Equilibrium (DEQ) and 1-step-gradient approximation

All you need to know here is the following. **DEQ** is loosely the idea of iterating repeatedly until a result reaches the optimal point, called an *equilibrium*, with the proper checks that the proper point has been reached. **1-step-gradient approximation** says that only the learnings from the equilibrium point are need to optimize your network, you can discard the rest. HRM is motivated by DEQ/1-step-gradient approximation to justify their architecture.

#### Concern

The author of TRM says the way HRM is designed doesn't guarantee a fixed point is ever reached: there aren't enough recursions steps and there is no convergence verification-TRM removes this by backpropagating through the full unrolled recursion (for the final pass) while using no-grad passes for refinement.

### ACT adds complexity

ACT is what HRM uses to early-stop deep supervision. Deep supervision runs for $N_{sup} = 16$ number of steps. That's 16 additional supervision steps during training which include the recursive `hrm` call. ACT uses an additional output head of HRM called `Q_head` and the latent (see Figure 2 in appendix) to predict if the target is reached and deep supervising should halt

#### Concern

The author of TRM notes that ACT is implemented by running an additional forward pass (call to `hrm`) on each deep supervision step. That's two calls to `hrm` every iteration.

Also, personally, the implementation of ACT is hard to grok. Namely, `ACT_continue`, which TRM removes all together.

## Tiny Recursive Model (TRM)

#### Figure 1

```python
def latent_recursion(x, y, z, n=6):
  """
  y = zH
  z = zL
  """
  for i in range(n): # refine latent reasoning
    z = net(x, y, z)
  y = net(y, z) # refine answer
  return y, z

def deep_recursion(x, y, z, n=6, T=3):
  """
  y = zH
  z = zL
  """
  with torch.no_grad(): # No gradients
    for j in range(T-1):
      y, z = latent_recursion(x, y, z, n)
  y, z = latent_recursion(x, y, z, n) # gradients
  return (y.detach(), z.detach()), output_head(y), Q_head(y)

# Deep supervision
for x_input, y_true in train_dl:
  y, z = y_init, z_init
  for step in range(N_supervision):
    x = input_embedding(x_input)
    (y, z), y_hat, q_hat = deep_recursion(x, y, z)
    loss = softmax_cross_entropy(y_hat, y_true)
    loss += bce_with_logit(q_hat, (y_hat == y_true))

    loss.backward()
    opt.step()
    opt.zero_grad()

    # q_hat is a logit; q_hat > 0 ⇔ sigmoid(q_hat) > 0.5**
    if q_hat > 0:
      break
```

**Because $q$ is a logit, $q>0$ is exactly the $p>0.5$ decision boundary; other thresholds would just trade compute for confidence via $q>\log(\tau/(1-\tau))$.

Say we define a threshold $\tau$ such that $\sigma(q) > \tau$. And since $\sigma^{-1} = \text{logit}$, we get $$q > \log \frac{\tau}{1 - \tau}$$.

So we can change our confidence by varying $\tau$.

### Data

Just a quick note on this. The datasets are Sudoku-Extreme, Maze-Hard, and ARC-AGI I & II. Consider them all square puzzle grids of initial shape MxM. A batch looks like (B, L=MxM)
* Sudoku-Extreme: $L = MxM = 9x9$ $\to$ a batch looks like $(B, 81)$
* Maze-Hard: $L = 30x30$ $\to$ a batch looks like $(B, 900)$
* ARC-AGI: This one is trickier. The paper says a single training sample has *multiple examples*. Then it says they are *serialized* into a single token. So I'm inferring $L$ is the size of all the flattened examples concatenated together.

Then if $x$ is our input, `x = input_embedding(x)` (Figure 1) is called, turning each token into an embedding. Then the data becomes shape (B, L, D).

### Improvements

* No fixed point theorem
* ACT: No additional forward pass
* No heirarchical features + single network

#### No fixed point theorem

HRM only backpropogates through the last two function evaluations of $z_L \larr f_L(z_L + z_H + x)$ and $z_H \larr f_H(z_L + z_H)$, which the author of TRM points out is highly unlikely for zL and zH to reach a fixed point. Thus the *backpropogating at an equilibrium* is no longer justified.

TRM solves this by doing the following. 1) defining backpropagating over $n$ number of recursion steps as defined in `latent_recursion`, not just the last two steps. 2) Calling `latent_recursion` T-1 times, updating zL and zH each time with *no gradients*. Then calling `latent_recursion` one final time *with gradients*.

`latent_recursion` over $n$ steps removes the need for a 1-step gradient approximation, *and* calling `latent_recursion` $T-1$ times inside of `deep_recursion` allows zH and zL to be improved recusively without backpropagation.

So `latent_recursion` is a module you can deploy with/without gradients attached to iteratively refine an answer. Call it as many times as you want without gradients to refine your answer. Then as soon as you want to update weights, attach gradients.

#### ACT: no additional forward pass
TRM fixes the additional forward pass simply by only learning a halt probabiliity, thus removing the complicated next pass prediction and continue path.

#### No hierarchical features + single network
TRM reinterprets $z_H$ as $y$ and $z_L$ as $z$, removing the need to create to latent embeddings from separate networks ($f_L$ and $f_H$), and reducing two networks down to a single network.

#### Attention free architecture
For tasks where $L \leq D$, like Sudoku-Extreme, TRM uses an MLP-based architecture over a transformer and imporoves performance from 74.7% to 87.4%.

Also, just a quick implementation detail. An MLP-mixer (token mixer), which could capture global context across tokens would look something like this:

```python
x_t = x.transpose(1, 2)          # (B, D, L)  <-- tokens are now the last dim
x_t = nn.Linear(L, L)(x_t)      # mixes across tokens, independently per channel
x   = x_t.transpose(1, 2)        # back to (B, L, D)
```

TRM mentions that MLP works with Sudoku because $L = 9x9$, but produces suboptimal results on Maze-Hard because $L = 30x30 = 900$. This is for a couple of reasons. First, 900 is very large compared to the small dataset size. For an MLP-mixer, the number of parameters scales with the number of tokens $L$. This can hurt the model's ability to generalize. TRM uses self-attention to mix tokens for large $L$ because the number of paramters in self-attention scales with $D$ instead of $L$.

#### Remaining improvements
* Less layers: More MLP layers degrades performance on Sudoku-Extreme.
* EMA of weights: To mitigate the tendency of HRM to overfit on small datasets, TRM uses Exponential Moving Average (EMA) of the weights.

### OOM Issue

The author notes time complexity and memory issues of TRM
1. Increasing $T$ or $n$ leads to "massive" slowdowns because of the complexity of recursion nested in deep supervision
2. Increasing $n$ can lead to OOM errors because TRM is backpropagating through through the full recursion graph (See Figure 3).

## Results


What to know about the datasets:
* Sudoku-Extreme is extremely difficult Sudoku puzzles. 423K samples for testing
* Maze-Hard is 30x30 procedurally generated mazes. Train/test are both of size 1000
* ARC-AGI-1/ARC-AGI-2 are geometric puzzles
    * Each puzzle task contains 2-3 example solutions and 1-2 test puzzles to be solved
    * Easy for humans, hard for AI
    * ARC-AGI data also augmented with 160 tasks from ConceptARC, a closely related dataset
* Datasets are small so augmentation is used: Sudoku-Extreme, Maze-Hard, and ARC-AGI all use shuffling or data transformation per example 
    
Quick notes on the results:
* Small datasets → overfitting pressure → why EMA & data augmentation matters
* Maze/ARC need global context → why attention may dominate there
* Sudoku has structure that MLP can exploit → why attention might be unnecessary

### Improved
- Sota test accuracy on Sudoku-Extreme $55\% \to 87\%$ and Maze-Hard $75\% \to 85\%$
- ARC-AGI-1 from $40\% \to 45\%$
- ARC-AGI-2 from $5\% \to 8\%$

Should investigate these ^ metrics to understand the datasets
Good, this is exactly the right thing to zoom in on.

## My Approach

### “If I were implementing TRM this week”

5–8 bullets, like:

* Start with Figure 3 exactly: verify detach/no_grad boundaries before touching architecture.
* Treat halting head output as a **logit**; threshold at 0 (0.5 prob).
* Expect OOM as you increase `n`; consider gradient checkpointing or smaller batch.
* Expect diminishing returns for bigger MLP depth on Sudoku-Extreme; keep it shallow.
* Use EMA from day 1; small data will overfit fast without it.
* For Maze/ARC, don’t assume attention-free will hold; test attention early.

This takes 15 minutes and instantly separates you from “summary posts.”

### “Ablations I’d run first (predictions)”

Even without running them, list 3–5 ablations and what you expect:

* remove deep supervision (`N_sup=1`) → big drop
* vary `n` vs `T` → which saturates first?
* remove EMA → instability / worse generalization
* swap MLP ↔ attention across Sudoku vs Maze/ARC → attention helps on larger-context tasks




