# Hierarchical Reasoning Model (HRM) for builders: recursion, deep supervision, and why ACT is annoying

[Original Paper](https://arxiv.org/abs/2506.21734)

This is my understanding of the concept of HRM, distilled while reading the TRM paper (and cross-checking HRM pseudocode in the paper). I got so annoyed by how challenging it was to understand ACT, that I basically did a whole deep-dive on HRM because of it.

## Summary
- Recursive refinement + deep supervision is the real juice
- HRM’s gradient story is DEQ-ish / 1-step-ish and arguably shaky
- ACT is conceptually dense and easy to misread; here’s my best mechanical interpretation.

#### Figure 1: HRM pseudocode

```python
def act_halt(halt, y_hat, y_true):
    """halt = q[0]"""
    return 0.5 * bce_logit(halt, y_hat == y_true)
    # If the answer is correct: target = 1 ⇒ push q[0] (halt) up.
    # If the answer is wrong: target = 0 ⇒ push q[0] down.

def act_continue(q_next, is_last_step):
    # BCE = -(tlog(p) + (1 - t)log(1 - p))
    halt_next, cont_next = q_next[0], q_next[1]
    if is_last_step:
        # if halt is small, push continue to be smaller. If halt
        # is big, push continue to be bigger.
        return 0.5 * bce_logit(pred=cont_next, label=halt_next) 
    else:
        return 0.5 * bce_logit(pred=cont_next, label=max(halt_next, cont_next))

for x_input, y_true in train_dl:
    z = z_init
    for step in range(N_sup):  # deep supervision
        x = input_embedding(x_input)
        z, y_pred, q = hrm(z, x)
        loss = softmax_cross_entropy(y_pred, y_true)
        # ACT stuff...
        loss += ACT_halt(q, y_pred, y_true)
        _, _, q_next = hrm(z, x)  # extra forward pass
        loss += ACT_continue(q_next, step == N_sup - 1)

        z = z.detach() # z now contains the value of what 'z' was, but it's
                       # detached from the computation graph
        loss.backward()
        opt.step()
        opt.zero_grad()
        if q[0] > q[1]:  # early-stopping
            break
```

## Recursive hierarchical reasoning
Happens inside the `hrm` function of *Figure 1*

The "hierarchy" means high and low frequencies
$f_L$ at high frequency network that produces latent feature $z_L$
$f_H$ at low frequency network that produces latent feature $z_H$
Both networks are 4 layer transformers.
high frequency = called more times in the iteration
low frequency = called less times in the iteration

"Recursion" just means iterating over the latents, passing them through the same network repeatedly. The network produces two latent features $z_L$ that are used as inputs on the next go-around


## Deep supervision
Deep supervision is performing supervised learning on a training sample $N_{sup}$ times. It's the `for step in range(N_sup)` part.

Use $z = z.detach()$ to detach gradients so you don't backprop through the previous step. z is then passed into 


#### Figure 2: HRM forward pass (paper pseudocode, reconstructed)
$$
\begin{aligned}
x      &\gets f_I(\tilde{x}) \\
z_L    &\gets f_L(z_L + z_H + x) &&\text{(without gradients)}\\
z_L    &\gets f_L(z_L + z_H + x) &&\text{(without gradients)}\\
z_H    &\gets f_H(z_L + z_H)     &&\text{(without gradients)}\\
z_L    &\gets f_L(z_L + z_H + x) &&\text{(without gradients)}\\
z_L    &\gets z_L.\mathrm{detach()} \\
z_H    &\gets z_H.\mathrm{detach()} \\
z_L    &\gets f_L(z_L + z_H + x) &&\text{(with gradients)}\\
z_H    &\gets f_H(z_L + z_H)     &&\text{(with gradients)}\\
\hat{y}&\gets \arg\max f_O(z_H)
\end{aligned}
$$

Deep supervision seems to yield most of the performance gains in HRMs.
Authors motivation for wanting to improve *recursive reasoning*.

## Deep Equilibrium (DEQ) and 1-step-gradient approximation

All you need to know here is the following. **DEQ** is loosely the idea of iterating repeatedly until a result reaches the optimal point, called an *equilibrium*, with the proper checks that the proper point has been reached.

## 1-step gradient approximation

**1-step-gradient approximation** says that only the learnings from the equilibrium point are need to optimize your network, you can discard the rest. HRM is motivated by DEQ/1-step-gradient approximation to justify their architecture.

#### Details
By the Implicit Function Theorem [Krantz & Parks, 2002] with the 1-step gradient approximation, backpropagation uses only the gradients from the last two steps, $f_L$ and $f_H$, in `hrm` of *Figure 1*.
IFT says that if a recurrent function converges to a fixed point, we can apply a single backpropagation step at that equilibrium point.

A **fixed point** of a function $f$ is a point $z^*$ such that
$$ z^* = f(z^*) $$

**Fixed-point iteration** is the process
$$ z_{k+1} = f(z_k) $$

where you keep updating $z_k$ until it *converges*, meaning the residual $|f(z_k)-z_k|$ is very small.

The HRM authors assume fixed points
$$
\begin{aligned}
    z^*_L \approx f_L(z^*_L + z_H + x) \\
    z^*_H \approx f_H(z_L + z_H^*)
\end{aligned}
$$

Basically, for the hyperparams used in HRM ($n = 2$, $T = 2$), 4 recursions are called, then detach is called, and the two lines below are called
$$
\begin{aligned}
z_L    &\gets f_L(z_L + z_H + x) &&\text{(with gradients)}\\
z_H    &\gets f_H(z_L + z_H)     &&\text{(with gradients)}\\
\end{aligned}
$$
and assumes equilibrium is reached. If HRM actually ran a proper solver until the residuals were tiny, then applied the IFT/1-step gradient at $(z_L^*, z_H^*)$, then the theory would line up with DEQ-style models.

## ACT
To me, this is the most mechanically complicated part of HRM by far.

What is this? HRM wants, at training time, to learn to predict when to halt deep supervision based on a latent $z_H$. To do this, the model has a `q_head` that outputs two logits, $q_halt$ and $q_continue$. During inference, the model runs the full $N_{sup}$ deep supervision steps.

### How ACT works during training
If you look at the deep supervision portion of the training loop, without ACT, each minibatch `(x_input, y_true)` must run `N_sup` number of times. With ACT, you may terminate early when $q_{halt} > q_{continue}$

- ACT reduces the average number of supervision steps per sample (good), **but**
- the continue objective requires an **extra forward pass**, so ACT ends up using **two forward passes per optimization step**

To reduce the number of supervision steps, the authors of HRM use Adaptive computational time (ACT). ACT is an algorithm that uses a Q-learning objective to determine if the deep supervision should halt on the current iteration or continue. ACT often terminates the deep supervision in less than 2 steps, greatly improving training time.

`act_halt(q, y_pred, y_hat)` takes in the `q` output head after the first `hrm` call, and essentially teaches the controller to halt by raising the halting prediction toward 1 if the prediction matches the target, and lowering the halting prediction toward 0 if the prediction does not match the target.


`act_continue(q_next, is_last_step)` is called after the second call to `hrm`, and takes in `q_next`, which is the `Q_head` output from the second `hrm` call, and `is_last_step` which is a boolean that basically says if the current step is the *last step*. `q_next` is the controller’s estimate after one more refinement, used to train what “continue” should mean.

##### case 1: is_last_step == True

If `is_last_step` is true, meaning this is the last step `step == N_sup - 1`. We can see the line

```python
bce_logit(pred=cont_next, label=halt_next) 
```

Whatever number the network settles on for “halt here” (`halt_next`), we're teaching it to give the same number to “continue here”. So at the last state, the model learns to be indifferent between halt and continue: they’re numerically the same $cont\_next \approx halt\_next$.

*What does this mean?* We're basically using `halt_next` to help the model learn a bound for what what the value for *continue* should be at the last step. If we don't do this, the value for *continue* could be unreasonably high, then earlier steps (that do use the max) would inherit that inflated “continue looks great” signal and over-continue near the horizon.

##### case 2: else
```python
bce_logit(pred=cont_next, label=max(halt_next, cont_next))
```

For non-last steps, ACT_continue biases the controller so that continuing is always as good as the best available action at the next state
- Yes: it avoids the situation “halt looks like the best just because cont_next never learned to have a reasonable value.”
- No: it doesn’t force the policy to pick continue; it just refuses to let continue sit artificially low compared to what the model thinks is achievable at that next state.

*Quick summary*
Not last step: “Continue now → next step I can still decide → so continue should match ‘best next option’.”

Last step: “Continue now → next step I’m basically forced to stop → so continue should match ‘halt next’.”

### Deciding to halt

Q_head outputs two logits:
1. q[0] determines *halt*
2. q[1] determines *continue*

If q[0] > q[1] deep supervision stops

Q_head is trained jointly with hrm, which means it's not optimal at the beginning of training. If you look at Figure 1, you can see how `ACT_halt` and `ACT_continue` both generate loss to penalize false confidence in q[0] or q[1]

## Final Notes on HRM

I've summarized the main components of HRM. Don't get me wrong, I think HRM is an incredible achievement. To be able to create something like this is monumental even if some parts are a bit more complicated. Bringing a model like this into reality is an impressive feat. Hats off to the authors.
