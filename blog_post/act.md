# ACT

What is this? HRM wants, at training time, to learn to predict when to halt deep supervision based on a latent $z_H$. To do this, the model has a `q_head` that outputs two logits, $q_halt$ and $q_continue$. During inference, the model runs the full $N_{sup}$ deep supervision steps.

### The training
If you look at the deep supervision portion of the training loop, without ACT, each minibatch `(x_input, y_true)` must run `N_sup` number of times. With ACT, you may terminate early when $q_{halt} > q_{continue}$

- ACT reduces the average number of supervision steps per sample (good), **but**
- the continue objective requires an **extra forward pass**, so ACT ends up using **two forward passes per optimization step**

To reduce the number of supervision steps, the authors of HRM use Adaptive computational time (ACT). ACT is an algorithm that uses a Q-learning objective to determine if the deep supervisin should halt on the current iteraton or continue. ACT often terminates the deep supervision in less than 2 steps, greatly improving training time.

### How it works

#### act_halt
```python
def act_halt(halt, y_hat, y_true):
    """halt = q[0]"""
    return 0.5 * bce_logit(halt, y_hat == y_true)
    # If the answer is correct: target = 1 ⇒ push q[0] (halt) up.
    # If the answer is wrong: target = 0 ⇒ push q[0] down.
```

`act_halt(q, y_pred, y_hat)` takes in the `q` output head after the first `hrm` call, and essentially teaches the controller to halt by raising the halting prediction toward 1 if the prediction matches the target, and lowering the halting prediction toward 0 if the prediction does not match the target.


#### act_continue

```python
def act_continue(q_next, is_last_step):
    # BCE = -(tlog(p) + (1 - t)log(1 - p))
    halt_next, cont_next = q_next[0], q_next[1]
    if is_last_step:
        # if halt is small, push continue to be smaller. If halt
        # is big, push continue to be bigger.
        return 0.5 * bce_logit(pred=cont_next, label=halt_next) 
    else:
        return 0.5 * bce_logit(pred=cont_next, label=max(halt_next, cont_next))
```

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

### Finally
Q_head outputs two logits:
1. q[0] determines *halt*
2. q[1] determines *continue*

If q[0] > q[1] deep supervision stops

Q_head is trained jointly with hrm, which means it's not optimal at the beginning of trainig. If you look at Figure 2, you can see how `ACT_halt` and `ACT_continue` both generate loss to penalize false confidence in q[0] or q[1]