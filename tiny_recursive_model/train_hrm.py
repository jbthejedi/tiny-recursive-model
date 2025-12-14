import torch
import torch.nn.functional as F
import torchvision

def main():
  print("yes")
  torch.nn.functional.binary_cross_entropy()


def latent_recursion(x, y, z, n=6):
  for i in range(n):
    z = net(x, y, z)

def deep_recursion(x, y, z, n=6, T=3):
  with torch.no_grad():
    for j in range(T-1):
      y, z = latent_recursion(x, y, z, n)
  y, z = latent_recursion(x, y, z, n)
  return (y.detach(), z.detach()), output_head(y), Q_head(y)

# Deep supervision
for x_input, y_true in train_dl:
  y, z = y_init, z_init
  for step in range(N_supervision):
    x = input_embmedding(x_input)
    (y, z), y_hat, q_hat = deep_recursion(x, y, z)
    loss = softmax_cross_entropy(y_hat, y_true)
    loss += binary_cross_entropy(q_hat, (y_hat == y_true))
    loss.backward()
    opt.step()
    opt.zero_grad()
    if q_hat > 0:
      break

if __name__ == '__main__':
  main()
