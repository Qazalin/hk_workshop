from tinygrad import Tensor

def solve(a, b):
  return (Tensor(a) * Tensor(b)).sum()
  """
  out = 0.0
  for i in range(len(a)):
    out += a[i] * b[i]
  return out
  """
