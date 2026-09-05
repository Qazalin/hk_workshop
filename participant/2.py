# UOp challenge: implement stable softmax as a custom fused kernel.
# Input/output: float32 [32, 4096].

def solve(x):
  return x.softmax(-1)
