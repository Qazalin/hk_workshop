# Program challenge: optimize signed int8 row-wise dot products using a custom CPU program.
# Inputs: int8 [256, 4096]. Output: int32 [256]. Use instructions supported by your evaluation CPU.
from tinygrad import dtypes

def solve(x, w):
  return (x.cast(dtypes.int32) * w.cast(dtypes.int32)).sum(-1)
