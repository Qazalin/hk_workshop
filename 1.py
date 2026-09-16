# input shapes: x (64, 1024), a (1024, 16), b (16, 1024)

def solve(x, a, b):
  return x @ (a @ b)
