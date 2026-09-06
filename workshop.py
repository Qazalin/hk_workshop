"""Check and submit workshop solutions."""
import argparse, json, os, pathlib, runpy, time, urllib.request

WORKSHOP_SERVER = "https://hk-workshop.vercel.app"

def local_check(path, challenge):
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.helpers import Context, DEBUG, PROFILE, GlobalCounters
  from tinygrad.uop.ops import TRACK_MATCH_STATS

  def rand(rng, *shape): return rng.standard_normal(shape).astype(np.float32)

  with Context(DEBUG=0, TRACK_MATCH_STATS=0, PROFILE=0):
    rng = np.random.default_rng()
    solve = runpy.run_path(str(path))["solve"]
    if challenge == "1":
      arrays = [rand(rng, 64, 1024), rand(rng, 1024, 16), rand(rng, 16, 1024)]
      x, a, b = [v.astype(np.float64) for v in arrays]
      expected, dtype = x @ (a @ b), dtypes.float32
    elif challenge == "2":
      x = rand(rng, 32, 4096) * 10
      exp = np.exp(x.astype(np.float64) - x.max(axis=-1, keepdims=True))
      arrays, expected, dtype = [x], exp / exp.sum(axis=-1, keepdims=True), dtypes.float32
    elif challenge == "3":
      x, w = [rng.integers(-128, 128, size=(256, 4096), dtype=np.int8) for _ in range(2)]
      arrays, expected, dtype = [x, w], (x.astype(np.int32) * w.astype(np.int32)).sum(-1, dtype=np.int32), dtypes.int32
    else: raise ValueError(f"unknown challenge: {challenge}")
    inputs = [Tensor(a.copy(), device="CPU").realize() for a in arrays]

  GlobalCounters.reset()
  with Context(DEBUG=DEBUG.value, TRACK_MATCH_STATS=TRACK_MATCH_STATS.value, PROFILE=PROFILE.value):
    start = time.perf_counter()
    result = solve(*inputs).realize()
    Device["CPU"].synchronize()
    elapsed = (time.perf_counter() - start) * 1000

  assert result.shape == expected.shape and result.dtype == dtype, "wrong output shape or dtype"
  if challenge == "3": np.testing.assert_array_equal(result.numpy(), expected)
  else: np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3, atol=1e-6 if challenge == "2" else 2e-3, equal_nan=False)
  for tensor, original in zip(inputs, arrays): np.testing.assert_array_equal(tensor.numpy(), original)
  print(f"PASS  {elapsed:.3f} ms")

def request(url, payload=None):
  data = None if payload is None else json.dumps(payload).encode()
  req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=30) as response: return json.load(response)

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)

  check_command = commands.add_parser("check")
  submit_command = commands.add_parser("submit")
  for command in (check_command, submit_command):
    command.add_argument("file")
    command.add_argument("--challenge", choices=["1", "2", "3"])

  submit_command.add_argument("--name", required=True)
  args = parser.parse_args()

  challenge = args.challenge or pathlib.Path(args.file).stem
  if args.command == "check": return local_check(args.file, challenge)

  url = WORKSHOP_SERVER
  current = request(url + "/dashboard").get("round")
  if not current: raise ValueError("no active challenge; wait for the organizer")
  job = request(url + "/submissions", {"name": args.name, "challenge": challenge,
                "round": os.environ.get("WORKSHOP_ROUND", current["id"]), "source": pathlib.Path(args.file).read_text()})
  print(f"Submitted {job['id']}", flush=True)

  while job["status"] in ("queued", "running"):
    time.sleep(1)
    job = request(url + "/submissions/" + job["id"])
  if job["status"] != "passed": raise SystemExit("FAIL  " + job.get("error", "evaluation failed"))
  result = job["result"]
  print(f"PASS  {result['submission_ms']:.3f} ms  ({result['speedup']:.2f}x baseline)")

if __name__ == "__main__": main()
