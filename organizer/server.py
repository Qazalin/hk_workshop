"""Workshop scoreboard and serial Docker-backed evaluation server."""
import argparse, concurrent.futures, hmac, json, math, os, pathlib, subprocess, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 64 * 1024

def request(url, token, payload=None):
  data = None if payload is None else json.dumps(payload).encode()
  req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=30) as response: return json.load(response)

def evaluate(path, challenge, repeats=11):
  import runpy, statistics
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.helpers import Context, GlobalCounters

  references = {
    "1": lambda x, a, b: x @ (a @ b),
    "2": lambda x: x.softmax(-1),
    "3": lambda x, w: (x.cast(dtypes.int32) * w.cast(dtypes.int32)).sum(-1),
  }

  def make_case():
    rng = np.random.default_rng()

    def rand(*shape): return rng.standard_normal(shape).astype(np.float32)

    if challenge == "1":
      arrays = [rand(64, 1024), rand(1024, 16), rand(16, 1024)]
      x, a, b = [v.astype(np.float64) for v in arrays]
      return arrays, x @ (a @ b), dtypes.float32
    if challenge == "2":
      x = rand(32, 4096) * 10
      exp = np.exp(x.astype(np.float64) - x.max(axis=-1, keepdims=True))
      return [x], exp / exp.sum(axis=-1, keepdims=True), dtypes.float32
    x, w = [rng.integers(-128, 128, size=(256, 4096), dtype=np.int8) for _ in range(2)]
    return [x, w], (x.astype(np.int32) * w.astype(np.int32)).sum(-1, dtype=np.int32), dtypes.int32

  def run(solve, arrays, expected, dtype):
    inputs = [Tensor(a.copy(), device="CPU").realize() for a in arrays]
    GlobalCounters.reset()
    start = time.perf_counter()
    result = solve(*inputs).realize()
    Device["CPU"].synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    assert result.shape == expected.shape and result.dtype == dtype, "wrong output shape or dtype"
    if challenge == "3": np.testing.assert_array_equal(result.numpy(), expected)
    else: np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3, atol=1e-6 if challenge == "2" else 2e-3, equal_nan=False)
    for tensor, original in zip(inputs, arrays): np.testing.assert_array_equal(tensor.numpy(), original)
    return elapsed

  with Context(DEBUG=0, TRACK_MATCH_STATS=0, PROFILE=0):
    solve, case = runpy.run_path(str(path))["solve"], make_case()
    times = []
    for fn in (references[challenge], solve):
      run(fn, *case)
      times.append(statistics.median(run(fn, *case) for _ in range(repeats)))
  baseline, submission = times
  return {"baseline_ms": baseline, "submission_ms": submission, "speedup": baseline / submission, "repeats": repeats}

class Evaluator:
  def __init__(self, directory, image, timeout, cpuset):
    self.directory, self.image, self.timeout, self.cpuset = pathlib.Path(directory).resolve(), image, timeout, cpuset
    self.directory.mkdir(parents=True, exist_ok=True)
    self.lock = threading.Lock()
    self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self.jobs = {}
    self.state_path = self.directory / "workshop.json"
    self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"rounds": []}
    for path in self.directory.glob("*/result.json"):
      job = json.loads(path.read_text())
      if job["status"] in ("queued", "running"):
        job.update(status="failed", error="evaluation server restarted; please resubmit")
      self.jobs[job["id"]] = job
      self.save(job)

  def save(self, job):
    path = self.directory / job["id"] / "result.json"
    path.with_suffix(".tmp").write_text(json.dumps(job))
    path.with_suffix(".tmp").replace(path)

  def save_state(self):
    self.state_path.with_suffix(".tmp").write_text(json.dumps(self.state))
    self.state_path.with_suffix(".tmp").replace(self.state_path)

  def start_round(self, challenge):
    if challenge not in ("1", "2", "3"): raise ValueError("unknown challenge")
    with self.lock:
      if any(j["status"] in ("queued", "running") for j in self.jobs.values()): raise ValueError("wait for evaluation to finish")
      for previous in self.state["rounds"]: previous["status"] = "closed"
      current = {"id": uuid.uuid4().hex, "challenge": challenge, "status": "active", "created": time.time()}
      self.state["rounds"].append(current)
      self.save_state()
      return dict(current)

  def submit(self, payload):
    if not isinstance(payload, dict): raise ValueError("expected a JSON object")
    name, source = payload.get("name"), payload.get("source")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64: raise ValueError("name must be 1–64 characters")
    if payload.get("challenge") not in ("1", "2", "3"): raise ValueError("unknown challenge (available: 1, 2, 3)")
    if not isinstance(source, str) or not source.strip(): raise ValueError("source is required")

    with self.lock:
      current = self.state["rounds"][-1] if self.state["rounds"] else None
      if not current or current["status"] != "active": raise ValueError("no active challenge; wait for the organizer")
      if payload["challenge"] != current["challenge"]: raise ValueError("this challenge is not active")
      if payload.get("round") != current["id"]: raise ValueError("round changed; use the current workshop submission command")
      if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= 64: raise ValueError("queue full; try again later")
      job = {"id": uuid.uuid4().hex, "name": name.strip(), "challenge": payload["challenge"],
             "status": "queued", "round": current["id"], "created": time.time()}
      folder = self.directory / job["id"]
      folder.mkdir(mode=0o755)
      source_path = folder / "solution.py"
      source_path.write_text(source)
      source_path.chmod(0o644)
      self.jobs[job["id"]] = job
      self.save(job)
      response = dict(job)
    self.pool.submit(self.run, job["id"])
    return response

  def update(self, job_id, **fields):
    with self.lock:
      if fields.get("status") in ("passed", "failed"): fields["finished"] = time.time()
      self.jobs[job_id].update(fields)
      self.save(self.jobs[job_id])

  def run(self, job_id):
    self.update(job_id, status="running", started=time.time())
    container = f"workshop-{job_id}"
    source = self.directory / job_id / "solution.py"
    command = ["docker", "run", "--rm", "--name", container, "--network=none", "--read-only", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=2g", "--cpus=1", "--log-driver=none",
               "--tmpfs=/tmp:rw,nosuid,size=256m", "--mount", f"type=bind,src={source},dst=/submission/solution.py,readonly"]
    if self.cpuset: command += ["--cpuset-cpus", self.cpuset]
    command += ["-e", "CHALLENGE=" + self.jobs[job_id]["challenge"], self.image]

    try:
      # Files keep runaway submission output out of the server's memory.
      with (source.parent / "stdout.log").open("wb") as stdout, (source.parent / "stderr.log").open("wb") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=self.timeout)
      if process.returncode:
        raise ValueError("evaluation failed; check correctness locally (organizer can inspect stderr.log)")
      with (source.parent / "stdout.log").open("rb") as output:
        output.seek(0, 2)
        output.seek(max(0, output.tell() - 4096))
        result = json.loads(output.read().splitlines()[-1])
      for key in ("baseline_ms", "submission_ms", "speedup"):
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or result[key] <= 0:
          raise ValueError("invalid benchmark result")
      self.update(job_id, status="passed", result=result)
    except subprocess.TimeoutExpired:
      self.update(job_id, status="failed", error=f"evaluation exceeded {self.timeout}s")
    except Exception as error:
      self.update(job_id, status="failed", error=str(error))
    finally:
      subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)

def serve(args):
  if not os.environ.get("WORKSHOP_ADMIN_TOKEN"): raise ValueError("set WORKSHOP_ADMIN_TOKEN before starting the server")
  subprocess.run(["docker", "image", "inspect", args.image], check=True, stdout=subprocess.DEVNULL)
  evaluator = Evaluator(args.data, args.image, args.timeout, args.cpuset)
  architecture = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Architecture}}", args.image], text=True).strip().upper()

  class Handler(BaseHTTPRequestHandler):
    def setup(self):
      super().setup()
      self.connection.settimeout(15)

    def reply(self, status, payload):
      body = json.dumps(payload).encode()
      self.send_response(status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def authorized_admin(self):
      expected = os.environ.get("WORKSHOP_ADMIN_TOKEN", "")
      if expected and hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {expected}".encode()): return True
      self.reply(401, {"error": "invalid WORKSHOP_ADMIN_TOKEN"})
      return False

    def do_POST(self):
      if self.path not in ("/submissions", "/round"): return self.reply(404, {"error": "not found"})
      if self.path == "/round" and not self.authorized_admin(): return
      try:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= MAX_BODY: raise ValueError("request must be at most 64 KiB")
        payload = json.loads(self.rfile.read(size))
        if not isinstance(payload, dict): raise ValueError("expected JSON object")

        if self.path == "/round": result = evaluator.start_round(payload.get("challenge"))
        else: result = evaluator.submit(payload)
      except (ValueError, UnicodeError) as error:
        return self.reply(400, {"error": str(error)})
      self.reply(202, result)

    def do_GET(self):
      if self.path == "/":
        body = pathlib.Path(__file__).with_name("index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return self.wfile.write(body)
      if self.path == "/dashboard":
        with evaluator.lock:
          payload = {"jobs": [dict(job) for job in evaluator.jobs.values()],
                     "round": dict(evaluator.state["rounds"][-1]) if evaluator.state["rounds"] else None,
                     "rounds": [dict(r) for r in evaluator.state["rounds"]],
                     "architecture": architecture}
        return self.reply(200, payload)

      with evaluator.lock:
        if self.path.startswith("/submissions/") and self.path.removeprefix("/submissions/") in evaluator.jobs:
          payload = dict(evaluator.jobs[self.path.removeprefix("/submissions/")])
        else: return self.reply(404, {"error": "not found"})
      self.reply(200, payload)

  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"Evaluation server listening on {args.host}:{args.port}", flush=True)
  try: server.serve_forever()
  except KeyboardInterrupt: pass
  finally:
    server.server_close()
    evaluator.pool.shutdown(wait=True)

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)

  server = commands.add_parser("serve")
  server.add_argument("--host", default="127.0.0.1")
  server.add_argument("--port", type=int, default=3000)
  server.add_argument("--image", default="hk-workshop-eval")
  server.add_argument("--data", default=str(pathlib.Path(__file__).parent / ".runtime/submissions"))
  server.add_argument("--timeout", type=int, default=180)
  server.add_argument("--cpuset", help="dedicated host CPU ID for official timings")

  start = commands.add_parser("start", help="open a challenge for participants")
  start.add_argument("challenge", choices=["1", "2", "3"])
  start.add_argument("--server", default=os.environ.get("WORKSHOP_SERVER", "http://localhost:3000"))

  evaluation = commands.add_parser("eval")
  evaluation.add_argument("file")
  evaluation.add_argument("--challenge", choices=["1", "2", "3"], default=os.environ.get("CHALLENGE", "1"))
  args = parser.parse_args()

  if args.command == "eval":
    print(json.dumps(evaluate(args.file, args.challenge)))
    return
  if args.command == "start":
    current = request(args.server.rstrip("/") + "/round", os.environ.get("WORKSHOP_ADMIN_TOKEN", ""), {"challenge": args.challenge})
    print(f"Challenge {current['challenge']} is open.")
    return
  return serve(args)

if __name__ == "__main__": main()
