import argparse
import http.server
import os
import re
import subprocess
import threading
import time


class TraceServer(http.server.BaseHTTPRequestHandler):
    requests = []
    failures = 1
    stall = False
    entered = threading.Event()
    release = threading.Event()

    def do_POST(self):
        payload = self.rfile.read(int(self.headers["Content-Length"])).decode()
        self.requests.append((self.path, payload, self.headers.get("Authorization")))
        if self.stall:
            self.entered.set()
            self.release.wait(timeout=10)
        status = 503 if self.failures else 204
        type(self).failures = max(0, self.failures - 1)
        self.send_response(status)
        self.end_headers()

    def log_message(self, *_):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--probe", required=True)
parser.add_argument("--bundle")
args = parser.parse_args()
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TraceServer)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
env = dict(os.environ, INFLUXDB_URL=f"http://127.0.0.1:{server.server_port}",
           INFLUXDB_ORG="trace-test", INFLUXDB_BUCKET="trace-test",
           INFLUXDB_TOKEN="test-token", RUNTIME_RUN_ID="hush-trace-test")
try:
    result = subprocess.run([args.probe], env=env, check=True, timeout=20,
                            capture_output=True, text=True)
    print(result.stdout, end="")
    accepted = int(re.search(r"accepted=(\d+)", result.stdout)[1])
    assert len(TraceServer.requests) >= 2
    assert TraceServer.requests[0][1] == TraceServer.requests[1][1]
    assert all(len(payload.encode()) <= 64 * 1024 for _, payload, _ in TraceServer.requests)
    lines = "".join(payload for _, payload, _ in TraceServer.requests[1:]).splitlines()
    assert len(lines) == accepted
    sequences = [int(re.search(r"sequence=(\d+)i", line)[1]) for line in lines]
    assert sequences == list(range(1, accepted + 1))
    assert all('quote=\\"value\\" end' in line for line in lines)
    assert all("org=trace-test" in path and auth == "Token test-token"
               for path, _, auth in TraceServer.requests)
    TraceServer.failures = 100
    subprocess.run([args.probe, "bounded"], env=env, check=True, timeout=10)
    TraceServer.failures = 0
    TraceServer.stall = True
    if args.bundle:
        env["HUSH_MODEL_BUNDLE"] = args.bundle
    process = subprocess.Popen([args.probe, "stalled"], env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "READY"
    assert TraceServer.entered.wait(timeout=5), "upload request did not reach the stalled server"
    start = time.monotonic()
    output, error = process.communicate(input="\n", timeout=2)
    assert process.returncode == 0, (output, error)
    assert time.monotonic() - start < 2
    assert not TraceServer.release.is_set()
    if args.bundle:
        assert "hush_frames=32" in output
    print("Stalled HTTP server:", output.strip())
finally:
    TraceServer.release.set()
    server.shutdown()
    server.server_close()
print("Influx trace: concurrent ordering, retry, escaping, flush, bounded queue passed")
