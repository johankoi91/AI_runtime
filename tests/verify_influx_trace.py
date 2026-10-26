import argparse
import http.server
import os
import re
import subprocess
import threading


class TraceServer(http.server.BaseHTTPRequestHandler):
    requests = []
    failures = 1

    def do_POST(self):
        payload = self.rfile.read(int(self.headers["Content-Length"])).decode()
        self.requests.append((self.path, payload, self.headers.get("Authorization")))
        status = 503 if self.failures else 204
        type(self).failures = max(0, self.failures - 1)
        self.send_response(status)
        self.end_headers()

    def log_message(self, *_):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--probe", required=True)
args = parser.parse_args()
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TraceServer)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
env = dict(os.environ, INFLUXDB_URL=f"http://127.0.0.1:{server.server_port}",
           INFLUXDB_ORG="trace-test", INFLUXDB_BUCKET="trace-test",
           INFLUXDB_TOKEN="test-token", RUNTIME_RUN_ID="hush-trace-test")
try:
    subprocess.run([args.probe], env=env, check=True, timeout=20)
    assert len(TraceServer.requests) >= 2
    assert TraceServer.requests[0][1] == TraceServer.requests[1][1]
    assert all(len(payload.encode()) <= 64 * 1024 for _, payload, _ in TraceServer.requests)
    lines = "".join(payload for _, payload, _ in TraceServer.requests[1:]).splitlines()
    assert len(lines) == 400
    sequences = [int(re.search(r"sequence=(\d+)i", line)[1]) for line in lines]
    assert sequences == list(range(1, 401))
    assert all('quote=\\"value\\" end' in line for line in lines)
    assert all("org=trace-test" in path and auth == "Token test-token"
               for path, _, auth in TraceServer.requests)
    TraceServer.failures = 100
    subprocess.run([args.probe, "bounded"], env=env, check=True, timeout=10)
finally:
    server.shutdown()
    server.server_close()
print("Influx trace: concurrent ordering, retry, escaping, flush, bounded queue passed")
