"""tools/loop_bench.py: detector / statistics self-test and an end-to-end run against the mock backend (loop vs no loop)."""
import json, os, socket, subprocess, sys, tempfile, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QV_SERVER = os.environ.get("QV_SERVER", os.path.join(ROOT, "build-cpu", "qv_server"))
TOK = os.environ.get("QV_INKLING_TOKENIZER", "/workspace/q-volta-models/Inkling-2-Mini/tokenizer.json")
BENCH = os.path.join(ROOT, "tools", "loop_bench.py")


def test_selftest():
    assert subprocess.run([sys.executable, "-I", BENCH, "selftest"], capture_output=True, text=True).stdout.strip() == "selftest ok"


def _run(extra):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = subprocess.Popen([QV_SERVER, "--backend", "mock", "--tokenizer", TOK, "--port", str(port), "-c", "8192"] + extra,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1); break
            except Exception:
                time.sleep(0.1)
        out = tempfile.mktemp(suffix=".jsonl")
        subprocess.run([sys.executable, "-I", BENCH, "run", "--url", "http://127.0.0.1:%d" % port, "--tokenizer", TOK, "--cells", "short_chat",
                        "--n", "2", "--max-tokens", "400", "--out", out, "--label", "t"], check=True, capture_output=True)
        return [json.loads(l) for l in open(out)]
    finally:
        srv.terminate(); srv.wait()


def test_mock_loop_detected():
    rows = _run(["--mock-text", "The quick brown fox jumps over the lazy dog while the cat watches from the old stone wall. ", "--mock-loop"])
    assert rows and all(r["kind"] == "loop" for r in rows), rows


def test_mock_no_loop():
    rows = _run(["--mock-text", "A short answer that ends."])
    assert rows and all(r["kind"] == "eos" for r in rows), rows
