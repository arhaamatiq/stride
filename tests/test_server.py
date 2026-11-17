#!/usr/bin/env python3
"""stride_server end to end on the tiny random model (CPU backend).

Builds tests/data/tiny (model + a small Llama-3-style tokenizer.json) if missing,
starts stride_server on an ephemeral port and drives it over HTTP. The same
prompts are also run through stride_cli and tokenize_cli, so outputs and
fingerprints are checked against the engine's own batch path. STRIDE_BUILD names
the build dir (default build-cpu).

  python tests/test_server.py
"""
import concurrent.futures
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

DATA = ROOT / "tests" / "data"
TINY = DATA / "tiny"
BUILD = Path(os.environ.get("STRIDE_BUILD", ROOT / "build-cpu"))
SERVER, CLI, TOKCLI = BUILD / "stride_server", BUILD / "stride_cli", BUILD / "tokenize_cli"
FP = re.compile(r"^stride-[0-9a-f]{16}$")

PROMPTS = [
    "The quick brown fox",
    "def fibonacci(n):\n    ",
    "café 日本語 \U0001f600 test",
    "1 2 3 4 5 6 7 8 9 10",
    "x",
    "    leading spaces and\ttabs\n\nand newlines",
    "A longer prompt " * 12,
    "<html><body><p>hello</p></body></html>",
    "Once upon a time, in a land far away, there lived a",
    "مرحبا بالعالم",
    "SELECT name, age FROM people WHERE age > 30 ORDER BY name;",
    "Q: What is 2+2?\nA:",
]


class Proc:
    """A stride_server process; stderr is collected for the verbose-log test."""

    def __init__(self, model, *flags):
        self.proc = subprocess.Popen(
            [str(SERVER), "--model", str(model), "--host", "127.0.0.1", "--port", "0", *flags],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.err = []
        self.reader = threading.Thread(target=lambda: [self.err.append(l) for l in self.proc.stderr], daemon=True)
        self.reader.start()
        line = self.proc.stdout.readline()
        m = re.search(r":(\d+) \(max_context (\d+), kv_blocks (\d+)\)", line)
        if not m:
            self.proc.kill()
            raise RuntimeError(f"server did not start: {line!r} {''.join(self.err)}")
        self.port, self.max_context, self.kv_blocks = (int(g) for g in m.groups())

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        try:
            code = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = None
        self.reader.join(timeout=5)
        self.proc.stdout.close()
        self.proc.stderr.close()
        return code

    def conn(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)

    def get(self, path):
        c = self.conn()
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, json.loads(body)

    def post(self, path, body, conn=None):
        c = conn or self.conn()
        c.request("POST", path, body if isinstance(body, str) else json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if conn is None:
            c.close()
        return r.status, json.loads(data)

    def stream(self, path, body, conn=None, stop_after=None):
        """Returns (status, headers, data payloads). stop_after closes early."""
        c = conn or self.conn()
        c.request("POST", path, json.dumps(dict(body, stream=True)), {"Content-Type": "application/json"})
        r = c.getresponse()
        events = []
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode("utf-8").rstrip("\n")
            if not line.startswith("data: "):
                continue
            events.append(line[6:])
            if stop_after and len(events) >= stop_after:
                c.close()
                return r.status, dict(r.getheaders()), events
        r.read()  # the chunked terminator, so the connection can be reused
        if conn is None:
            c.close()
        return r.status, dict(r.getheaders()), events

    def stats(self):
        return self.get("/stats")[1]


def run(cmd, **kw):
    out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", **kw)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {out.stderr[-400:]}")
    return out.stdout


def tokenize(op):
    return json.loads(run([TOKCLI, "--tokenizer", TINY / "tokenizer.json"], input=json.dumps(op) + "\n"))


def prompt_ids(text):
    return tokenize({"op": "encode", "text": "<|begin_of_text|>" + text, "special": True})["ids"]


def decode(ids):
    return tokenize({"op": "decode", "ids": ids, "skip": True})["text"]


def cli_generate(ids, max_new, temperature=0.0, seed=0, model=TINY):
    """stride_cli on the same ids: tokens and fingerprint."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        inp.write_text(json.dumps({"id": 0, "tokens": ids}) + "\n")
        args = ["--model", model, "--input", inp, "--output", out, "--max-new", max_new, "--kv-gb", "0.25"]
        if temperature > 0:
            args += ["--temperature", temperature, "--seed", seed]
        run([CLI, *args])
        r = json.loads(out.read_text())
    return r["tokens"], "stride-" + r["fingerprint"]


def content(events):
    """Text of a chat or completions SSE stream, plus the parsed chunks."""
    chunks = [json.loads(e) for e in events if e != "[DONE]"]
    text = ""
    for ch in chunks:
        for choice in ch["choices"]:
            text += choice["delta"].get("content", "") if "delta" in choice else choice["text"]
    return text, chunks


SRV = SMALL = None


def setUpModule():
    global SRV, SMALL
    for exe in (SERVER, CLI, TOKCLI):
        if not exe.exists():
            raise RuntimeError(f"{exe} not built; run cmake with -DSTRIDE_SERVER=ON")
    if not (TINY / "model.safetensors").exists():
        subprocess.run([sys.executable, str(ROOT / "tests" / "make_tiny_model.py"), "--out", str(TINY)], check=True, capture_output=True)
    if not (TINY / "tokenizer.json").exists():
        from make_tiny_tokenizer import build

        build(str(TINY))
    SRV = Proc(TINY, "--kv-gb", "0.25", "--max-seqs", "16", "--max-rows", "64", "--prefill-chunk", "16")
    # 10 KV blocks: concurrent requests must be preempted and replayed.
    SMALL = Proc(TINY, "--kv-gb", "0.0005", "--max-seqs", "16", "--max-rows", "64", "--prefill-chunk", "16")


def tearDownModule():
    codes = [p.stop() for p in (SRV, SMALL) if p]
    assert codes == [0, 0], f"servers did not exit cleanly on SIGTERM: {codes}"
    time.sleep(0.2)
    assert SRV.err == [] and SMALL.err == [], f"servers logged without --verbose: {SRV.err[:3]} {SMALL.err[:3]}"


class Basics(unittest.TestCase):
    def test_health_models_stats(self):
        self.assertEqual(SRV.get("/health"), (200, {"status": "ok"}))
        status, models = SRV.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(models["object"], "list")
        self.assertEqual(models["data"][0]["id"], "tiny")
        self.assertEqual(models["data"][0]["max_model_len"], 2048)
        stats = SRV.stats()
        for key in ("steps", "rows", "generated", "preemptions", "replayed_rows", "peak_seqs", "peak_blocks"):
            self.assertIsInstance(stats[key], int, key)
        self.assertEqual(stats["kv_total_blocks"], SRV.kv_blocks)

    def test_unknown_route(self):
        c = SRV.conn()
        c.request("GET", "/nope")
        self.assertEqual(c.getresponse().status, 404)

    def test_completion_non_streaming(self):
        body = {"model": "anything", "prompt": PROMPTS[0], "max_tokens": 12, "ignore_eos": True}
        status, r = SRV.post("/v1/completions", body)
        self.assertEqual(status, 200)
        self.assertEqual(r["object"], "text_completion")
        self.assertEqual(r["model"], "anything")
        self.assertRegex(r["id"], r"^cmpl-")
        self.assertRegex(r["system_fingerprint"], FP)
        choice = r["choices"][0]
        self.assertEqual((choice["index"], choice["finish_reason"]), (0, "length"))
        self.assertIsInstance(choice["text"], str)
        n_prompt = len(prompt_ids(PROMPTS[0]))
        self.assertEqual(r["usage"], {"prompt_tokens": n_prompt, "completion_tokens": 12, "total_tokens": n_prompt + 12})

    def test_chat_non_streaming_prompt_tokens_match_template(self):
        msgs = [{"role": "system", "content": " Be brief. "}, {"role": "user", "content": "Hi there"}]
        status, r = SRV.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 6, "ignore_eos": True})
        self.assertEqual(status, 200)
        self.assertEqual(r["object"], "chat.completion")
        self.assertEqual(r["choices"][0]["message"]["role"], "assistant")
        ref = tokenize({"op": "chat", "messages": msgs, "gen": True, "date": time.strftime("%d %b %Y")})
        self.assertEqual(r["usage"]["prompt_tokens"], len(ref["ids"]))
        self.assertEqual(r["usage"]["completion_tokens"], 6)

    def test_content_parts_and_max_completion_tokens(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        status, r = SRV.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 9, "max_completion_tokens": 4, "ignore_eos": True})
        self.assertEqual(status, 200)
        self.assertEqual(r["usage"]["completion_tokens"], 4)

    def test_default_max_new(self):
        status, r = SRV.post("/v1/completions", {"prompt": "hi", "ignore_eos": True})
        self.assertEqual(status, 200)
        self.assertEqual(r["usage"]["completion_tokens"], 256)  # --max-new-default

    def test_keep_alive_across_requests(self):
        c = SRV.conn()
        body = {"prompt": "hi", "max_tokens": 3, "ignore_eos": True}
        for _ in range(3):
            self.assertEqual(SRV.post("/v1/completions", body, conn=c)[0], 200)
            status, _, events = SRV.stream("/v1/completions", body, conn=c)
            self.assertEqual((status, events[-1]), (200, "[DONE]"))
        c.close()

    def test_errors(self):
        bad = [
            ("/v1/chat/completions", "{not json", "valid JSON"),
            ("/v1/chat/completions", {"messages": []}, "messages"),
            ("/v1/chat/completions", {"messages": [{"role": "user"}], "n": 2}, "n=1"),
            ("/v1/chat/completions", {"messages": [{"role": "tool", "content": "x"}]}, "unsupported role"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "temperature": -1}, "temperature"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "top_p": 0}, "top_p"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "logprobs": True}, "logprobs"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "stop": [""]}, "stop"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "min_tokens": 3}, "min_tokens"),
            ("/v1/completions", {"max_tokens": 3}, "prompt"),
            ("/v1/completions", {"prompt": "hi", "max_tokens": 5000}, "context"),
            ("/v1/completions", {"prompt": "word " * 3000, "max_tokens": 5}, "context"),
        ]
        for path, body, needle in bad:
            status, r = SRV.post(path, body)
            self.assertEqual(status, 400, (body, r))
            self.assertIn(needle, r["error"]["message"])
            self.assertEqual(r["error"]["type"], "invalid_request_error")
        status, r = SRV.post("/v1/completions", {"prompt": "hi", "max_tokens": 3, "ignore_eos": True, "min_tokens": 3})
        self.assertEqual(status, 200)  # min_tokens is moot once EOS is ignored

    def test_seed_and_temperature(self):
        body = {"prompt": PROMPTS[1], "max_tokens": 24, "temperature": 1.0, "ignore_eos": True}
        a = SRV.post("/v1/completions", dict(body, seed=7))[1]
        b = SRV.post("/v1/completions", dict(body, seed=7))[1]
        c = SRV.post("/v1/completions", dict(body, seed=8))[1]
        self.assertEqual((a["choices"][0]["text"], a["system_fingerprint"]), (b["choices"][0]["text"], b["system_fingerprint"]))
        self.assertNotEqual(a["system_fingerprint"], c["system_fingerprint"])
        d = SRV.post("/v1/completions", dict(body, seed=7, top_k=5, top_p=0.9))[1]
        e = SRV.post("/v1/completions", dict(body, seed=7, top_k=5, top_p=0.9))[1]
        self.assertEqual(d["system_fingerprint"], e["system_fingerprint"])


class Streaming(unittest.TestCase):
    def test_chat_stream_shape(self):
        body = {"messages": [{"role": "user", "content": "Tell me a story"}], "max_tokens": 10, "ignore_eos": True}
        status, headers, events = SRV.stream("/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        self.assertEqual(events[-1], "[DONE]")
        text, chunks = content(events)
        self.assertEqual(len({c["id"] for c in chunks}), 1)
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))
        last = chunks[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "length")
        self.assertRegex(last["system_fingerprint"], FP)
        self.assertNotIn("usage", last)
        _, ns = SRV.post("/v1/chat/completions", body)
        self.assertEqual(text, ns["choices"][0]["message"]["content"])
        self.assertEqual(last["system_fingerprint"], ns["system_fingerprint"])

    def test_include_usage(self):
        body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 7, "ignore_eos": True, "stream_options": {"include_usage": True}}
        _, _, events = SRV.stream("/v1/chat/completions", body)
        _, chunks = content(events)
        usage = chunks[-1]
        self.assertEqual(usage["choices"], [])
        self.assertEqual(usage["usage"]["completion_tokens"], 7)
        self.assertEqual(usage["usage"]["total_tokens"], usage["usage"]["prompt_tokens"] + 7)
        self.assertRegex(usage["system_fingerprint"], FP)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[-2]["system_fingerprint"], usage["system_fingerprint"])
        _, ns = SRV.post("/v1/chat/completions", {k: v for k, v in body.items() if k != "stream_options"})
        self.assertEqual(usage["usage"]["prompt_tokens"], ns["usage"]["prompt_tokens"])

    def test_completions_stream(self):
        body = {"prompt": PROMPTS[2], "max_tokens": 15, "ignore_eos": True, "stream_options": {"include_usage": True}}
        _, _, events = SRV.stream("/v1/completions", body)
        text, chunks = content(events)
        self.assertTrue(all(c["object"] == "text_completion" for c in chunks))
        final = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]][0]
        self.assertEqual(final["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 15)
        _, ns = SRV.post("/v1/completions", {k: v for k, v in body.items() if k != "stream_options"})
        self.assertEqual(text, ns["choices"][0]["text"])
        self.assertEqual(final["system_fingerprint"], ns["system_fingerprint"])

    def test_text_is_valid_and_matches_batch_path(self):
        # High temperature on a random model produces byte fragments that split
        # characters across tokens; the streamed text must still equal decoding
        # the same token ids in one go.
        for seed in (1, 2, 3):
            body = {"prompt": PROMPTS[4], "max_tokens": 120, "temperature": 1.5, "seed": seed, "ignore_eos": True}
            _, _, events = SRV.stream("/v1/completions", body)
            text, chunks = content(events)
            tokens, fp = cli_generate(prompt_ids(PROMPTS[4]), 120, temperature=1.5, seed=seed)
            self.assertEqual(text, decode(tokens), f"seed {seed}")
            self.assertEqual(chunks[-1]["system_fingerprint"], fp)
            text.encode("utf-8")  # no lone surrogates


class Parity(unittest.TestCase):
    def test_greedy_matches_cli(self):
        for p in PROMPTS[:6]:
            tokens, fp = cli_generate(prompt_ids(p), 20)
            _, r = SRV.post("/v1/completions", {"prompt": p, "max_tokens": 20, "ignore_eos": True})
            self.assertEqual(r["choices"][0]["text"], decode(tokens), p)
            self.assertEqual(r["system_fingerprint"], fp, p)

    def test_sampling_matches_cli(self):
        for seed in (11, 12):
            tokens, fp = cli_generate(prompt_ids(PROMPTS[8]), 24, temperature=0.9, seed=seed)
            _, r = SRV.post("/v1/completions", {"prompt": PROMPTS[8], "max_tokens": 24, "temperature": 0.9, "seed": seed, "ignore_eos": True})
            self.assertEqual(r["choices"][0]["text"], decode(tokens))
            self.assertEqual(r["system_fingerprint"], fp)


class Stops(unittest.TestCase):
    def test_stop_strings(self):
        body = {"prompt": PROMPTS[3], "max_tokens": 60, "temperature": 1.0, "seed": 5, "ignore_eos": True}
        _, full = SRV.post("/v1/completions", body)
        text = full["choices"][0]["text"]
        self.assertGreater(len(text), 20)
        for needle in (text[7:9], text[12:18], text[20:21]):
            expect = text[: text.index(needle)]
            _, r = SRV.post("/v1/completions", dict(body, stop=needle))
            self.assertEqual((r["choices"][0]["text"], r["choices"][0]["finish_reason"]), (expect, "stop"), repr(needle))
            self.assertLess(r["usage"]["completion_tokens"], 60)
            _, _, events = SRV.stream("/v1/completions", dict(body, stop=[needle, "\x00never"]))
            streamed, chunks = content(events)
            self.assertEqual(streamed, expect, repr(needle))
            final = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]][0]
            self.assertEqual(final["choices"][0]["finish_reason"], "stop")
            self.assertRegex(final["system_fingerprint"], FP)
        _, r = SRV.post("/v1/completions", dict(body, stop="\x00never"))
        self.assertEqual((r["choices"][0]["text"], r["choices"][0]["finish_reason"]), (text, "length"))

    def test_stop_token_from_generation_config(self):
        tokens, _ = cli_generate(prompt_ids(PROMPTS[0]), 12)
        stop_id = tokens[5]
        first = tokens.index(stop_id)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                shutil.copy(TINY / name, tmp)
            (Path(tmp) / "generation_config.json").write_text(json.dumps({"eos_token_id": [stop_id, 999]}))
            srv = Proc(tmp, "--kv-gb", "0.25", "--verbose")
            try:
                body = {"prompt": PROMPTS[0], "max_tokens": 12}
                _, r = srv.post("/v1/completions", body)
                self.assertEqual(r["choices"][0]["finish_reason"], "stop")
                self.assertEqual(r["usage"]["completion_tokens"], first + 1)  # the stop token counts, its text does not
                self.assertEqual(r["choices"][0]["text"], decode(tokens[:first]))
                _, _, events = srv.stream("/v1/completions", body)
                streamed, chunks = content(events)
                self.assertEqual(streamed, decode(tokens[:first]))
                _, r2 = srv.post("/v1/completions", dict(body, ignore_eos=True))
                self.assertEqual((r2["choices"][0]["finish_reason"], r2["usage"]["completion_tokens"]), ("length", 12))
                lines = [l for l in srv.err if l.startswith("stride_server:")]
                self.assertEqual(len(lines), 3, lines)  # one line per request with --verbose
                self.assertIn("finish=stop", lines[0])
            finally:
                self.assertEqual(srv.stop(), 0)


class Determinism(unittest.TestCase):
    def fingerprints(self, srv, jobs):
        def one(job):
            p, stream, extra = job
            body = dict({"prompt": p, "max_tokens": 40, "ignore_eos": True}, **extra)
            if stream:
                _, _, events = srv.stream("/v1/completions", body)
                text, chunks = content(events)
                final = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]][0]
                return text, final["system_fingerprint"]
            _, r = srv.post("/v1/completions", body)
            return r["choices"][0]["text"], r["system_fingerprint"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            return list(ex.map(one, jobs))

    def check(self, srv, long_output=False):
        extras = [{}, {"temperature": 0.8, "seed": 3}, {"temperature": 1.2, "seed": 4, "top_k": 20}]
        jobs = [(p, i % 2 == 0, extras[i % 3]) for i, p in enumerate(PROMPTS)]
        if long_output:
            jobs = [(p, s, dict(e, max_tokens=200)) for p, s, e in jobs]
        sequential = [self.fingerprints(srv, [j])[0] for j in jobs]
        for round_ in range(3):
            concurrent_ = self.fingerprints(srv, jobs)
            self.assertEqual(concurrent_, sequential, f"round {round_}")
        for _, fp in sequential:
            self.assertRegex(fp, FP)
        self.assertEqual(len({fp for _, fp in sequential}), len(jobs))
        return sequential

    def test_concurrent_equals_sequential(self):
        self.check(SRV)

    def test_equal_under_kv_preemption(self):
        before = SMALL.stats()["preemptions"]
        small = self.check(SMALL, long_output=True)
        self.assertGreater(SMALL.stats()["preemptions"], before, "KV pressure never forced a preemption")
        # Same requests with a roomy arena give the same outputs.
        jobs = [(p, i % 2 == 0, dict(e, max_tokens=200)) for i, (p, e) in enumerate(zip(PROMPTS, [{}, {"temperature": 0.8, "seed": 3}, {"temperature": 1.2, "seed": 4, "top_k": 20}] * 4))]
        self.assertEqual(self.fingerprints(SRV, jobs), small)


class Cancellation(unittest.TestCase):
    def wait_for(self, predicate, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_streaming_disconnect_cancels(self):
        base = SRV.stats()
        body = {"prompt": PROMPTS[0], "max_tokens": 1900, "ignore_eos": True}
        SRV.stream("/v1/completions", body, stop_after=3)
        self.assertTrue(self.wait_for(lambda: SRV.stats()["active"] == 0), "request still active after disconnect")
        time.sleep(0.3)  # the engine drops the sequence at its next step
        a = SRV.stats()
        self.assertEqual(a["cancelled"], base["cancelled"] + 1)
        self.assertLess(a["generated"] - base["generated"], 1900)
        print(f"[generated {a['generated'] - base['generated']} of 1900 tokens before the stream was cancelled]", end=" ")
        time.sleep(0.3)
        self.assertEqual(SRV.stats()["generated"], a["generated"], "engine kept generating for a closed client")
        self.assertEqual(SRV.post("/v1/completions", {"prompt": "hi", "max_tokens": 3, "ignore_eos": True})[0], 200)

    def test_non_streaming_disconnect_cancels(self):
        base = SRV.stats()
        c = SRV.conn()
        c.request("POST", "/v1/completions", json.dumps({"prompt": PROMPTS[0], "max_tokens": 1900, "ignore_eos": True}), {"Content-Type": "application/json"})
        time.sleep(0.15)
        c.close()
        self.assertTrue(self.wait_for(lambda: SRV.stats()["active"] == 0), "request still active after disconnect")
        a = SRV.stats()
        self.assertEqual(a["cancelled"], base["cancelled"] + 1)
        self.assertLess(a["generated"] - base["generated"], 1900)

    def test_many_connections(self):
        # 700 connections held open at once (more than the 512 pool threads), each
        # then makes a request.
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096) if hard != resource.RLIM_INFINITY else 4096, hard))
        n = 700
        barrier = threading.Barrier(n, timeout=60)

        def one(i):
            c = SRV.conn()
            try:
                c.connect()
                barrier.wait()
                return SRV.post("/v1/completions", {"prompt": PROMPTS[i % len(PROMPTS)], "max_tokens": 4, "ignore_eos": True}, conn=c)[0]
            finally:
                c.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            self.assertEqual(set(ex.map(one, range(n))), {200})


if __name__ == "__main__":
    unittest.main(verbosity=2)
