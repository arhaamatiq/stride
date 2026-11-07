#!/usr/bin/env python3
"""Tokenizer parity with HF AutoTokenizer, through the tokenize_cli helper.

Needs tests/data/tokenizer.json and tokenizer_config.json from
unsloth/Llama-3.2-1B-Instruct (fetched here if missing) and a build with
-DSTRIDE_SERVER=ON (STRIDE_BUILD names the build dir, default build-cpu).

  python tests/test_tokenizer.py
"""
import functools
import glob
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

DATA = ROOT / "tests" / "data"
BUILD = Path(os.environ.get("STRIDE_BUILD", ROOT / "build-cpu"))
CLI = BUILD / "tokenize_cli"
REPO = "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/"
TOKENIZER_BYTES = 17_209_920

HF = None
HF_SPLIT = None
STATS = {}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (DATA / name).exists():
            urllib.request.urlretrieve(REPO + name, DATA / name)
    if (DATA / "tokenizer.json").stat().st_size != TOKENIZER_BYTES:
        raise RuntimeError("tests/data/tokenizer.json is not the expected file")


def run_cli(tokenizer_json, requests):
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [str(CLI), "--tokenizer", str(tokenizer_json)],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError("tokenize_cli failed: " + proc.stderr[-500:])
    out = [json.loads(line) for line in proc.stdout.split("\n") if line]
    if len(out) != len(requests):
        raise RuntimeError(f"tokenize_cli returned {len(out)} lines for {len(requests)} requests")
    return out


def setUpModule():
    global HF, HF_SPLIT
    from transformers import AutoTokenizer

    fetch()
    if not CLI.exists():
        raise RuntimeError(f"{CLI} not built; run cmake with -DSTRIDE_SERVER=ON")
    HF = AutoTokenizer.from_pretrained(str(DATA))
    HF_SPLIT = AutoTokenizer.from_pretrained(str(DATA), split_special_tokens=True)


def stdlib_texts():
    import sysconfig

    stdlib = sysconfig.get_paths()["stdlib"]
    with open(os.path.join(stdlib, "pydoc_data", "topics.py"), encoding="utf-8") as f:
        prose = f.read()
    code = []
    for path in sorted(glob.glob(os.path.join(stdlib, "*.py"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            code.append(f.read())
    return prose, code


def pools():
    def rng(a, b):
        return [chr(c) for c in range(a, b + 1)]

    return {
        "ascii": rng(32, 126),
        "latin": rng(0xC0, 0x17F),
        "greek": rng(0x370, 0x3FF),
        "cyrillic": rng(0x400, 0x4FF),
        "arabic": rng(0x600, 0x6FF),
        "hebrew": rng(0x5D0, 0x5EA),
        "thai": rng(0xE01, 0xE5B),
        "devanagari": rng(0x900, 0x97F),
        "cjk": rng(0x4E00, 0x4E00 + 400),
        "hangul": rng(0xAC00, 0xAC00 + 200),
        "kana": rng(0x3041, 0x30FF),
        "emoji": rng(0x1F300, 0x1F64F) + ["\u200d", "\ufe0f", "\u2764", "\u2705"] + rng(0x1F3FB, 0x1F3FF),
        "combining": rng(0x300, 0x36F),
        "math": rng(0x2200, 0x22FF) + rng(0x2190, 0x21FF),
        "punct": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + ["\u2018", "\u2019", "\u201c", "\u201d", "\u2014", "\u2026"],
        "digits": list("0123456789") + rng(0x660, 0x669) + rng(0x2160, 0x2170) + ["\u00b2", "\u00b3", "\u00bd"],
        "space": [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x85", "\xa0", "\u1680", "\u180e", "\u2000", "\u2003",
                  "\u200a", "\u200b", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\ufeff"],
        "control": [chr(c) for c in range(0, 32)] + ["\x7f", "\x80", "\x9f"],
        "private": rng(0xE000, 0xE0FF) + rng(0x1F000, 0x1F02B),
    }


EDGE = [
    "", " ", "  ", "\n", "\n\n", "\r\n", "\t", "a", "A", "0", "hello world", "Hello, World!", " hello", "hello ", " hello ",
    "it's", "IT'S", "I'M", "we're", "They'VE", "don't", "can't", "'s", "'S", "'ll", "'LL", "x'd", "'\u017ftill", "I'\u017f",
    "\u2019s", "rock 'n' roll", "12", "123", "1234", "12345678901234567890", "3.14159", "1,000,000", "x1y22z333w4444",
    "\u0661\u0662\u0663\u0664", "\u2160\u2161\u2162", "\u00b2\u00b3", "a\u00a0b", "\u3000\u3000x", "x\u2003\u2003y",
    "x\u180e y", "x\u180e\u180ey", "x\u2029\u2029y", "a\x85b", "a\x1cb", "a\x0bb", "a\x0cb", "\x00", "a\x00b", "\x1b[31mred\x1b[0m",
    "   ", " " * 7, " " * 64, "\t\t\t", "  \n", "\n  ", "\n\n\n  \n x", "a\r\n\r\nb", "a \n b", "a\n \nb", "x  y   z",
    "foo.bar(baz, qux=1)", "def f(x):\n    return x + 1\n", "if (a<b && c>d) { e = f<g>(h); }", "<html><body>hi</body></html>",
    "https://example.com/path?q=1&r=2#frag", "user@example.com", "C:\\Users\\name\\file.txt", "/usr/local/bin/python3",
    '{"key": "value", "list": [1, 2, 3], "nested": {"a": null}}', "SELECT * FROM t WHERE id = 42;", "#include <stdio.h>\nint main() {}",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", "\uc548\ub155\ud558\uc138\uc694", "\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e", "\u0e01\u0e32", "\u0e01\u0e33", "a\u0301", "e\u0301\u0302", "\u00e9", "e\u0301",
    "\U0001f600", "\U0001f600 x", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466", "\U0001f1fa\U0001f1f8", "\U0001f44d\U0001f3fd",
    "\u2764\ufe0f", "caf\u00e9", "na\u00efve", "\u00fc\u00f6\u00e4\u00df", "\U0001d54f\U0001d550", "\ufb01nd", "\uff21\uff22\uff23", "\ud55c\uae00",
    "<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "a<|eot_id|>b", "<|eot_id|><|eot_id|>",
    "<|eot_id|", "<|eot_id|>>", "<<|eot_id|>", "<|eot", "<|", "|>", "<|reserved_special_token_250|>", "<|reserved_special_token_0|>x",
    "<|finetune_right_pad_id|>", "<|python_tag|>{}", "<|eom_id|>", "<|end_of_text|>", "<|not_a_token|>", "<|EOT_ID|>",
    "hello<|eot_id|>\n\nworld", "<|begin_of_text|><|begin_of_text|>", " <|eot_id|> ", "\n<|eot_id|>\n",
]


def make_cases(seed=0):
    rnd = random.Random(seed)
    prose, code = stdlib_texts()
    pool = pools()
    names = sorted(pool)
    with open(DATA / "tokenizer.json") as f:
        specials = [t["content"] for t in json.load(f)["added_tokens"]]
    cases = [("edge", s) for s in EDGE]
    for _ in range(500):
        src = rnd.choice(code)
        lines = src.split("\n")
        start = rnd.randrange(max(1, len(lines)))
        cases.append(("code", "\n".join(lines[start : start + rnd.randint(1, 60)])))
    for _ in range(400):
        start = rnd.randrange(len(prose) - 4000)
        cases.append(("prose", prose[start : start + rnd.choice([20, 60, 200, 800, 3000])]))
    for _ in range(600):
        parts = rnd.sample(names, rnd.randint(1, 4))
        chars = [c for p in parts for c in pool[p]]
        cases.append(("unicode", "".join(rnd.choice(chars) for _ in range(rnd.randint(1, 150)))))
    ws = pool["space"]
    for _ in range(250):
        pieces = []
        for _ in range(rnd.randint(1, 12)):
            r = rnd.random()
            if r < 0.5:
                pieces.append(rnd.choice(ws) * rnd.choice([1, 1, 2, 3, 8, 40]))
            elif r < 0.75:
                pieces.append("".join(rnd.choice(pool["ascii"]) for _ in range(rnd.randint(1, 6))))
            else:
                pieces.append(rnd.choice(["\n", "\r\n", "\n\n", "\t", "  "]))
        cases.append(("whitespace", "".join(pieces)))
    for _ in range(300):
        pieces = []
        for _ in range(rnd.randint(1, 8)):
            r = rnd.random()
            if r < 0.4:
                pieces.append(rnd.choice(specials))
            elif r < 0.5:
                pieces.append(rnd.choice(specials)[: rnd.randint(1, 8)])
            else:
                start = rnd.randrange(len(prose) - 100)
                pieces.append(prose[start : start + rnd.randint(0, 40)])
        cases.append(("special", "".join(pieces)))
    for _ in range(150):
        pieces = []
        for _ in range(rnd.randint(1, 10)):
            r = rnd.random()
            if r < 0.5:
                pieces.append("".join(rnd.choice("0123456789") for _ in range(rnd.randint(1, 12))))
            elif r < 0.8:
                pieces.append("".join(rnd.choice("abcxyz.,-/ ") for _ in range(rnd.randint(1, 4))))
            else:
                pieces.append(rnd.choice(pool["digits"]))
        cases.append(("digits", "".join(pieces)))
    mixed = "".join(rnd.choice(pool[n]) for n in rnd.choices(names, k=200_000) for _ in range(1))
    for text in (
        prose[:400_000],
        "a" * 300_000,
        " " * 100_000 + "x",
        "\n" * 50_000 + "y",
        "\u00e9" * 100_000,
        "\U0001f600" * 50_000,
        mixed,
        "hello world " * 40_000,
        "0123456789" * 30_000,
        ("x" * 999 + " ") * 300,
    ):
        cases.append(("long", text))
    return cases


class Encode(unittest.TestCase):
    def check(self, tokenizer_json, hf, cases, special, label):
        ours = run_cli(tokenizer_json, [{"op": "encode", "text": t, "special": special} for _, t in cases])
        texts = [t for _, t in cases]
        ref = hf(texts, add_special_tokens=False)["input_ids"]
        bad = [(c, t, o["ids"], r) for (c, t), o, r in zip(cases, ours, ref) if o["ids"] != r]
        per = {}
        for c, _ in cases:
            per[c] = per.get(c, 0) + 1
        STATS[label] = (len(cases), len(bad), per)
        if bad:
            lines = []
            for c, t, o, r in bad[:5]:
                i = next((k for k in range(min(len(o), len(r))) if o[k] != r[k]), min(len(o), len(r)))
                lines.append(f"[{c}] {t[:80]!r} len={len(t)} first diff at token {i}: ours={o[i:i+4]} hf={r[i:i+4]}")
            self.fail(f"{len(bad)}/{len(cases)} mismatches\n" + "\n".join(lines))

    def test_encode_special_allowed(self):
        self.check(DATA / "tokenizer.json", HF, make_cases(), True, "encode allow_special=True")

    def test_encode_special_split(self):
        cases = [c for c in make_cases(1) if c[0] in ("edge", "special", "unicode", "code")]
        self.check(DATA / "tokenizer.json", HF_SPLIT, cases, False, "encode allow_special=False")

    def test_merge_loop_without_ignore_merges(self):
        # The real file sets ignore_merges, so most words never reach the merge
        # loop. Turn it off and compare against HF loading the same file.
        from transformers import PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as tmp:
            with open(DATA / "tokenizer.json") as f:
                j = json.load(f)
            j["model"]["ignore_merges"] = False
            path = Path(tmp) / "tokenizer.json"
            with open(path, "w") as f:
                json.dump(j, f)
            hf = PreTrainedTokenizerFast(tokenizer_file=str(path))
            cases = random.Random(7).sample([c for c in make_cases(2) if c[0] != "long"], 1200)
            cases += [("long", "a" * 20000), ("long", "ab" * 10000), ("long", "x" * 999 + " y")]
            self.check(path, hf, cases, True, "encode ignore_merges=False (real merges)")

    def test_tiny_tokenizer(self):
        from make_tiny_tokenizer import build
        from transformers import PreTrainedTokenizerFast

        for ignore in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                path = build(tmp, ignore_merges=ignore)
                hf = PreTrainedTokenizerFast(tokenizer_file=path)
                cases = random.Random(8).sample([c for c in make_cases(3) if c[0] != "long"], 1000)
                self.check(path, hf, cases, True, f"encode tiny tokenizer ignore_merges={ignore}")

    def test_every_code_point(self):
        # Each scalar value in three contexts. Catches \p{L}, \p{N} and \s
        # differences between PCRE2 and Oniguruma. STRIDE_FULL_SWEEP=1 does all
        # 1.1M code points; by default all below U+3000 and every 13th above.
        full = os.environ.get("STRIDE_FULL_SWEEP")
        cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and (full or c < 0x3000 or c % 13 == 0)]
        cases = []
        for c in cps:
            ch = chr(c)
            cases += [("cp", "a" + ch + "b"), ("cp", " " + ch + ch + "1"), ("cp", ch + "\n" + ch)]
        self.check(DATA / "tokenizer.json", HF, cases, True, "encode every code point" + (" (full)" if full else " (subsampled)"))

    def test_invalid_utf8_input(self):
        # Invalid bytes become U+FFFD (one per maximal subpart), as Python's errors="replace".
        rnd = random.Random(11)
        raw = [bytes(rnd.choice([rnd.randrange(256), rnd.randrange(0x80, 0x100), 0x61]) for _ in range(rnd.randint(1, 40))) for _ in range(400)]
        raw += [b"\xf0\x9f\x98", b"abc\xc3", b"\xe0\x80\x80x", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"ok \xff ok", b"\x80\x80", b"\xc0\xaf", b"\xf0\x9f\x98\x80\xf0\x9f"]
        ours = run_cli(DATA / "tokenizer.json", [{"op": "encode", "hex": b.hex(), "special": True} for b in raw])
        ref = HF([b.decode("utf-8", errors="replace") for b in raw], add_special_tokens=False)["input_ids"]
        bad = [b for b, o, r in zip(raw, ours, ref) if o["ids"] != r]
        STATS["encode invalid utf-8"] = (len(raw), len(bad), {})
        self.assertEqual(bad[:3], [])


class Chat(unittest.TestCase):
    def test_templates(self):
        rnd = random.Random(5)
        prose, _ = stdlib_texts()
        pool = pools()
        trim_chars = pool["space"] + [" ", "\n"]
        specials = ["<|eot_id|>", "<|start_header_id|>", "<|begin_of_text|>"]

        def content():
            r = rnd.random()
            if r < 0.1:
                body = ""
            elif r < 0.6:
                start = rnd.randrange(len(prose) - 400)
                body = prose[start : start + rnd.randint(1, 300)]
            elif r < 0.8:
                body = "".join(rnd.choice(pool[rnd.choice(["cjk", "emoji", "ascii", "latin", "arabic"])]) for _ in range(rnd.randint(1, 40)))
            else:
                body = rnd.choice(specials) + "x"
            pad = lambda: "".join(rnd.choice(trim_chars) for _ in range(rnd.choice([0, 0, 1, 3])))
            return pad() + body + pad()

        convs = []
        dates = ["06 Nov 2026", "01 Jan 2024", "29 Feb 2028", "26 Jul 2024"]
        for _ in range(500):
            msgs = []
            if rnd.random() < 0.4:
                msgs.append({"role": "system", "content": content()})
            for k in range(rnd.randint(1, 7)):
                msgs.append({"role": "user" if k % 2 == 0 else "assistant", "content": content()})
            if rnd.random() < 0.1:
                msgs.insert(rnd.randrange(len(msgs) + 1), {"role": "system", "content": content()})
            convs.append((msgs, rnd.random() < 0.8, rnd.choice(dates)))
        convs.append(([{"role": "user", "content": "hi"}], True, "06 Nov 2026"))
        convs.append(([], True, "06 Nov 2026"))
        ours = run_cli(
            DATA / "tokenizer.json",
            [{"op": "chat", "messages": m, "gen": g, "date": d} for m, g, d in convs],
        )
        bad = []
        id_bad = 0
        for (msgs, gen, date), o in zip(convs, ours):
            try:
                ref = HF.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, date_string=date)
            except Exception:
                if msgs:
                    raise
                ref = None  # HF's template raises on an empty list
            if ref is None:
                continue
            if o["text"] != ref:
                bad.append((msgs, ref, o["text"]))
            elif o["ids"] != HF(ref, add_special_tokens=False)["input_ids"]:
                id_bad += 1
        STATS["chat templates"] = (len(convs), len(bad) + id_bad, {"text+ids": len(convs)})
        if bad:
            m, ref, got = bad[0]
            self.fail(f"{len(bad)} template mismatches; first:\nmsgs={m!r}\nhf  ={ref!r}\nours={got!r}")
        self.assertEqual(id_bad, 0)

    def test_default_date_matches_today(self):
        msgs = [{"role": "user", "content": "x"}]
        ref = HF.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = run_cli(DATA / "tokenizer.json", [{"op": "chat", "messages": msgs, "gen": True, "date": time.strftime("%d %b %Y")}])
        self.assertEqual(out[0]["text"], ref)

    def test_tool_role_rejected(self):
        proc = subprocess.run(
            [str(CLI), "--tokenizer", str(DATA / "tokenizer.json")],
            input=json.dumps({"op": "chat", "messages": [{"role": "tool", "content": "x"}], "date": "d"}) + "\n",
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)


class Decode(unittest.TestCase):
    @staticmethod
    @functools.lru_cache(maxsize=1)
    def oracle_tables():
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        with open(DATA / "tokenizer.json") as f:
            special = {t["id"] for t in json.load(f)["added_tokens"] if t["special"]}
        return {v: k for k, v in bytes_to_unicode().items()}, special

    @classmethod
    def byte_oracle(cls, ids, skip):
        # Independent of HF's decoder: token strings -> bytes -> lossy UTF-8.
        inv, special = cls.oracle_tables()
        out = bytearray()
        for i in ids:
            if skip and i in special:
                continue
            tok = HF.convert_ids_to_tokens(i)
            out += tok.encode("utf-8") if i in special else bytes(inv[c] for c in tok)
        return out.decode("utf-8", errors="replace")

    def test_random_ids_and_streaming(self):
        rnd = random.Random(9)
        n_vocab = len(HF)
        seqs = []
        for _ in range(300):
            kind = rnd.random()
            if kind < 0.5:
                seqs.append([rnd.randrange(n_vocab) for _ in range(rnd.randint(1, 40))])
            elif kind < 0.8:  # byte-fragment heavy: many partial UTF-8 sequences
                seqs.append([rnd.choice(range(100, 400)) for _ in range(rnd.randint(1, 40))])
            else:
                seqs.append([rnd.choice([128000, 128001, 128009, 128006]) if rnd.random() < 0.3 else rnd.randrange(1000) for _ in range(rnd.randint(1, 30))])
        reqs = []
        for ids in seqs:
            reqs.append({"op": "decode", "ids": ids, "skip": True})
            reqs.append({"op": "decode", "ids": ids, "skip": False})
        ours = run_cli(DATA / "tokenizer.json", reqs)
        bad = 0
        held_back = 0
        for k, ids in enumerate(seqs):
            for skip, o in ((True, ours[2 * k]), (False, ours[2 * k + 1])):
                want = self.byte_oracle(ids, skip)
                if o["text"] != want:
                    bad += 1
                elif "\ufffd" not in want:
                    hf = HF.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)
                    bad += hf != want
                if skip:
                    # streaming: pieces concatenate to the full text, never split a character
                    self.assertEqual("".join(o["chunks"]), o["text"])
                    held_back += sum(1 for c in o["chunks"][:-1] if c == "")
        STATS["decode"] = (2 * len(seqs), bad, {"random id sequences": len(seqs)})
        self.assertEqual(bad, 0)
        self.assertGreater(held_back, 0, "no token was held back; test did not exercise partial characters")

    def test_roundtrip_against_hf_decode(self):
        cases = [t for _, t in make_cases(4) if t and "\ufffd" not in t][:600]
        ids = [HF(t, add_special_tokens=False)["input_ids"] for t in cases]
        ours = run_cli(DATA / "tokenizer.json", [{"op": "decode", "ids": i, "skip": False} for i in ids])
        bad = 0
        for t, i, o in zip(cases, ids, ours):
            ref = HF.decode(i, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if o["text"] != ref or o["text"] != t.encode("utf-8", "surrogatepass").decode("utf-8"):
                bad += 1
        STATS["decode roundtrip"] = (len(cases), bad, {})
        self.assertEqual(bad, 0)

    def test_streaming_holds_partial_character(self):
        # U+1F44D is the bytes F0 9F 91 8D; the tokenizer splits it across tokens.
        ids = HF("a\U0001f44db", add_special_tokens=False)["input_ids"]
        out = run_cli(DATA / "tokenizer.json", [{"op": "decode", "ids": ids}])[0]
        self.assertEqual("".join(out["chunks"]), "a\U0001f44db")
        for chunk in out["chunks"]:
            self.assertNotIn("\ufffd", chunk)


class Speed(unittest.TestCase):
    def test_long_inputs_scale_linearly(self):
        prose, _ = stdlib_texts()
        reqs = [
            {"op": "time", "text": "a" * 1_000_000, "reps": 1},
            {"op": "time", "text": "a" * 4_000_000, "reps": 1},
            {"op": "time", "text": prose[:1_000_000], "reps": 3},
            {"op": "time", "text": " " * 2_000_000 + "x", "reps": 1},
        ]
        a1, a4, p, w = run_cli(DATA / "tokenizer.json", reqs)
        STATS["speed"] = (0, 0, {
            "1MB 'a' (one word) s": round(a1["seconds"], 3),
            "4MB 'a' s": round(a4["seconds"], 3),
            "1MB prose MB/s": round(3 / p["seconds"], 2),
            "2MB spaces s": round(w["seconds"], 3),
        })
        # 4x the input must not cost anywhere near 16x (quadratic).
        self.assertLess(a4["seconds"], 8 * max(a1["seconds"], 0.05))
        self.assertLess(w["seconds"], 5.0)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    print("\nsummary (compared, mismatched):")
    for label, (n, bad, extra) in STATS.items():
        print(f"  {label}: {n} compared, {bad} mismatched {extra}")
    sys.exit(0 if result.wasSuccessful() else 1)
