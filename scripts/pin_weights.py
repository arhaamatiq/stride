#!/usr/bin/env python3
"""Downloads the pinned model snapshot and writes weights.lock.json.

Every file comes from the commit-addressed URL, never from a moving branch. The
SHA256 of each downloaded file is computed locally and checked against the LFS
hash Hugging Face publishes for that commit, where one exists.
"""
import argparse
import datetime
import hashlib
import json
import os
import urllib.request

REPO = "unsloth/Llama-3.2-1B-Instruct"
API = "https://huggingface.co/api/models/" + REPO


def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--commit", default=None, help="pin to this commit; default is the current main")
    args = parser.parse_args()

    commit = args.commit or fetch_json(API + "/revision/main")["sha"]
    tree = fetch_json(f"{API}/tree/{commit}")
    os.makedirs(args.dest, exist_ok=True)

    files = []
    for entry in tree:
        if entry["type"] != "file" or entry["path"] in (".gitattributes", "README.md"):
            continue
        name = entry["path"]
        url = f"https://huggingface.co/{REPO}/resolve/{commit}/{name}"
        path = os.path.join(args.dest, name)
        urllib.request.urlretrieve(url, path)
        digest = sha256(path)
        size = os.path.getsize(path)
        lfs = entry.get("lfs", {}).get("oid")
        if size != entry["size"]:
            raise SystemExit(f"{name}: size {size} != published {entry['size']}")
        if lfs and lfs != digest:
            raise SystemExit(f"{name}: sha256 {digest} != published LFS oid {lfs}")
        files.append({"path": name, "bytes": size, "sha256": digest, "hf_lfs_oid": lfs, "url": url})
        print(f"{digest}  {size:>12}  {name}", flush=True)

    with open(args.lock, "w") as f:
        json.dump(
            {
                "repo": REPO,
                "commit": commit,
                "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gated_upstream": "meta-llama/Llama-3.2-1B-Instruct",
                "files": files,
            },
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
