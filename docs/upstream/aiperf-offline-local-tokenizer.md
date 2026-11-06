# AIPerf: a local tokenizer directory fails under HF_HUB_OFFLINE

Found while benchmarking with `aiperf` 0.13.0. Not yet reported upstream. Check for an
existing issue or fix before filing, since this was tested against the 0.13.0 release only.

## Behaviour

AIPerf's documentation says a local directory passed to `--tokenizer` is used as-is. That
holds online. With `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`, the same path crashes,
because the offline code path passes it to `snapshot_download`, which validates it as a
Hub repo ID:

```
TokenizerError: Failed to load tokenizer '/path/to/model': HFValidationError: Repo id must
be in the form 'repo_name' or 'namespace/repo_name'
```

Air-gapped runs are the main reason to set those variables, and a local tokenizer path is
the natural thing to use there, so the combination should work.

## Reproduce

```bash
HF_HUB_OFFLINE=1 python -c "from aiperf.common.tokenizer import Tokenizer; Tokenizer.from_pretrained('/path/to/local/tokenizer/dir')"
```

## Fix

[`aiperf-offline-local-tokenizer.patch`](aiperf-offline-local-tokenizer.patch) returns a
local directory unchanged from `_resolve_local_snapshot`.

Checked by hand against the installed 0.13.0 package:

| Case | Before | After |
| --- | --- | --- |
| Online, local directory | loads | loads (path not changed) |
| `HF_HUB_OFFLINE=1`, local directory | `HFValidationError` | loads |
| `HF_HUB_OFFLINE=1`, uncached repo ID | clear "not present in the cache" error | same error |

A pull request should also add a regression test for the offline local-directory case; the
patch here does not include one.
