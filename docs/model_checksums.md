# Model checksums

SHA256 of each GGUF in `models/` at the time it was downloaded. Used as
an integrity reference when re-pulling, debugging cache corruption, or
verifying a fresh install matches what we tested against. Update when a
model file is replaced.

## Current

| File | Size (bytes) | SHA256 | Source | First captured |
|---|---|---|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | 5,290,008,448* | `03B74727A860A56338E042C4420BB3F04B2FEC5734175F4CB9FA853DAF52B7E8` | `unsloth/Qwen3.5-9B-GGUF` | (pre-existing in main checkout; checksum captured 2026-05-08) |
| `Qwen3.5-4B-Q4_K_M.gguf` | 2,740,937,888 | `00FE7986FF5F6B463E62455821146049DB6F9313603938A70800D1FB69EF11A4` | `unsloth/Qwen3.5-4B-GGUF` | 2026-05-08 (4B plan Stage B) |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | 532,517,120 | `BD258782E35F7F458F8ACED1ADC053E6E92E89BC735BA3BE89D38A06121DC517` | `unsloth/Qwen3.5-0.8B-GGUF` | 2026-05-08 (4B plan Stage B; speculative-decoding draft model) |

\* 9B size approximate; recompute with `Get-FileHash` if exact bytes
matter.

## Re-verify

```powershell
Get-FileHash -Algorithm SHA256 `
  C:\STC\ultronPrototype\models\Qwen3.5-9B-Q4_K_M.gguf, `
  C:\STC\ultronPrototype\models\Qwen3.5-4B-Q4_K_M.gguf, `
  C:\STC\ultronPrototype\models\Qwen3.5-0.8B-Q4_K_M.gguf
```

If any hash drifts, the file was either replaced (intentional — update
this doc) or corrupted (re-download via `python scripts/download_models.py`).

## Why no upstream-published SHA256 comparison

Unsloth doesn't publish a centralised checksum file alongside each GGUF
release. The integrity guarantee relies on HuggingFace Hub's
content-addressed transfer (atomic temp-file + rename, ETag check). The
hashes above lock down what we actually received so a future re-pull can
be cross-checked against this record.
