# Performance tuning (BabelDOC / DocLayout)

For most scientific PDFs, the dominant bottleneck is usually **page layout parsing** (DocLayout / onnxruntime),
not the LLM translation step. This is especially true when the runtime has limited CPU (e.g. 2 vCPU).

## Recommended env vars (2 vCPU / 16 GB)

Limit native library thread oversubscription (important in containers where `nproc` can be misleading):

- `OMP_NUM_THREADS=2`
- `OPENBLAS_NUM_THREADS=2`
- `MKL_NUM_THREADS=2`
- `NUMEXPR_NUM_THREADS=2`

Increase DocLayout throughput by using larger batches (uses more RAM, reduces per-batch overhead):

- `BABELDOC_DOCLAYOUT_BATCH_SIZE=64`

## Advanced: auto batch controls

If you prefer dynamic batching, BabelDOC also supports caps and a best-effort memory-based limiter:

- `BABELDOC_DOCLAYOUT_MAX_BATCH_SIZE` (default: `64`)
- `BABELDOC_DOCLAYOUT_AUTO_BATCH_MEM_FRACTION` (default: `0.25`)
- `BABELDOC_DOCLAYOUT_PER_IMAGE_MB` (default: `24`)

