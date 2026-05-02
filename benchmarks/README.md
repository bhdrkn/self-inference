# Benchmarks

Each post that runs benchmarks puts its scripts in this directory under a per-post folder (e.g. `benchmarks/post-01/`).

## Workflow

```
inference server (RunPod)
        ▲
        │ HTTP
        │
vllm benchmark_serving.py  ──▶  benchmarks/results/post-N/*.json
        (runs locally)
                                          │
                                          ▼
                                 scripts/plot_results.py
                                          │
                                          ▼
                                 benchmarks/results/post-N/*.svg
```

1. Run `vllm benchmark_serving.py` from your laptop against the RunPod server. Save raw JSON output to `benchmarks/results/post-N/`.
2. Run `scripts/plot_results.py` to convert the JSON files into SVGs. SVGs are saved alongside the raw data and embedded in the post.

## Conventions

- Raw JSON outputs are **gitignored** — they can be large and are fully reproducible.
- Generated SVGs are **checked in** — they are what the post embeds and what readers see.
- Every number cited in a published post must trace back to a JSON file in `benchmarks/results/post-N/`.

## ShareGPT dataset

All posts use the ShareGPT dataset for prompt distribution:

```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Download once, reuse across all posts. The file is ~200 MB and gitignored.
