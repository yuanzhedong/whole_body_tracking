"""Shared builder for the 'Compute & deployment' report section.

Measured per-step inference latency (isolated from per-process startup) and model
sizes for HoloMotion vs BFM-Zero. Numbers are committed constants from the
benchmarks (HoloMotion ONNX params via onnx initializers; BFM-Zero actor params;
latency via warmed inference loops on the same device).
"""


def compute_blocks(wr):
    tbl = (
        "| | params | architecture | CPU latency (1 thr) | GPU latency (batch 1) |\n"
        "|---|---|---|---|---|\n"
        "| **HoloMotion** | **408.7 M** | sparse MoE (1024 experts) | 0.9 ms → ~1090 Hz | 0.49 ms → ~2030 Hz |\n"
        "| **BFM-Zero** (actor) | **31.9 M** | dense MLP | 4.6 ms → ~220 Hz | 0.89 ms → ~1130 Hz |\n")
    return [
        wr.H2(text="Compute & deployment"),
        wr.MarkdownBlock(text=(
            "Inference cost per control step (isolated from per-process startup), measured on the same "
            "device for both:")),
        wr.MarkdownBlock(text=tbl),
        wr.MarkdownBlock(text=(
            "Notes: (1) **HoloMotion is the *larger* model** — 408.7 M params — but it's a **sparse "
            "Mixture-of-Experts**, so only a few experts fire per token and it stays cheap per step. "
            "BFM-Zero is a smaller (31.9 M) dense net. So *capacity* is not why HoloMotion fails "
            "near-ground (see the root-cause section). (2) The per-step gap is **~5× on CPU but only "
            "~1.8× on GPU** — GPU parallelism favors BFM-Zero's dense net more than the MoE. (3) **Both "
            "clear the ~50 Hz humanoid control loop with large margin** → neither is a deployment "
            "bottleneck. (4) HoloMotion is measured as its deployed **ONNX**; BFM-Zero as eager "
            "PyTorch, so part of the CPU gap is packaging, not the model.")),
    ]
