# Jetson benchmark bundle — run instructions

Everything needed to measure fire-detection inference cost on a Jetson Orin Nano.
Copy this folder to the device, run two commands, copy the results back.

---

## Safety: what this does and does not do

The device is not yours, so the bundle is deliberately conservative.

**It does NOT:**
- install anything (no `pip`, no `apt`, no virtualenv)
- require `sudo` for any step in the normal path
- change `nvpmodel`, `jetson_clocks`, or any system/power setting
- write anything outside this folder
- access the network
- modify, upgrade, or downgrade any existing package

**It does:**
- read device info from `/proc` and `/etc` (read-only)
- load model files that ship inside this folder
- run inference on random in-memory tensors
- write results to `./results/` inside this folder

To remove every trace afterwards: `rm -rf ~/jetson`. Nothing else is touched.

> The one optional step that needs `sudo` is switching power modes (Step 5). It
> is clearly marked, entirely optional, and comes with restore instructions. Skip
> it if you would rather not change the device's state at all — the benchmark
> still produces a complete result set at whatever mode the device is already in.

---

## What gets measured

Five models, each in whatever backends the device already supports:

| Model | Task | Input |
|---|---|---|
| `iteration1` | FireCNN, binary classification | 3×224×224 |
| `iteration2` | MobileNetV3-Small, 4-class | 3×224×224 |
| `iteration3` | MobileNetV3-Small robust, 4-class | 3×224×224 |
| `iteration4` | YOLO26n, object detection | 3×640×640 |
| `iteration5` | LightweightU-Net, segmentation | 3×256×256 |

Backends attempted (each skipped cleanly if unavailable):

- ONNX Runtime + **TensorRT** (FP16) — edge-GPU tier
- ONNX Runtime + **CUDA** (FP32) — edge-GPU tier
- ONNX Runtime + **CPU** (FP32) — ARM CPU tier
- PyTorch **CUDA** FP32 / FP16, PyTorch **CPU** FP32

Models ship as **ONNX** and **TorchScript**, so the device needs neither
`torchvision`, `ultralytics`, nor this project's source code.

Protocol, identical to the workstation so numbers are comparable: random
in-memory inputs (no disk I/O in the timed region), 50 warmup + 200 timed
iterations, `torch.cuda.synchronize()` around the timed region on CUDA, and
**median + p95** reported rather than just the mean.

---

## Step 1 — Copy the bundle to the device

From the workstation (in `C:\git\fireDetection`):

```bash
scp jetson_bundle.tar.gz USER@JETSON_HOST:~/
```

Then on the Jetson:

```bash
ssh USER@JETSON_HOST
tar -xzf jetson_bundle.tar.gz
cd jetson
```

The archive is ~89 MB and expands to ~91 MB.

## Step 2 — Pre-flight check (read-only, ~5 seconds)

```bash
python3 check_env.py
```

This prints the board model, L4T/JetPack version, RAM, power mode, which Python
packages exist, which backends will be usable, and whether all model files
arrived. It ends with `READY` or `NOT READY`.

**If it says `numpy MISSING`**, stop — that is the only hard requirement. Do not
install it; tell me and I will adapt the bundle.

**If `onnxruntime` and `torch` are both missing**, there is nothing to measure.
Report back rather than installing anything.

## Step 3 — Quick smoke test (~1–2 minutes)

```bash
python3 benchmark_jetson.py --quick --skip-tensorrt
```

Only 5 warmup / 10 timed iterations, so the numbers are noisy and **should not be
used** — this only confirms every model loads and runs. Expect one line per
(model × backend).

## Step 4 — Full benchmark run

```bash
python3 benchmark_jetson.py --tag MAXN
```

Set `--tag` to whatever power mode the device is currently in (see Step 5); it is
recorded in every row so results can be grouped later.

**Runtime:** roughly 10–20 minutes without TensorRT. With TensorRT the *first*
run additionally spends several minutes per model building engines — that is
normal and not a hang. Engines are cached in `results/trt_cache/`, so a repeat
run is fast.

If TensorRT engine building is too slow or fails, this is a complete and valid
run without it:

```bash
python3 benchmark_jetson.py --tag MAXN --skip-tensorrt
```

Useful variants:

```bash
python3 benchmark_jetson.py --models iteration1 iteration4    # subset
python3 benchmark_jetson.py --batch-sizes 1 4 8               # batch sweep
python3 benchmark_jetson.py --skip-torch                      # ONNX only
```

Results append to `results/jetson_benchmarks.csv`, so multiple runs accumulate
rather than overwrite.

## Step 5 — Power modes *(optional, needs sudo — skip if you prefer)*

This is the one step that changes device state. It is worth doing because it
turns one hardware data point into several, but the thesis has a complete result
set without it.

First record the current mode so you can restore it:

```bash
sudo nvpmodel -q          # note the mode number, e.g. "NV Power Mode: 15W"
```

Then for each mode, switch and re-run:

```bash
sudo nvpmodel -m 0        # MAXN (check available modes with: sudo nvpmodel -q --verbose)
python3 benchmark_jetson.py --tag MAXN --skip-tensorrt

sudo nvpmodel -m 1        # e.g. 15W
python3 benchmark_jetson.py --tag 15W --skip-tensorrt

sudo nvpmodel -m 2        # e.g. 7W
python3 benchmark_jetson.py --tag 7W --skip-tensorrt
```

**Restore the original mode when finished:**

```bash
sudo nvpmodel -m <ORIGINAL_MODE_NUMBER>
sudo nvpmodel -q          # confirm it is back
```

Mode numbers differ between Jetson models — always read them from
`sudo nvpmodel -q --verbose` rather than assuming the numbering above.

Do **not** run `jetson_clocks`; it disables dynamic frequency scaling and leaves
the board in a non-default state until reboot.

## Step 6 — Copy results back

From the workstation:

```bash
scp -r USER@JETSON_HOST:~/jetson/results ./jetson_results
```

Then hand me `jetson_results/` and I will merge it into `results/benchmarks.csv`
and regenerate every figure and table.

Two files matter:
- `jetson_benchmarks.csv` — the measurements
- `environment.json` — device model, JetPack version, package versions (needed
  for the thesis Methodology chapter)

`results/trt_cache/` is only a build cache and does not need copying.

## Step 7 — Clean up (optional)

```bash
cd ~ && rm -rf jetson jetson_bundle.tar.gz
```

---

## Troubleshooting

| Symptom | Cause / action |
|---|---|
| `NOT READY` from `check_env.py` | Read which item is `MISSING`. Missing model files mean the copy was incomplete — re-`scp`. |
| `numpy MISSING` | Hard requirement. Do not install; report back. |
| TensorRT provider absent | Normal on some JetPack images. Use `--skip-tensorrt`; ONNX-CUDA still covers the GPU tier. |
| First TensorRT run "hangs" | Engine building, several minutes per model. Leave it, or use `--skip-tensorrt`. |
| `session failed` / `run failed` for one model | That model is skipped, the rest continue. Send me the message. |
| Killed / out of memory | Use `--batch-sizes 1` and `--models` to run a few at a time. Batch 1 is the deployment-relevant case anyway. |
| `nvpmodel: permission denied` | Reading the mode needs sudo on some images. Just pass it manually: `--tag 15W`. |
| Numbers vary between repeat runs | Expected on a thermally constrained board — that is why p95 is recorded. Re-run to quantify it. |

## Files in this bundle

```
jetson/
  README.md              this file
  check_env.py           read-only pre-flight check
  benchmark_jetson.py    the benchmark (writes ./results/)
  models/
    MANIFEST.txt         what was exported, from which checkpoints
    iteration*.onnx      ONNX graphs (ONNX Runtime)
    iteration*_weights.pt  TorchScript archives (PyTorch, self-describing)
  results/               created on first run
```

Nothing in this bundle writes outside `jetson/`.
