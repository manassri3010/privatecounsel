# GPU session checklist (E2E Networks)

A step-by-step plan for one rented-GPU benchmark session. The meter runs from **Launch** until **Delete**, so everything that can be done on the Mac is done before launching.

Plan: main benchmarks on **L40S**, budget comparison on **L4** (one session per GPU type). The optional H100 run is not covered here: on E2E, H100s are served through the separate TIR platform, not as a normal GPU node.

## Time and cost

Prices are on-demand, per GPU-hour, excluding 18% GST (checked 17 Nov 2026). E2E's docs don't say how billing is rounded, so plan as if every started hour is charged in full.

| Step | Time | L40S (₹102/h) | L4 (₹49/h) |
| --- | --- | --- | --- |
| A. Prepare on the Mac | 10 min | free | free |
| B. Launch the node | 5–10 min | ₹9–17 | ₹4–8 |
| C. Check the machine | 5 min | ₹9 | ₹4 |
| D. Copy the code over | 2 min | ₹3 | ₹2 |
| E. Smoke test (AWQ, tiny run) | 10 min | ₹17 | ₹8 |
| F. Full runs | 35 min (L40S, 3 models) / 30–50 min (L4, 2–3 models) | ₹60 | ₹25–41 |
| G. Copy results to the Mac | 3 min | ₹5 | ₹2 |
| H. **Delete the node** | 2 min | ₹3 | ₹2 |
| **Session total** | **about 1–1.5 h** | **₹102–153 (₹120–181 with GST)** | **₹49–74 (₹58–87 with GST)** |

Both sessions together come to roughly ₹180–270 including GST, well inside the ₹2,000 credit, which leaves room for a re-run.

Rough benchmark time per model: L40S about 4 min startup and 5–8 min benchmark. On the L4 the BF16 model is slow (about 15 tokens/s for one user) and only has room for about 13 users' worth of KV cache at once. Running FP8 and AWQ on the L4 is enough for the budget comparison; add BF16 only if time allows.

---

## A. Prepare on the Mac (before launching; free)

- [ ] `main` is up to date and CI is green: `git switch main && git pull && gh run list -L1`
- [ ] SSH key exists: `cat ~/.ssh/id_ed25519.pub`. Paste it into E2E MyAccount when creating the node.
- [ ] Optional: a Hugging Face **read-only** token (huggingface.co → Settings → Access Tokens). Qwen3 isn't gated; a token only avoids download rate limits.
- [ ] Note the time you'll launch and set a phone timer for 1 hour.
- [ ] Have this file open.

## B. Launch the node (5–10 min)

- [ ] MyAccount → **Compute → Nodes → Add New Node → GPU** tab. The GPU tab only appears in the **Delhi** region; switch from Chennai if needed.
- [ ] Pick the card (**L40S** or **L4**), single-GPU plan (not 2x/4x/8x).
- [ ] OS: **Ubuntu 22.04**. GPU images on 22.04 come with Docker and the NVIDIA Container Toolkit; 24.04 images don't.
- [ ] Check the plan's **disk is at least 100 GB**: the vLLM image is about 20 GB and the three models about 32 GB.
- [ ] **Billing: On-Demand.** Not Committed: committed plans are charged for the whole term even if the node is deleted early.
- [ ] **SSH Keys:** select your key.
- [ ] Check **Estimated Cost** in the Summary panel, then **Launch**.
- [ ] Wait for **Running**; the public IP is on the same row. The username and password arrive by email.

On the Mac, save the address for the commands below (use the username from the email):

```bash
export NODE=root@<public-ip>
ssh $NODE
```

## C. Check the machine (5 min)

On the node:

```bash
nvidia-smi                       # GPU name, memory, driver version, max CUDA version
df -h /                          # at least 60 GB free
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L   # containers see the GPU
which tmux || sudo apt install -y tmux
```

- **Driver below 580** (first line of `nvidia-smi`): use the CUDA 12.9 build of vLLM by running `export VLLM_TAG=v0.32.0-cu129`. Don't install a new driver; that needs a reboot and costs time. `run.sh` checks this too.
- **`permission denied` from docker:** `sudo usermod -aG docker $USER`, then log out and back in.
- **`unknown flag: --gpus`** or **`could not select device driver`:** the NVIDIA Container Toolkit is missing; see E2E's "Run GPU workloads in Docker" guide.

## D. Copy the code over (2 min)

The repo is private, so instead of cloning (which would need a GitHub token on a rented machine), send the committed files from the Mac. `git archive` copies exactly what's in `main`, never `.env` or `.venv`.

On the Mac:

```bash
cd ~/Desktop/main/Projects/privatecounsel
git archive --format=tar main | ssh $NODE 'mkdir -p privatecounsel && tar -x -C privatecounsel'
```

On the node, start a **tmux** session so the benchmarks keep running if the SSH connection drops (reconnect with `tmux attach`):

```bash
tmux new -s bench
cd ~/privatecounsel
read -rs HF_TOKEN && export HF_TOKEN    # optional: paste the token (not saved in shell history), press Enter
```

## E. Smoke test (10 min)

A tiny run with the smallest model, to prove the whole pipeline works before spending time on full runs. It also downloads the vLLM image and the AWQ weights, so the real AWQ run starts faster.

```bash
CONCURRENCIES=1 NUM_PROMPTS=4 benchmarks/run.sh smoke Qwen/Qwen3-8B-AWQ
ls benchmarks/results/smoke_*/          # c1.json, metadata.json, nvidia-smi.txt, ...
rm -rf benchmarks/results/smoke_*       # keep it out of the report
```

If this fails, read the last lines it prints and `benchmarks/results/smoke_*/vllm.log`. Fix, re-run. Don't start the full runs until it passes.

## F. Full runs (L40S about 35 min, L4 30–50 min)

Use exactly `L40S` or `L4` as the GPU name; `report.py` matches prices by it. Smallest model first, so a problem shows up early and cheaply.

**L40S:**

```bash
benchmarks/run.sh L40S Qwen/Qwen3-8B-AWQ
benchmarks/run.sh L40S Qwen/Qwen3-8B-FP8
benchmarks/run.sh L40S Qwen/Qwen3-8B
```

**L4:**

```bash
benchmarks/run.sh L4 Qwen/Qwen3-8B-AWQ
benchmarks/run.sh L4 Qwen/Qwen3-8B-FP8
benchmarks/run.sh L4 Qwen/Qwen3-8B      # optional: slow, and short of KV-cache memory at 32 users
```

Each run prints the median TTFT, ITL and output throughput per concurrency level as it goes. If a run fails, it says why and removes the container; fix and re-run just that model.

## G. Copy results to the Mac (3 min)

On the Mac:

```bash
cd ~/Desktop/main/Projects/privatecounsel
rsync -av $NODE:privatecounsel/benchmarks/results/ benchmarks/results/
ls benchmarks/results/
python benchmarks/report.py --price L40S=102 --price L4=49 --inr-per-usd <today's rate>
```

Check that every run folder is there and the report shows every model **before** deleting the node. After deletion, the data is gone.

## H. Delete the node (2 min)

**Powering off does not stop billing. Only deleting does.**

- [ ] MyAccount → **Nodes** → the node's **Actions** → **Delete Node** → confirm. If it says accidental protection is enabled, turn that off in **Actions** first.
- [ ] The node is gone from the list.
- [ ] Nothing else is left billing: saved images, reserved IPs, volumes.
- [ ] Later that day: **Billing** in MyAccount shows the session cost and no ongoing charge.

## After the session (on the Mac, free)

- [ ] Branch, commit the results folders (`vllm.log` stays local; it's gitignored), open a PR: `git switch -c bench/results-<gpu>-<date>`
- [ ] Save the report as `benchmarks/results.md` and update the Results table in `README.md`.
- [ ] Note anything that went wrong or took longer than planned in this file, so the next session is faster.
