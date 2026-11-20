# Compute runbook (RunPod)

Written from what actually worked, and failed, while running v2 on 2026-11-19/20.

## Accounts and secrets

Secrets are never committed. The helpers in `tools/` read them from `~/.secrets/` on the controller machine:
- `runpod_key`: RunPod API key with **GraphQL read/write**. The REST API (rest.runpod.io) may be unreachable
  from sandboxed environments; GraphQL at api.runpod.io works.
- `hf_token`: Hugging Face read token. Gated access is needed for `meta-llama/Llama-3.1-*` (8B and 70B
  approved) and `google/gemma-2-*` (9B and 27B approved). `Llama-3.3-70B-Instruct` was **not** approved.
- `jupyter_pw`: password set as the pod's `JUPYTER_PASSWORD`.

On each pod, the RunPod key is stored at `/root/.rp/key` (chmod 600) so the pod can stop itself.

**When a work session ends:** revoke or rotate the RunPod key and HF token if they were shared with anyone or
any tool, and terminate pods you no longer need. A stopped pod still bills for its volume.

## Pods

```bash
POD_NAME=ad-v2 python tools/rp.py create            # 2x A100-SXM4-80GB, 200 GB volume, Jupyter on 8888
python tools/rp.py list                              # balance + pods
python tools/rp.py status <pod-id>
python tools/rp.py stop <pod-id>
```

- **Cost:** 2× A100 SXM was $4.38/hr per pod (Nov 2026). Two pods ran at $8.83/hr. Check
  `currentSpendPerHr` with a GraphQL query.
- **Resuming a stopped pod can fail** ("not enough GPUs on host"). Create a new one instead; the volume is
  per pod.
- **Volume quota is 200 GB.** Writes fail with "Disk quota exceeded" even though `df` shows terabytes free
  (it is a network filesystem). Llama-3.1-70B alone is 132 GB, so delete unneeded models first
  (`rm -rf /workspace/hf/hub/models--...`).
- **Setup:** clone the repo to `/workspace/ad_v2` and run `bash v2/setup_pod.sh` (vLLM 0.11.0,
  transformers 4.57.3, datasets < 4.5, hf_transfer; model downloads). Install `peft` for StrongREJECT.

## Talking to a pod (no SSH needed)

`tools/jc.py` drives the pod's Jupyter server through `https://<pod-id>-8888.proxy.runpod.net`:

```bash
export POD_URL=https://<pod-id>-8888.proxy.runpod.net
python -c "import sys; sys.path.insert(0,'tools'); import jc; jc.login(); print(jc.sh('nvidia-smi', 60))"
python -c "...; jc.upload('local.sh', '/workspace/x.sh')"                    # small files (base64 through the kernel)
python -c "...; jc.download2('/workspace/results_v2/f.jsonl', 'f.jsonl')"   # large files: gzip on pod + /files endpoint
```

After every download, check the first row's `model`/`rid`. An earlier version of `download2` reused a temp
name and silently returned stale files. It now uses a unique name.

## Running long jobs

- **Launch detached:** `python tools/launch.py <script.sh> <log>` on the pod. Launching with `nohup … &` from
  the Jupyter shell was killed with that shell twice.
- **Never `pkill -f <pattern>` from the same shell command.** The pattern matches the shell itself and
  kills it.
- **Queue with flags.** Each script ends with `echo NAME_DONE`, and dependants wait on
  `until grep -q NAME_DONE log; do sleep 30; done`. The `run name cmd` helper in the scripts writes
  `.done_<name>` markers, so reruns skip finished steps. Touch a marker to skip a step deliberately.
- **GPU sharing.** vLLM claims about 90% of a GPU, so pin jobs with `CUDA_VISIBLE_DEVICES`. A 70B model needs
  both GPUs (`--tp 2` for vLLM; `device_map="auto"` for HF hooks).
- **Auto-stop.** Copy `tools/deadman.sh` to the pod and launch it detached with `STOP_AT=HH:MM` (UTC). It stops
  the pod through GraphQL even if the controller goes away. To move the time, edit it and relaunch (kill the
  old process by PID, not with `pkill -f`).

## Observed runtimes (2× A100-80GB)

| Job | Time |
|---|---|
| Main grid generation + judging, 3 models (E1) | ~2.5 h |
| Gemma-2-27B scale grid (vLLM, tp=2) | 13 min gen + 7 min judge |
| Llama-3.1-70B scale grid (vLLM, tp=2) | ~2.5 h+. The KV cache is small after 140 GB of weights, so the 32-item lists are slow |
| Held-out ablation, 600 generations, 8–14B (HF hooks) | 35–40 min |
| Held-out ablation, Gemma-2-27B (HF) | ~1.2 h |
| StrongREJECT on 3 main files (~34k windows) | ~15 min |

## Private data handling

Generated responses can contain harmful content. They stay on the pods (`/workspace/results_v2`) and in a
private zip. Before stopping or terminating pods, download everything needed. Commit only what
`v2/export_labels.py` produces.
