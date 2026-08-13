# Community Local Gateway

`tools/local_gateway.py` is a small, standard-library-only scheduler for maintaining and testing the
localhost optimization path without a deployed atrex-gateway server. It implements the public HTTP shapes
used by `agate dev` and `tools/sandbox.py`, and queues local accelerator commands FIFO.

## Start the Server

Run it from the repository root in the Python environment that contains the workload's accelerator stack:

```bash
python tools/local_gateway.py serve \
  --host 127.0.0.1 \
  --port 8000 \
  --state-dir .atrex-local-gateway
```

On an Ascend 910B1 host, source the installed CANN 8.5 environment before starting the server so the
worker inherits `ccec`, `bisheng`, and `msprof` paths. Adjust the install prefix if CANN is elsewhere:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
python -c 'import torch, torch_npu; print(torch.npu.get_device_name(0))'
python tools/local_gateway.py serve \
  --host 127.0.0.1 --port 8000 \
  --state-dir .atrex-local-gateway
```

If an existing workflow uses a hardware token instead of `local`, register it as an alias without changing
the optimizer arguments:

```bash
python tools/local_gateway.py serve --gpu-alias YOUR_GPU_TOKEN
```

The default `--workers 1` serializes commands for one local accelerator. A larger value enables
parallel command execution and should only be used when the machine and workloads can safely share
the device.

The state directory contains `jobs.db`, per-job uploaded files, stdout, and stderr. Queued jobs survive a
restart. A job that was running when the server stopped is marked failed on the next start rather than being
silently executed twice.

## Use the Existing Clients

The regular `agate` client needs no special adapter:

```bash
agate health --url http://127.0.0.1:8000
agate list hw --url http://127.0.0.1:8000

agate dev "python -c 'import torch; print(torch.cuda.get_device_capability())'" \
  --url http://127.0.0.1:8000 --gpu local
```

For Ascend, probe the same server environment through its normal job path:

```bash
agate dev "python -c 'import torch, torch_npu; print(torch.npu.get_device_name(0))'" \
  --url http://127.0.0.1:8000 --gpu local
```

Submission, queue inspection, long polling, and cancellation use the normal commands:

```bash
agate dev "sleep 30" --url http://127.0.0.1:8000 --gpu local --no-wait
agate jobs --url http://127.0.0.1:8000
agate get <job_id> --url http://127.0.0.1:8000 --wait
agate cancel <job_id> --url http://127.0.0.1:8000
```

The optimizer continues to use `tools/sandbox.py`, so correctness, performance, and profiler commands have
the same packaging and artifact-return behavior as a remote gateway:

```bash
python orchestrator/optimize.py \
  --op-dir /path/to/operator \
  --platform H20 --framework Triton \
  --sandbox-hardware local \
  --sandbox-url http://127.0.0.1:8000
```

The local gateway supplies transport, detection, and profiling; it does not retrofit NPU support into
an evaluator. Only when `--op-dir` already contains an NPU/`torch_npu`-ready evaluator and operator
with AscendC multi-source support, select AscendC and provide the canonical architecture explicitly
when automatic probing cannot observe the server runtime:

```bash
python orchestrator/optimize.py \
  --op-dir /path/to/npu-ready/operator \
  --platform Ascend910B1 --framework AscendC --arch ascend910b1 \
  --sandbox-hardware local \
  --sandbox-url http://127.0.0.1:8000
```

`--framework` can be omitted to start all frameworks supported by the runtime accelerator architecture in
parallel. Their sandbox requests still enter this scheduler's FIFO queue, and their local optimizer state
uses flat framework/hardware-suffixed names such as `kernel_opt_<name>_triton_h20`. Production
campaigns use a separate path ending in `_production`.

`--sandbox-hardware local` selects the gateway backend independently of the logical `--platform` value.
The optimizer does not compare platform and inventory names because a gateway may expose an alias or a
desensitized accelerator description. It uses `torch`/`torch_npu` and then `npu-smi` for runtime
architecture probing. `--arch ascend910b1` is the explicit fallback; it overrides detection but does
not select a physical device.

## Compatibility Surface

The community scheduler implements:

- `GET /healthz`
- `GET /v1/env` and `GET /v1/env/local`
- `POST /v1/jobs/eval` for native Atrex-Bench evaluation; the current upstream evaluator contract
  remains CPU/CUDA-oriented and is not made NPU-ready by this route
- `POST /v1/jobs/profile` for supported `ncu`/`rocprofv3`/`msprof` profiling requests
- `POST /v1/jobs/dev`, including uploaded text files, environment variables, timeouts, and idempotency keys
- `GET /v1/jobs` with the standard kind/user/status/limit filters
- `GET /v1/jobs/<job_id>`, including `wait=true&timeout=<seconds>` long polling
- `POST /v1/jobs/<job_id>/cancel`
- legacy `GET /v1/evals/<job_id>` polling compatibility

`tools/sandbox.py` prefers typed `eval` and `profile` requests when the workspace fits their source
contract, and falls back to a self-contained `dev` request for SOL, aggregate, custom-input, or otherwise
unrepresentable commands. `disassemble` remains unsupported and returns a structured HTTP 501
`kind_not_supported` response.

Given an NPU-ready `profile_driver.py`, an Ascend profile can be submitted through the same interface:

```bash
python tools/sandbox.py --hardware local --url http://127.0.0.1:8000 \
  --kind profile --profiler msprof --include-raw-profile --sync profiles/v1 -- \
  bash tools/profile_ascend.sh profile_driver.py --output-dir profiles/v1
```

The wrapper invokes `msprof op` and retains the raw CANN output. Structured metric parsing is currently
reported as unavailable because CANN schemas vary by version; do not fabricate metrics or reinterpret
NCU/rocprof counters as Ascend fields. A missing `msprof` executable is a hard, visible profile failure.

## Security Boundary

This server is not a container or privilege boundary. Uploaded files and commands execute as the account
running the server, and job payloads and output remain in the state directory. The server rejects a
non-loopback bind unless `--allow-remote` is supplied, but that flag does not add authentication or
isolation. Use trusted inputs and keep the default loopback bind.
