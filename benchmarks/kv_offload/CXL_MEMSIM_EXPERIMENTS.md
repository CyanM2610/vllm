# CXLMemSim KV Offloading Experiments

This integration uses an externally managed CXLMemSim process as a vLLM
secondary KV-cache tier. It keeps the existing `cxl_numa` tier as a baseline
and adds a `cxl_memsim` tier for experiments that require explicit CXL latency,
directional bandwidth contention, and logical 64-byte transaction accounting.

## Build

Use the `cxl` Conda environment. A separate build directory avoids the stale
source path stored in older CMake caches:

```bash
source /home/zjhuang/miniforge3/etc/profile.d/conda.sh
conda activate cxl

cmake -S /home/zjhuang/cxl_offloading/CXLMemSim \
  -B /home/zjhuang/cxl_offloading/CXLMemSim/build-vllm \
  -DCMAKE_CXX_FLAGS='-isystem /home/zjhuang/miniforge3/envs/cxl/include -I/home/zjhuang/miniforge3/envs/cxl/targets/x86_64-linux/include -isystem /home/zjhuang/cxl_offloading/CXLMemSim/build/deps/libbpf/usr/include'
cmake --build /home/zjhuang/cxl_offloading/CXLMemSim/build-vllm -j2
ctest --test-dir /home/zjhuang/cxl_offloading/CXLMemSim/build-vllm \
  --output-on-failure
```

The vLLM process loads
`build-vllm/libcxlmemsim_client.so` through the configured absolute path; the
library does not need to be installed system-wide.

## Start the simulator

Start the server before vLLM and keep it running for the complete sample:

```bash
/home/zjhuang/cxl_offloading/CXLMemSim/build-vllm/cxlmemsim_server \
  --comm-mode=bulk-shm \
  --bulk-shm-name=/cxlmemsim_bulk \
  --capacity=1024 \
  --default_latency=100 \
  --bulk-read-bandwidth=25 \
  --bulk-write-bandwidth=25
```

`--capacity` is the total POSIX mapping size in MiB; the usable capacity is a
small header shorter. The configured vLLM range must fit the usable capacity.
The server owns and removes `/cxlmemsim_bulk` on shutdown. Its data object,
`/cxlmemsim_shared`, follows `SharedMemoryManager` reuse semantics and can
retain bytes between restarts.

For controlled host-side placement, start the simulator with `numactl`, for
example `numactl --cpunodebind=1 --membind=1` before the server command. Record
the actual topology first; node numbering is machine-specific. Modeled CXL
latency and bandwidth are independent of NUMA placement, while the separately
reported host-copy time is not.

## Start vLLM

The following example selects physical GPU 1 and avoids reserving 90% of its
memory. Adjust the model path and capacities for the host:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve \
  /home/zjhuang/models/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 --seed 20260718 \
  --max-model-len 16384 \
  --block-size 16 \
  --gpu-memory-utilization 0.5 \
  --kv-cache-memory-bytes 2147483648 \
  --enable-prefix-caching \
  --generation-config vllm \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 536870912,
      "block_size": 64,
      "eviction_policy": "lru",
      "offload_prompt_only": true,
      "secondary_tiers": [{
        "type": "cxl_memsim",
        "client_library": "/home/zjhuang/cxl_offloading/CXLMemSim/build-vllm/libcxlmemsim_client.so",
        "control_shm_name": "/cxlmemsim_bulk",
        "cxl_bytes_to_use": 805306368,
        "cxl_offset_bytes": 0,
        "n_load_threads": 4,
        "n_store_threads": 2,
        "request_timeout_ms": 30000
      }]
    }
  }'
```

Startup is fail-closed: vLLM rejects a missing client library, absent or
incompatible control mapping, stopped server, zero capacity, and a configured
range beyond the published data mapping. A stable companion ownership lock and
server generation reject duplicate live servers and stale control rings after
`SIGKILL`; atomic slot-owner tokens let the server reclaim reserved and active
requests from dead or zombie clients. Assign disjoint `cxl_offset_bytes` ranges
when multiple vLLM processes share one simulator.

## Native acceptance test

With the server running, verify both the C ABI and the vLLM tier manager:

```bash
VLLM_TEST_CXLMEMSIM_LIBRARY=/home/zjhuang/cxl_offloading/CXLMemSim/build-vllm/libcxlmemsim_client.so \
VLLM_TEST_CXLMEMSIM_CONTROL=/cxlmemsim_bulk \
python -m pytest \
  tests/v1/kv_offload/tiering/test_cxl_memsim_integration.py -q
```

The test writes and reads an unaligned 257-byte range and requires five
logical cache lines, then performs a byte-exact STORE/LOAD through
`CxlMemSimSecondaryTierManager`.

## Timing semantics

For a request touching `N` cache lines, CXLMemSim models:

```text
wire_bytes = N * 64
serialization_ns = ceil(wire_bytes / bandwidth_GBps)
service_start_ns = max(submission_ns, next_direction_free_ns)
completion_ns = service_start_ns + serialization_ns + base_latency_ns
```

Reads and writes have independent bandwidth timelines. The client performs one
bulk host copy but the server counts every logical line; the bulk IPC operation
is not treated as one CXL transaction. vLLM reports wall-clock, host-copy,
modeled, serialization, byte, and cache-line metrics separately.

The simulator does not reproduce CPU cache coherence or physical CXL protocol
packets. Use it for controlled timing/contention studies, and retain
`cxl_numa` measurements as the host-memory baseline.
