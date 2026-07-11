#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL="Qwen/Qwen2.5-7B-Instruct"
BASE_URL="http://127.0.0.1:8000"
PORT=8000
LOCAL_NODE=0
REMOTE_NODE=1
GPU_PCI_ADDRESS="0000:17:00.0"
PROFILE="auto"
PHASES="preflight,microbench,correctness,proof,sensitivity,online,analyze"
REPETITIONS=5
SENSITIVITY_REPETITIONS=3
ONLINE_REPETITIONS=3
OUTPUT_ROOT="${REPO_ROOT}/results/cxl_numa"
SEED=20260710
DRY_RUN=0
SERVER_PID=""
SERVER_PROCESS_GROUP=0

usage() {
  cat <<'EOF'
Usage: bash benchmarks/kv_offload/run_cxl_numa_matrix.sh [options]

  --model NAME
  --base-url URL
  --port PORT
  --local-node NODE
  --remote-node NODE
  --gpu-pci-address DOMAIN:BUS:DEVICE.FUNCTION
  --profile auto|standard|low-memory
  --phases CSV
  --repetitions N
  --sensitivity-repetitions N
  --online-repetitions N
  --output-root PATH
  --seed N
  --dry-run
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --base-url) BASE_URL=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --local-node) LOCAL_NODE=$2; shift 2 ;;
    --remote-node) REMOTE_NODE=$2; shift 2 ;;
    --gpu-pci-address) GPU_PCI_ADDRESS=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --phases) PHASES=$2; shift 2 ;;
    --repetitions) REPETITIONS=$2; shift 2 ;;
    --sensitivity-repetitions) SENSITIVITY_REPETITIONS=$2; shift 2 ;;
    --online-repetitions) ONLINE_REPETITIONS=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  auto|standard|low-memory) ;;
  *) echo "invalid profile: ${PROFILE}" >&2; exit 2 ;;
esac

has_phase() {
  case ",${PHASES}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

select_profile() {
  if [[ "$PROFILE" == "auto" ]]; then
    local meminfo="/sys/devices/system/node/node${LOCAL_NODE}/meminfo"
    local free_kib=0
    if [[ -r "$meminfo" ]]; then
      free_kib=$(awk '/MemFree:/ {print $4; exit}' "$meminfo")
    fi
    if ((free_kib >= 8 * 1024 * 1024)); then
      PROFILE="standard"
    else
      PROFILE="low-memory"
    fi
  fi

  if [[ "$PROFILE" == "standard" ]]; then
    HBM_BYTES=2147483648
    CPU_PRIMARY_BYTES=1073741824
    SECONDARY_BYTES=4294967296
    CPU_ONLY_BYTES=4294967296
    MAIN_CHURN=6
    ONLINE_WORKING_SET=8
    PREFIX_SWEEP=(2048:20 4096:10 8192:6 12288:4)
  else
    HBM_BYTES=1073741824
    CPU_PRIMARY_BYTES=536870912
    SECONDARY_BYTES=2147483648
    CPU_ONLY_BYTES=2147483648
    MAIN_CHURN=3
    ONLINE_WORKING_SET=4
    PREFIX_SWEEP=(2048:10 4096:6 8192:3 12288:2)
  fi
}

expected_tier() {
  case "$1" in
    no_offload) echo cold ;;
    cpu_only) echo cpu ;;
    local_secondary|remote_secondary) echo cxl ;;
    *) return 2 ;;
  esac
}

secondary_node() {
  case "$1" in
    local_secondary) echo "$LOCAL_NODE" ;;
    remote_secondary) echo "$REMOTE_NODE" ;;
    *) echo "" ;;
  esac
}

build_kv_config() {
  local mode=$1 block_size=$2 load_threads=$3 store_threads=$4 capacity=$5
  if [[ "$mode" == "cpu_only" ]]; then
    printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":%s,"block_size":%s,"eviction_policy":"lru","offload_prompt_only":true,"secondary_tiers":[]}}' \
      "$CPU_ONLY_BYTES" "$block_size"
    return
  fi
  local node
  node=$(secondary_node "$mode")
  printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":%s,"block_size":%s,"eviction_policy":"lru","offload_prompt_only":true,"secondary_tiers":[{"type":"cxl_numa","numa_node":%s,"numa_bytes_to_use":%s,"n_load_threads":%s,"n_store_threads":%s,"prefault":true,"verify_placement":true}]}}' \
    "$CPU_PRIMARY_BYTES" "$block_size" "$node" "$capacity" \
    "$load_threads" "$store_threads"
}

render_command() {
  printf '%q ' "$@"
  printf '\n'
}

build_server_command() {
  local mode=$1 block_size=$2 load_threads=$3 store_threads=$4 capacity=$5
  SERVER_COMMAND=(
    vllm serve "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16 --seed "$SEED"
    --max-model-len 16384
    --block-size 16
    --kv-cache-memory-bytes "$HBM_BYTES"
    --enable-prefix-caching
    --generation-config vllm
    --max-num-seqs 16
    --max-num-batched-tokens 16384
    --numa-bind --numa-bind-nodes "$LOCAL_NODE"
  )
  if [[ "$mode" != "no_offload" ]]; then
    SERVER_COMMAND+=(
      --kv-transfer-config
      "$(build_kv_config "$mode" "$block_size" "$load_threads" "$store_threads" "$capacity")"
    )
  fi
}

capture_process_state() {
  local output_dir=$1
  git -C "$REPO_ROOT" rev-parse HEAD >"${output_dir}/git_commit.txt"
  nvidia-smi >"${output_dir}/nvidia_smi.txt" 2>&1 || true
  nvidia-smi topo -m >"${output_dir}/nvidia_topology.txt" 2>&1 || true
  numactl --hardware >"${output_dir}/numactl_hardware.txt" 2>&1 || true
  {
    echo "server_pid=${SERVER_PID}"
    command -v pstree >/dev/null && pstree -ap "$SERVER_PID" || true
    ps -ef | grep -E '[v]llm|[E]ngineCore' || true
  } >"${output_dir}/processes.txt"

  local pids=("$SERVER_PID")
  if command -v pgrep >/dev/null; then
    while IFS= read -r child; do
      [[ -n "$child" ]] && pids+=("$child")
    done < <(pgrep -P "$SERVER_PID" || true)
  fi
  for pid in "${pids[@]}"; do
    if [[ -r "/proc/${pid}/numa_maps" ]]; then
      cp "/proc/${pid}/numa_maps" "${output_dir}/numa_maps_${pid}.txt"
    fi
    if command -v numastat >/dev/null; then
      numastat -p "$pid" >"${output_dir}/numastat_${pid}.txt" 2>&1 || true
    fi
  done
}

start_server() {
  local mode=$1 output_dir=$2 block_size=$3 load_threads=$4 store_threads=$5 capacity=$6
  mkdir -p "$output_dir"
  build_server_command "$mode" "$block_size" "$load_threads" "$store_threads" "$capacity"
  {
    printf 'CUDA_VISIBLE_DEVICES=0 '
    render_command "${SERVER_COMMAND[@]}"
  } >"${output_dir}/server_command.txt"
  if ((DRY_RUN)); then
    cat "${output_dir}/server_command.txt"
    return 0
  fi

  if command -v setsid >/dev/null; then
    CUDA_VISIBLE_DEVICES=0 setsid "${SERVER_COMMAND[@]}" \
      >"${output_dir}/server.log" 2>&1 &
    SERVER_PROCESS_GROUP=1
  else
    CUDA_VISIBLE_DEVICES=0 "${SERVER_COMMAND[@]}" \
      >"${output_dir}/server.log" 2>&1 &
    SERVER_PROCESS_GROUP=0
  fi
  SERVER_PID=$!
  local attempt
  for ((attempt = 0; attempt < 600; attempt++)); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      sleep 3
      capture_process_state "$output_dir"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited during startup" >&2
      return 1
    fi
    sleep 1
  done
  echo "server health check timed out" >&2
  return 1
}

server_alive() {
  if ((SERVER_PROCESS_GROUP)); then
    kill -0 -- "-${SERVER_PID}" 2>/dev/null
  else
    kill -0 "$SERVER_PID" 2>/dev/null
  fi
}

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if server_alive; then
    if ((SERVER_PROCESS_GROUP)); then
      kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    else
      kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    local attempt
    for ((attempt = 0; attempt < 60; attempt++)); do
      server_alive || break
      sleep 1
    done
    if server_alive; then
      if ((SERVER_PROCESS_GROUP)); then
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
      else
        kill -KILL "$SERVER_PID" 2>/dev/null || true
      fi
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PROCESS_GROUP=0
  local health_attempt
  for ((health_attempt = 0; health_attempt < 60; health_attempt++)); do
    curl -fsS "${BASE_URL}/health" >/dev/null 2>&1 || break
    sleep 1
  done
}
trap stop_server EXIT INT TERM

run_preflight() {
  local dir="${OUTPUT_ROOT}/preflight"
  mkdir -p "$dir"
  git -C "$REPO_ROOT" rev-parse HEAD >"${dir}/git_commit.txt"
  python --version >"${dir}/python_version.txt" 2>&1
  python -m pip freeze >"${dir}/pip_freeze.txt" 2>&1 || true
  nvidia-smi >"${dir}/nvidia_smi.txt" 2>&1 || true
  nvidia-smi topo -m >"${dir}/nvidia_topology.txt" 2>&1 || true
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv \
    >"${dir}/nvidia_state.csv" 2>&1 || true
  numactl --hardware >"${dir}/numactl_hardware.txt" 2>&1 || true
  lscpu >"${dir}/lscpu.txt" 2>&1 || true
  free -h >"${dir}/free.txt" 2>&1 || true
  swapon --show >"${dir}/swap.txt" 2>&1 || true
  cp /proc/self/status "${dir}/proc_status.txt"
  cp /proc/sys/kernel/numa_balancing "${dir}/numa_balancing.txt" 2>/dev/null || true
  cp /sys/kernel/mm/transparent_hugepage/enabled \
    "${dir}/thp_enabled.txt" 2>/dev/null || true
  local gpu_numa_path="/sys/bus/pci/devices/${GPU_PCI_ADDRESS}/numa_node"
  if [[ ! -r "$gpu_numa_path" ]]; then
    echo "cannot read ${gpu_numa_path}" | tee "${dir}/preflight_error.txt" >&2
    return 1
  fi
  local actual_gpu_node
  actual_gpu_node=$(<"$gpu_numa_path")
  printf '%s\n' "$actual_gpu_node" >"${dir}/gpu0_numa_node.txt"
  if [[ "$actual_gpu_node" != "$LOCAL_NODE" ]]; then
    echo "GPU NUMA node ${actual_gpu_node} != local node ${LOCAL_NODE}" \
      | tee "${dir}/preflight_error.txt" >&2
    return 1
  fi
  if [[ -x /sbin/ldconfig ]]; then
    /sbin/ldconfig -p >"${dir}/ldconfig.txt" 2>&1 || true
  fi
  printf 'profile=%s\nlocal_node=%s\nremote_node=%s\n' \
    "$PROFILE" "$LOCAL_NODE" "$REMOTE_NODE" >"${dir}/selected_profile.txt"
}

run_timed() {
  local timeout_value=$1 log_file=$2
  shift 2
  if ((DRY_RUN)); then
    render_command timeout "$timeout_value" "$@"
    return 0
  fi
  set +e
  timeout "$timeout_value" "$@" 2>&1 | tee "$log_file"
  local status
  status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" >"${log_file}.exit_code"
  return 0
}

run_workload() {
  local scenario=$1 category=$2 mode=$3 output_dir=$4 run_id=$5 block_size=$6
  local load_threads=$7 store_threads=$8 capacity=$9
  shift 9
  local prompt_length=$1 churn=$2 concurrency=$3 working_set=$4 num_requests=$5
  local tier node
  tier=$(expected_tier "$mode")
  node=$(secondary_node "$mode")
  local command=(
    python "${SCRIPT_DIR}/cxl_numa_workload.py"
    --scenario "$scenario" --experiment-category "$category"
    --mode "$mode" --expected-tier "$tier"
    --profile "$PROFILE" --run-id "$run_id"
    --base-url "$BASE_URL" --model "$MODEL"
    --prompt-length "$prompt_length" --churn-prompts "$churn"
    --working-set-size "$working_set" --num-requests "$num_requests"
    --concurrency "$concurrency" --output-tokens 32 --seed "$SEED"
    --connector-block-size "$block_size"
    --n-load-threads "$load_threads" --n-store-threads "$store_threads"
    --output "${output_dir}/result.json"
  )
  if [[ "$mode" == "no_offload" || "$mode" == "cpu_only" ]]; then
    command+=(--cxl-capacity-bytes 0)
  else
    command+=(--cxl-capacity-bytes "$capacity")
  fi
  [[ -n "$node" ]] && command+=(--expected-numa-node "$node")
  local timeout_value=45m
  [[ "$scenario" == "online" ]] && timeout_value=60m
  run_timed "$timeout_value" "${output_dir}/workload.log" "${command[@]}"
}

run_configuration() {
  local scenario=$1 mode=$2 category=$3 run_id=$4 block_size=$5
  local load_threads=$6 store_threads=$7 capacity=$8 prompt_length=$9
  shift 9
  local churn=$1 concurrency=$2 working_set=$3 num_requests=$4
  local output_dir="${OUTPUT_ROOT}/${category}/${run_id}"
  echo "running ${run_id}"
  if ! start_server "$mode" "$output_dir" "$block_size" "$load_threads" "$store_threads" "$capacity"; then
    echo "server startup failed for ${run_id}" | tee "${output_dir}/startup_error.txt"
    stop_server
    return 0
  fi
  run_workload "$scenario" "$category" "$mode" "$output_dir" "$run_id" \
    "$block_size" \
    "$load_threads" "$store_threads" "$capacity" "$prompt_length" \
    "$churn" "$concurrency" "$working_set" "$num_requests"
  stop_server
}

run_microbench() {
  mkdir -p "${OUTPUT_ROOT}/microbench"
  for blocks in 1 4 16 64; do
    run_timed 30m "${OUTPUT_ROOT}/microbench/blocks_${blocks}.log" \
      python "${SCRIPT_DIR}/cxl_numa_microbench.py" \
      --local-node "$LOCAL_NODE" --remote-node "$REMOTE_NODE" \
      --block-size-bytes 3670016 --blocks-per-job "$blocks" \
      --thread-counts 1 2 4 8 --warmup 5 --repetitions 30 \
      --output "${OUTPUT_ROOT}/microbench/blocks_${blocks}.json"
  done
}

run_correctness() {
  mkdir -p "${OUTPUT_ROOT}/correctness"
  for rep in 1 2 3; do
    run_timed 30m "${OUTPUT_ROOT}/correctness/rep_${rep}.log" \
      env VLLM_TEST_CXL_NUMA_NODE="$REMOTE_NODE" \
      python -m pytest --optional -q \
      tests/v1/kv_offload/tiering/test_cxl_numa_integration.py
  done
}

run_proof_matrix() {
  local rep mode
  local modes=(no_offload cpu_only local_secondary remote_secondary)
  local -a shuffled
  for ((rep = 1; rep <= REPETITIONS; rep++)); do
    mapfile -t shuffled < <(
      python -c 'import random,sys; x=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(x); print(*x, sep="\n")' \
        "$((SEED + rep))" "${modes[@]}"
    )
    for mode in "${shuffled[@]}"; do
      run_configuration proof "$mode" proof \
        "${PROFILE}_${mode}_rep${rep}" 64 4 2 "$SECONDARY_BYTES" \
        8192 "$MAIN_CHURN" 1 8 1
    done
  done
}

run_sensitivity_matrix() {
  local rep block pair load_threads store_threads prompt churn gib capacity
  for block in 16 64 256; do
    for ((rep = 1; rep <= SENSITIVITY_REPETITIONS; rep++)); do
      run_configuration proof remote_secondary sensitivity \
        "block_${block}_rep${rep}" "$block" 4 2 "$SECONDARY_BYTES" \
        8192 "$MAIN_CHURN" 1 8 1
    done
  done
  for pair in 1:1 2:1 4:2 8:4; do
    IFS=: read -r load_threads store_threads <<<"$pair"
    for ((rep = 1; rep <= SENSITIVITY_REPETITIONS; rep++)); do
      run_configuration proof remote_secondary sensitivity \
        "threads_${load_threads}_${store_threads}_rep${rep}" 64 \
        "$load_threads" "$store_threads" "$SECONDARY_BYTES" \
        8192 "$MAIN_CHURN" 1 8 1
    done
  done
  for pair in "${PREFIX_SWEEP[@]}"; do
    IFS=: read -r prompt churn <<<"$pair"
    for ((rep = 1; rep <= SENSITIVITY_REPETITIONS; rep++)); do
      run_configuration proof remote_secondary sensitivity \
        "prefix_${prompt}_rep${rep}" 64 4 2 "$SECONDARY_BYTES" \
        "$prompt" "$churn" 1 8 1
    done
  done
  for gib in 2 4 8; do
    capacity=$((gib * 1024 * 1024 * 1024))
    for ((rep = 1; rep <= SENSITIVITY_REPETITIONS; rep++)); do
      run_configuration proof remote_secondary sensitivity \
        "capacity_${gib}gib_rep${rep}" 64 4 2 "$capacity" \
        8192 "$MAIN_CHURN" 1 8 1
    done
  done
}

run_online_matrix() {
  local concurrency rep mode
  local modes=(no_offload cpu_only remote_secondary)
  local -a shuffled
  for concurrency in 1 4 8 16; do
    for ((rep = 1; rep <= ONLINE_REPETITIONS; rep++)); do
      mapfile -t shuffled < <(
        python -c 'import random,sys; x=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(x); print(*x, sep="\n")' \
          "$((SEED + concurrency * 100 + rep))" "${modes[@]}"
      )
      for mode in "${shuffled[@]}"; do
        run_configuration online "$mode" online \
          "${mode}_c${concurrency}_rep${rep}" 64 4 2 "$SECONDARY_BYTES" \
          8192 "$MAIN_CHURN" "$concurrency" "$ONLINE_WORKING_SET" 200
      done
    done
  done
  for concurrency in 1 8; do
    for ((rep = 1; rep <= ONLINE_REPETITIONS; rep++)); do
      run_configuration online local_secondary online \
        "local_secondary_c${concurrency}_rep${rep}" 64 4 2 "$SECONDARY_BYTES" \
        8192 "$MAIN_CHURN" "$concurrency" "$ONLINE_WORKING_SET" 200
    done
  done
}

run_analysis() {
  local dir="${OUTPUT_ROOT}/summary"
  mkdir -p "$dir"
  if ((DRY_RUN)); then
    render_command python "${SCRIPT_DIR}/analyze_cxl_numa_results.py" \
      --input-root "$OUTPUT_ROOT" --output-json "${dir}/summary.json" \
      --output-csv "${dir}/summary.csv"
    return
  fi
  python "${SCRIPT_DIR}/analyze_cxl_numa_results.py" \
    --input-root "$OUTPUT_ROOT" \
    --output-json "${dir}/summary.json" \
    --output-csv "${dir}/summary.csv"
}

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"
select_profile
echo "profile=${PROFILE} output_root=${OUTPUT_ROOT}"

has_phase preflight && run_preflight
has_phase microbench && run_microbench
has_phase correctness && run_correctness
has_phase proof && run_proof_matrix
has_phase sensitivity && run_sensitivity_matrix
has_phase online && run_online_matrix
has_phase analyze && run_analysis

echo "experiment phases complete; results are under ${OUTPUT_ROOT}"
