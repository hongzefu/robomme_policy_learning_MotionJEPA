#!/usr/bin/env bash
# 仅验证参数绑定；不执行原模板main、guard、tmux、uv或任何Python/模型入口。
set -euo pipefail
draft=${1:?须指定草案}
wrapper=${2:?须指定原run模板}
judge=${3:?须指定原judge模板}
wrapper_sha=45b4cb8d7b04d3a177b5550f781b1ff97cd24aecd70d0a8647e7757ddf76d42f
judge_sha=17c0d6deea3ab3cd132bf406c968097ff9e7b455c9f22f5e3614c8641d7aafce
file_sha() { local answer; answer=$(sha256sum -- "$1"); printf '%s\n' "${answer%% *}"; }
test "$(file_sha "$wrapper")" = "$wrapper_sha"
test "$(file_sha "$judge")" = "$judge_sha"
scratch=$(mktemp -d /tmp/timing20-runtime-bindings.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
awk '/^write_bound_runtime\(\) \{/ {active=1} active {print} active && /^}$/ {exit}' "$draft" > "$scratch/generator.sh"
test -s "$scratch/generator.sh"
bash -n "$scratch/generator.sh"
source "$scratch/generator.sh"
common="$scratch/common wrapper.sh"
cp -- "$wrapper" "$common"
test "$(file_sha "$common")" = "$wrapper_sha"
test "$(head -n 1 "$common")" = '#!/usr/bin/env bash'
cat > "$scratch/argument-template.stub" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq "$EXPECTED_ARGC" || exit 51
actual=$(sha256sum -- "$0");actual=${actual%% *}
test "$1" = "$actual" || exit 52
printf '%s\0' "$@" > "$BINDINGS_OUT"
SH
stub="$scratch/argument-template.stub"
stub_sha=$(file_sha "$stub")
main=/scratch/hongze/robomme_policy_learning_MotionJEPA
store="$main/v1-store"
head=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
tag=test_20260926
first_runtime='' first_argc=0

check_case() {
  local name=$1 kind=$2 template=$3 template_sha=$4
  shift 4
  local work="$scratch/$name" runtime real_copy runtime_sha expected_argc
  mkdir "$work"
  runtime="$work/runtime with space.command"
  real_copy="$work/real-template-not-executed.command"
  expected_argc=$(($# + 1))
  write_bound_runtime "$stub" "$stub_sha" "$runtime" "$@"
  runtime_sha=$(file_sha "$runtime")
  (
    cd "$work"
    EXPECTED_ARGC="$expected_argc" BINDINGS_OUT="$work/actual.args" bash "$runtime" "$runtime_sha"
  )
  printf '%s\0' "$runtime_sha" "$@" > "$work/expected.args"
  cmp "$work/expected.args" "$work/actual.args"
  test ! -e "$work/SHOULD_NOT_RUN"
  write_bound_runtime "$template" "$template_sha" "$real_copy" "$@"
  [[ "$(head -n 1 "$real_copy")" == 'set -- "$1"'* ]]
  tail -n +2 "$real_copy" > "$work/template-tail"
  cmp "$template" "$work/template-tail"
  bash -n "$real_copy"
  if [ -z "$first_runtime" ]; then first_runtime=$runtime;first_argc=$expected_argc;fi
  printf '%s\t%s\t%s\t%s\tPASS\n' "$name" "$kind" "$#" "$expected_argc"
}

printf 'case\tkind\tfixed_args\tfinal_args\tresult\n'
for group in full count; do
  if [ "$group" = full ]; then
    gpus=0,1,2,3 lib="$store/datasets/16task-pub-1600ep" assets="$store/train-assets/mme_vla_suite"
    norm=f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5
  else
    gpus=4,5,6,7 lib="$store/datasets/4task-counting-pub-400ep" assets="$store/train-assets/mme_vla_suite/4task-counting-pub-400ep"
    norm=a77075cd024dcb1f0e82de6702332e5005b1ef926b485535ed0de0187e9a0ec9
  fi
  for mode in off on; do
    run="timing20-$group-$mode-0926-$tag";session="orig80k-$run"
    check_case "$group-$mode" run "$wrapper" "$wrapper_sha" \
      "$head" "$run" "$mode" "$gpus" "$lib" "$assets" "$norm" "$session" "$store/logs/$session.wrapper.log"
  done
  session="orig80k-timing20-$group-judge-$tag"
  check_case "$group-judge" judge "$judge" "$judge_sha" \
    "$common" "$wrapper_sha" "$head" "timing20-$group-off-0926-$tag" "timing20-$group-on-0926-$tag" \
    "$store/bench/future-$tag/$group.json" "$session" "$store/logs/$session.wrapper.log"
done

check_case special_chars run "$wrapper" "$wrapper_sha" \
  "$head" 'run with space' off 0,1,2,3 'single quote '\''' 'backslash\literal' \
  '$(touch SHOULD_NOT_RUN)' '`touch SHOULD_NOT_RUN`' $'literal\tand\nnewline'

set +e
EXPECTED_ARGC="$first_argc" BINDINGS_OUT="$scratch/wrong-sha.args" bash "$first_runtime" "$stub_sha"
wrong=$?
set -e
test "$wrong" = 52
test ! -e "$scratch/wrong-sha.args"
printf 'runtime_sha_not_template_sha\tnegative\t9\t10\tPASS\n'

before=$(file_sha "$first_runtime")
if write_bound_runtime "$stub" "$stub_sha" "$first_runtime" ignored > "$scratch/reuse.stdout" 2> "$scratch/reuse.stderr"; then exit 1; fi
test "$(file_sha "$first_runtime")" = "$before"
printf 'existing_runtime_rejected\tnegative\t1\t2\tPASS\n'
if write_bound_runtime "$stub" "$(printf '%064d' 0)" "$scratch/wrong-template.command" ignored; then exit 1; fi
test ! -e "$scratch/wrong-template.command"
printf 'wrong_template_sha_rejected\tnegative\t1\t2\tPASS\n'

(
  set -- runtime-sha "$common" "$wrapper_sha" "$head" off-run on-run out-json session log
  printf '%s\0' "$@" > "$scratch/before-source.args"
  source "$common"
  printf '%s\0' "$@" > "$scratch/after-source.args"
  test "$#" = 9
  cmp "$scratch/before-source.args" "$scratch/after-source.args"
)
test "$(file_sha "$common")" = "$wrapper_sha"
printf 'common_source_preserves_judge_argv\tsource\t8\t9\tPASS\n'
test "$(file_sha "$wrapper")" = "$wrapper_sha"
test "$(file_sha "$judge")" = "$judge_sha"
