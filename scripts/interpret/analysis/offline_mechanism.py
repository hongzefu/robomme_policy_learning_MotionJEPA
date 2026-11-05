"""固定成功案例的同输入注意力分析；只描述关联，不实施输入干预。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value, *, fixed=False):
    path = Path(path)
    if fixed and path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"固定选择不能覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def episode_number(key):
    return int(re.findall(r"\d+", key)[-1])


def select_cases(summary):
    """参考模型按身份排序；只用其比例选点，两模型配对同一时刻。"""
    rows = summary["rows"]
    models = sorted({r["model_id"] for r in rows})
    if len(models) != 2 or any(not Path(model).is_absolute() for model in models):
        raise ValueError("必须提供两个绝对检查点身份")
    identities = [(r["model_id"], r["episode_key"], r["env_step"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("汇总中存在重复询问身份")
    successful = {r["episode_key"] for r in summary["calibration_results"]
                  if r["kind"] == "fixed_release" and r["valid"] is True and r["task_success"] == 1}
    cases, missing = [], []
    for period in sorted({r["move_interval"] for r in rows}):
        keys = sorted({r["episode_key"] for r in rows if r["move_interval"] == period}
                      & successful, key=episode_number)
        if not keys:
            missing.append({"move_interval": period, "reason": "该周期没有有效固定放手成功局"})
            continue
        key = keys[0]
        by_model = {model: {r["env_step"]: r for r in rows if r["model_id"] == model
                           and r["episode_key"] == key and not r["pre_reach"]} for model in models}
        common = sorted(set(by_model[models[0]]) & set(by_model[models[1]]))
        reference = [by_model[models[0]][step] for step in common]
        decision = [r for r in reference if r["decision"]]
        early = [r for r in reference if r["early_visit"]]
        choices = [("决策起点附近", min(decision, key=lambda r: r["env_step"]) if decision else None),
                   ("高比例", min(decision, key=lambda r: (-r["p"], r["env_step"])) if decision else None),
                   ("低比例", min(decision, key=lambda r: (r["p"], r["env_step"])) if decision else None),
                   ("较早经过最高比例", min(early, key=lambda r: (-r["p"], r["env_step"])) if early else None)]
        points = {}
        for category, choice in choices:
            if choice is None:
                missing.append({"episode_key": key, "category": category, "reason": "没有两模型共同保存的候选"})
                continue
            step = choice["env_step"]
            point = points.setdefault(step, {"env_step": step, "categories": [], "inputs": []})
            point["categories"].append(category)
            if not point["inputs"]:
                point["inputs"] = [dict(by_model[model][step]) for model in models]
        cases.append({"episode_key": key, "move_interval": period, "reference_model": models[0],
                      "points": list(points.values()), "success_evidence": [r for r in summary["calibration_results"]
                      if r["episode_key"] == key and r["kind"] == "fixed_release"],
                      "calibration_evidence": [r for r in summary["calibration_results"] if r["episode_key"] == key]})
    if len(cases) > 3 or sum(len(c["points"]) * 2 for c in cases) > 24:
        raise ValueError("案例规模超过计划上限")
    return {"schema_version": 1, "models": models, "selection_rule": "身份字典序首模型为固定参考；共同候选按比例和最早步选点",
            "cases": cases, "missing": missing}


def select(summary_path, output):
    summary = json.loads(Path(summary_path).read_text())
    selection = select_cases(summary)
    selection["source"] = fingerprint(summary_path)
    table_path = summary["table"]["path"]
    selection["table"] = fingerprint(table_path)
    if any(selection["table"][k] != summary["table"][k] for k in ("path", "sha256")):
        raise ValueError("动作表与汇总指纹不同")
    for case in selection["cases"]:
        for point in case["points"]:
            for item in point["inputs"]:
                item["query_fingerprint"] = fingerprint(item["query_path"])
                item["cache_fingerprint"] = fingerprint(item["cache_path"])
                with np.load(item["cache_path"], allow_pickle=False) as cache:
                    indices = np.flatnonzero(cache["env_step"] == item["env_step"])
                    if len(indices) != 1 or str(cache["query_paths"][indices[0]]) != item["query_path"]:
                        raise ValueError("高度缓存与询问身份不一致")
                    # 原十份预测按原2cm、前16行判据分组，不用展开路径结果改标签。
                    table = json.loads(Path(table_path).read_text())
                    entries = table.get("episodes", table)
                    if isinstance(entries, list):
                        entries = {e["episode_key"]: e for e in entries}
                    hover = entries[case["episode_key"]]["z_hover"]
                    heights = cache["action_z"][indices[0]]
                    if heights.shape != (10, 20) or not np.isfinite(heights).all():
                        raise ValueError("原预测高度必须为有限的10×20数组")
                    item["press_predictions"] = (heights[:, :16] < hover - .02).any(axis=1).tolist()
                    if not np.isclose(np.mean(item["press_predictions"]), item["p"], atol=1e-12, rtol=0):
                        raise ValueError("原高度判据与固定汇总P不符")
    write_json(output, selection, fixed=True)
    count = sum(len(c["points"]) * 2 for c in selection["cases"])
    print(f"MECHANISM_SELECTION=PASS episodes={len(selection['cases'])} inputs={count} predictions={count * 10}")
    return selection


def original_prompt_positions(observation):
    """只读原有效文字位置及词元编号，不重新分词或解码。"""
    ids = np.asarray(observation.tokenized_prompt).reshape(-1)
    mask = np.asarray(observation.tokenized_prompt_mask).reshape(-1).astype(bool)
    return [{"position": int(position), "token_id": int(ids[position])} for position in np.flatnonzero(mask)]


def token_groups(observation, layout, frame_times, tokens_per_frame):
    """按原mem_order映射视觉历史；motion无原始时间字段时明确不臆造。"""
    p, mem = layout["P"], layout["MEM"]
    static = np.asarray(observation.static_mask).reshape(-1).astype(bool)
    motion = np.asarray(observation.motion_mask).reshape(-1).astype(bool) if observation.motion_mask is not None else np.zeros(0, bool)
    order = np.asarray(observation.mem_order).reshape(-1).astype(int) if observation.mem_order is not None else np.arange(mem)
    valid = np.concatenate([static, motion])[order]
    if len(order) != mem or len(valid) != mem:
        raise ValueError("记忆布局不符")
    groups = {name: np.zeros(p + 20, bool) for name in ("motion", "历史视觉", "当前视觉", "文字", "其余")}
    groups["motion"][:mem] = valid & (order >= len(static))
    groups["历史视觉"][:mem] = valid & (order < len(static))
    # 与HistoryPi0.embed_prefix相同，按obs.images插入顺序拼接，每张图沿用原image_masks。
    image_names = list(observation.images)
    if not image_names or layout["IMG"] % len(image_names):
        raise ValueError("当前图像布局不符")
    image_tokens = layout["IMG"] // len(image_names)
    for index, name in enumerate(image_names):
        image_valid = np.asarray(observation.image_masks[name]).reshape(-1)
        if image_valid.shape != (1,):
            raise ValueError("图像mask必须保留原单样本批量")
        start = mem + index * image_tokens
        groups["当前视觉"][start:start + image_tokens] = bool(image_valid[0])
    text_mask = np.asarray(observation.tokenized_prompt_mask).reshape(-1).astype(bool)
    groups["文字"][mem + layout["IMG"]:p] = text_mask
    # 前缀padding从不属于有效key；其余只包含有效但未归上述四类的动作key。
    valid_keys = np.zeros(p + 20, bool)
    valid_keys[:mem] = valid
    valid_keys[mem:mem + layout["IMG"]] = groups["当前视觉"][mem:mem + layout["IMG"]]
    valid_keys[mem + layout["IMG"]:p] = text_mask
    valid_keys[p:] = True
    groups["其余"] = valid_keys & ~np.logical_or.reduce(list(groups.values()))
    times = np.full(mem, -1, int)
    frame_times = np.asarray(frame_times).reshape(-1)
    for index in np.flatnonzero(groups["历史视觉"][:mem]):
        frame = order[index] // tokens_per_frame
        if frame >= len(frame_times):
            raise ValueError("有效视觉token缺少真实frame_times")
        times[index] = frame_times[frame]
    for frame in sorted(set(times[times >= 0].tolist())):
        mask = np.zeros(p + 20, bool)
        mask[:mem] = (times == frame)
        groups[f"真实帧:{frame}"] = mask
    for token in original_prompt_positions(observation):
        mask = np.zeros(p + 20, bool)
        mask[mem + layout["IMG"] + token["position"]] = True
        groups[f"文字位置:{token['position']}:词元:{token['token_id']}"] = mask
    return groups


def aggregate_attention(probabilities, groups):
    """保留层、头；平均扩散步、批量及前16个动作位置，只保存组质量。"""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 6 or values.shape[2] != 1 or not np.isfinite(values).all():
        raise ValueError("动作注意力应为扩散步×层×1×头×动作×key且有限")
    key_mean = values[..., :16, :].mean(axis=(0, 2, 4))
    result = {}
    for name, mask in groups.items():
        if len(mask) != values.shape[-1]:
            raise ValueError("token分组与注意力key长度不符")
        count = int(mask.sum())
        mass = key_mean[..., mask].sum(-1)
        result[name] = {"tokens": count, "mass": mass.tolist(),
                        "per_token": (mass / count).tolist() if count else None}
    return result


def analyze_input(policy, adapter, item, entry):
    import jax
    import jax.numpy as jnp
    from scripts.interpret.offline_query import replay_query, unpack_observation
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    for key in ("query_fingerprint", "cache_fingerprint"):
        if fingerprint(item[key]["path"]) != item[key]:
            raise ValueError("选择后输入发生变化")
    path = item["query_path"]
    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    if str(record["model_id"].item()) != item["model_id"] or int(record["env_step"]) != item["env_step"]:
        raise ValueError("原输入身份不符")
    if len(record["noise"]) != 10 or len(item["press_predictions"]) != 10:
        raise ValueError("必须复用原十噪声")
    replay = replay_query(policy, path)
    result = {"input": item, "replay": replay, "predictions": []}
    # 原路径失败仍保存证据，但不能伪造注意力分析通过。
    if replay["status"] != "PASS":
        return result
    obs = unpack_observation(record, HistAugObservation)
    layout = adapter.layout(obs)
    groups = token_groups(obs, layout, record["frame_times"], int(policy.config.token_per_image) * int(policy.config.num_views))
    for name, mask in list(groups.items()):
        if name.startswith("真实帧:"):
            frame = int(name.split(":")[1])
            env_frame = frame + int(record["env_step"]) - int(record["step_idx"])
            visit = next((v["target_pass"] for v in entry["visits"] if v["in"] <= env_frame <= v["out"]), None)
            group = f"经过:{visit}" if visit is not None else "经过:区间外"
            groups[group] = groups.get(group, np.zeros_like(mask)) | mask
    kwargs = json.loads(str(record["sample_kwargs"].item()))
    if set(kwargs) - {"num_steps"}:
        raise ValueError(f"适配器不支持原采样参数：{sorted(kwargs)}")
    maximum = 0.0
    for index, noise in enumerate(record["noise"]):
        output = adapter.run(jnp.asarray(record["sample_rng"], dtype=jnp.uint32), obs,
                             noise=jnp.asarray(noise[None]), num_steps=int(kwargs.get("num_steps", 10)), save_prefix=False)
        action, attention = jax.device_get(output[:2])
        difference = float(np.max(np.abs(np.asarray(action)[0] - record["actions_normalized"][index])))
        if not np.isfinite(difference):
            raise ValueError("展开动作含非有限值")
        maximum = max(maximum, difference)
        result["predictions"].append({"noise_index": index, "press": item["press_predictions"][index],
            "attn_path_maxabs": difference, "groups": aggregate_attention(attention, groups)})
    replay.update(attention="ok", attn_path_maxabs=maximum)
    result["layout"] = layout
    result["prompt_positions"] = original_prompt_positions(obs)
    result["frame_times"] = record["frame_times"].tolist()
    result["frame_to_env_offset"] = int(record["env_step"]) - int(record["step_idx"])
    result["motion_time_note"] = "motion输入没有显式原始窗口时间字段；不从位置编码反推时间"
    return result


def analyze(selection_path, model_id, config_name, output):
    selection = json.loads(Path(selection_path).read_text())
    if model_id not in selection["models"]:
        raise ValueError("模型不在固定选择中")
    if fingerprint(selection["table"]["path"]) != selection["table"]:
        raise ValueError("动作表指纹改变")
    table = json.loads(Path(selection["table"]["path"]).read_text())
    entries = table.get("episodes", table)
    if isinstance(entries, list):
        entries = {e["episode_key"]: e for e in entries}
    report = {"model_id": model_id, "config": config_name, "selection": fingerprint(selection_path), "results": []}
    write_json(output, report)
    if not any(case["points"] for case in selection["cases"]):
        print(f"OFFLINE_REPLAY=FAIL models=1 points=0 same_path_maxabs=None tol=1e-6 "
              f"attn_path_maxabs=None attention=not_run reason=no_selected_inputs output={output}")
        return 1
    from scripts.interpret.attn_adapter import AttnAdapter, _ZeroMotion, _MV_DIR
    import sys
    sys.path.insert(0, str(_MV_DIR))
    from mv_model_adapter import build_policy
    from mme_vla_suite.training import config
    policy = build_policy(config.get_config(config_name), Path(model_id), seed=42,
                          motion_factory=lambda **kwargs: _ZeroMotion())
    adapter = AttnAdapter(policy)
    for case in selection["cases"]:
        for point in case["points"]:
            item = next(i for i in point["inputs"] if i["model_id"] == model_id)
            report["results"].append(analyze_input(policy, adapter, item, entries[case["episode_key"]]))
            write_json(output, report)
    passed = bool(report["results"]) and all(r["replay"]["status"] == "PASS" and r["replay"]["attention"] == "ok" for r in report["results"])
    same_max = max((r["replay"]["same_path_maxabs"] for r in report["results"]), default=None)
    attn_max = max((r["replay"]["attn_path_maxabs"] for r in report["results"] if r["replay"]["attn_path_maxabs"] is not None), default=None)
    print(f"OFFLINE_REPLAY={'PASS' if passed else 'FAIL'} models=1 points={len(report['results'])} "
          f"same_path_maxabs={same_max} tol=1e-6 attn_path_maxabs={attn_max} attention={'ok' if passed else 'incomplete'} output={output}")
    return 0 if passed else 1


def report(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(Path(path).read_text()) for path in paths]
    selections = [data["selection"] for data in reports]
    if not selections or any(s != selections[0] for s in selections):
        raise ValueError("两模型报告必须来自同一固定选择")
    selection = json.loads(Path(selections[0]["path"]).read_text())
    if sorted(data["model_id"] for data in reports) != selection["models"]:
        raise ValueError("报告必须完整包含固定选择的两个模型")
    for data in reports:
        expected = {(c["episode_key"], p["env_step"]) for c in selection["cases"] for p in c["points"]}
        actual = [(r["input"]["episode_key"], r["input"]["env_step"]) for r in data["results"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("分析结果有重复或缺失固定时刻")
    parts = ["<!doctype html><meta charset='utf-8'><title>同输入机制分析</title><h1>同输入机制分析</h1>",
             "<p>这些固定成功案例只描述关联，不能证明因果或替代50局总体比较。注意力路径与原路径数值差异分别报告。图需主会话目视复核。</p>",
             "<p>这里的成功指该局固定放手运行存在有效的任务成功结果，仅用于按既定规则选局。注意力实际来自同局已保存的Oracle询问输入与原十份噪声，不代表成功闭环全过程的注意力，也不代表闭环实际执行动作所用的噪声。</p>"]
    if not any(data["results"] for data in reports):
        parts.append("<p>没有固定选中的共同输入，分析未执行；未加载模型、未补跑寻找成功，不能判定机制分析通过。具体缺失原因如下。</p>")
    total, valid = 0, True
    for model_index, data in enumerate(reports):
        parts.append(f"<h2>模型{model_index + 1}</h2><pre>{html.escape(data['model_id'])}</pre>")
        selection = json.loads(Path(data["selection"]["path"]).read_text())
        if fingerprint(data["selection"]["path"]) != data["selection"]:
            raise ValueError("报告所用固定选择已改变")
        parts.append("<pre>固定参考身份与共同保存时刻、缺失及真实校准证据：" + html.escape(json.dumps(
            {"rule": selection["selection_rule"], "missing": selection["missing"],
             "cases": [{k: c[k] for k in ("episode_key", "reference_model", "calibration_evidence")} for c in selection["cases"]]}, ensure_ascii=False)) + "</pre>")
        for index, result in enumerate(data["results"]):
            total += 1
            item, replay = result["input"], result["replay"]
            valid &= replay["status"] == "PASS" and replay["attention"] == "ok"
            parts.append(f"<h3>{html.escape(item['episode_key'])}，环境步{item['env_step']}</h3><pre>{html.escape(json.dumps(replay, ensure_ascii=False))}</pre>")
            predictions = result["predictions"]
            if not predictions:
                continue
            parts.append(f"<p>原十噪声想按比例P={item['p']}；分组标签来自原预测的前16行、悬停高度减2cm判据。询问来源：<code>{html.escape(item['query_path'])}</code></p>")
            parts.append("<p>以下仅按原有效文字位置和词元编号展示注意力质量，数值对该输入既有预测、层及头取平均；单个词元可能只是子词，分组观察不能证明模型理解计数，也不能证明因果。</p>"
                         "<table><tr><th>原位置</th><th>原词元编号</th><th>全部预测质量</th><th>下压预测质量</th><th>非下压预测质量</th></tr>")
            for token in result.get("prompt_positions", []):
                group = f"文字位置:{token['position']}:词元:{token['token_id']}"
                cells = [str(token["position"]), str(token["token_id"])]
                for pressed in (None, True, False):
                    subset = [prediction for prediction in predictions if pressed is None or prediction["press"] == pressed]
                    cells.append(f"{np.mean([prediction['groups'][group]['mass'] for prediction in subset]):.8g}" if subset else "缺失")
                parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            parts.append("</table>")
            names = ["motion", "历史视觉", "当前视觉", "文字", "其余"]
            palette = {"motion": "#1f77b4", "历史视觉": "#ff7f0e", "当前视觉": "#2ca02c", "文字": "#d62728", "其余": "#9467bd"}
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            fig.subplots_adjust(top=.72, bottom=.15, wspace=.28)
            for pressed, style in [(False, "--"), (True, "-")]:
                subset = [p for p in predictions if p["press"] == pressed]
                if not subset:
                    parts.append(f"<p>{'下压' if pressed else '非下压'}预测组缺失，数量0。</p>")
                    continue
                for name in names:
                    mass = np.mean([p["groups"][name]["mass"] for p in subset], axis=0)
                    axes[0].plot(mass.mean(axis=1), style, color=palette[name], label=f"{name} {'下压' if pressed else '非下压'} n={len(subset)}")
                    per = subset[0]["groups"][name]["tokens"]
                    if per:
                        axes[1].plot(mass.mean(axis=1) / per, style, color=palette[name], label=f"{name} {'下压' if pressed else '非下压'}")
            for ax, title in zip(axes, ["总注意力质量", "每token平均注意力"]):
                ax.set(title=title, xlabel="层号（从0开始）", ylabel="注意力")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99), ncol=5, fontsize=7)
            image = f"模型{model_index + 1}_输入{index + 1}.png"
            fig.savefig(output / image, dpi=160, bbox_inches="tight")
            plt.close(fig)
            parts.append(f"<img width='100%' src='{image}'><p>逐噪声、逐层逐头、真实帧与经过分组原始聚合保存在分析JSON；不保存完整前缀矩阵。</p>")
            parts.append(f"<p>{html.escape(result['motion_time_note'])}</p>")
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            fig.subplots_adjust(top=.8, bottom=.15, wspace=.35)
            if predictions[0]["groups"]["motion"]["tokens"] == 0:
                axes[0].text(.5, .5, "无有效motion token", ha="center", va="center", transform=axes[0].transAxes)
                axes[0].set_axis_off()
            else:
                mass = np.mean([p["groups"]["motion"]["mass"] for p in predictions], axis=0)
                heat = axes[0].imshow(mass, interpolation="nearest", aspect="auto", vmin=0)
                fig.colorbar(heat, ax=axes[0], label="注意力质量")
                axes[0].set(title="motion质量（逐层逐头，十噪声均值）", xlabel="头号", ylabel="层号")
            frames = sorted((name for name in predictions[0]["groups"] if name.startswith("真实帧:")), key=lambda n: int(n.split(":")[1]))
            for pressed, style in [(False, "--"), (True, "-")]:
                subset = [p for p in predictions if p["press"] == pressed]
                if subset:
                    axes[1].plot([int(n.split(":")[1]) + result["frame_to_env_offset"] for n in frames],
                        [np.mean([p["groups"][n]["mass"] for p in subset]) for n in frames], style,
                        color=palette["历史视觉"], label=f"{'下压' if pressed else '非下压'} n={len(subset)}")
            axes[1].set(title="真实历史帧注意力质量", xlabel="原frame_times换算环境步", ylabel="质量（层头均值）")
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99), ncol=2, fontsize=8)
            image = f"模型{model_index + 1}_输入{index + 1}_帧与头.png"
            fig.savefig(output / image, dpi=160, bbox_inches="tight")
            plt.close(fig)
            parts.append(f"<img width='100%' src='{image}'>")
    parts.append("<p>输入与高度缓存、选择汇总及动作表均保存SHA256指纹；模型身份保留完整检查点路径。真实校准结果在固定选择的success_evidence中，禁止依注意力结果改选例。</p>")
    (output / "report.html").write_text("\n".join(parts))
    print(f"MECHANISM_REPORT={'PASS' if valid and total else 'FAIL'} models={len(reports)} inputs={total} visual_review=pending path={output / 'report.html'}")
    return 0 if valid and total else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--summary", required=True, type=Path)
    choose.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("analyze")
    run.add_argument("--selection", required=True, type=Path)
    run.add_argument("--model-id", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True, type=Path)
    render = commands.add_parser("report")
    render.add_argument("--analysis", required=True, nargs="+", type=Path)
    render.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "select":
        select(args.summary, args.output)
        return 0
    if args.command == "analyze":
        return analyze(args.selection, args.model_id, args.config, args.output)
    return report(args.analysis, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
