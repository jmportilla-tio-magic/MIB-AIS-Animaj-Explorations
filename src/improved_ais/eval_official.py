from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from improved_ais.checkpoint import load_improved_model, parse_checkpoint, resolve_device
from improved_ais.data.animaj_hdf5 import AnimajClip, iter_animaj_hdf5
from improved_ais.data.window import make_training_sample
from improved_ais.metrics import l1


TEST_SETS = ("held_out_algorithmic", "held_out_random", "production")


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    role: str
    controller_h5: Path
    block_keyframes_h5: Path
    protocol: str


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_specs(config: dict[str, Any]) -> dict[str, BenchmarkDataset]:
    data_cfg = config["data"]
    eval_sets = {item["name"]: item for item in config.get("eval", {}).get("sets", []) or []}
    selection_sets = data_cfg.get("selection_sets") or [
        {
            "controller_h5": data_cfg.get("val_controller_h5"),
            "block_keyframes_h5": data_cfg.get("val_block_keyframes_h5"),
        }
    ]
    selection = selection_sets[0]
    production = eval_sets.get("prod_test")
    production_role = "benchmark"
    if production is None:
        holdouts = data_cfg.get("holdout_sets", []) or []
        production = next((item for item in holdouts if item.get("name") in {"prod_test", "prod_holdout", "production"}), None)
        production_role = "holdout"
    if production is None:
        raise ValueError("config.eval.sets must include prod_test or data.holdout_sets must include prod_holdout/production")
    return {
        "held_out_algorithmic": BenchmarkDataset(
            name="held_out_algorithmic",
            role="selection",
            controller_h5=Path(selection["controller_h5"]),
            block_keyframes_h5=Path(selection["block_keyframes_h5"]),
            protocol="block_keyframes",
        ),
        "held_out_random": BenchmarkDataset(
            name="held_out_random",
            role="selection",
            controller_h5=Path(selection["controller_h5"]),
            block_keyframes_h5=Path(selection["block_keyframes_h5"]),
            protocol="random_uniform_90",
        ),
        "production": BenchmarkDataset(
            name="production",
            role=production_role,
            controller_h5=Path(production["controller_h5"]),
            block_keyframes_h5=Path(production["block_keyframes_h5"]),
            protocol="block_keyframes",
        ),
    }


def _official_npss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(pred.shape[0], -1)
    target = np.asarray(target, dtype=np.float64).reshape(target.shape[0], -1)
    pred_power = np.abs(np.fft.fft(pred, axis=0)) ** 2
    gt_power = np.abs(np.fft.fft(target, axis=0)) ** 2
    pred_power = pred_power.T
    gt_power = gt_power.T
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)
    gt_norm = gt_power / (gt_total_power[:, None] + eps)
    pred_norm = pred_power / (pred_total_power[:, None] + eps)
    cdf_gt = np.cumsum(gt_norm, axis=1)
    cdf_pred = np.cumsum(pred_norm, axis=1)
    emd = np.sum(np.abs(cdf_gt - cdf_pred), axis=1)
    return float(np.sum(emd * gt_total_power) / (np.sum(gt_total_power) + eps))


def _candidate_motions(start: int, end: int, pred: np.ndarray, min_frame: int, max_frame: int) -> list[np.ndarray]:
    motion_length = int(end - start)
    start_min = max(start - motion_length, min_frame)
    start_max = min(end + 1, max_frame - motion_length)
    return [pred[s : s + motion_length + 1] for s in range(start_min, start_max)]


def _official_shifted_distance(
    target: np.ndarray,
    pred: np.ndarray,
    animation_keyframes: np.ndarray,
    unmasked_frames: list[int],
) -> float:
    del animation_keyframes  # Upstream currently considers all frames in each segment.
    if len(unmasked_frames) < 2:
        return float("nan")
    total = 0.0
    total_frames = 0
    min_frame = int(unmasked_frames[0])
    max_frame = int(unmasked_frames[-1]) + 1
    for start, end in zip(unmasked_frames[:-1], unmasked_frames[1:]):
        start = int(start)
        end = int(end)
        gt_motion = target[start : end + 1]
        candidates = _candidate_motions(start, end, pred, min_frame=min_frame, max_frame=max_frame)
        if not candidates:
            continue
        distances = [float(np.linalg.norm(candidate - gt_motion, ord=1)) for candidate in candidates]
        best_motion = candidates[int(np.argmin(distances))]
        frames_to_consider = list(range(end - start + 1))
        total += float(np.linalg.norm(gt_motion[frames_to_consider] - best_motion[frames_to_consider], ord=1))
        total_frames += len(frames_to_consider)
    return float(total / max(1, total_frames))


def _random_uniform_unmasked(length: int, *, ratio_masked: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(length) < ratio_masked
    mask[0] = False
    mask[length - 1] = False
    return np.flatnonzero(~mask).astype(np.int64)


def _block_unmasked(clip: AnimajClip) -> np.ndarray:
    return clip.key_indices.astype(np.int64)


def _predict(model, clip: AnimajClip, unmasked: np.ndarray, device) -> np.ndarray:
    import torch

    sample = make_training_sample(clip.vectors, unmasked)
    kwargs = {
        "observed_mask": torch.from_numpy(sample.observed_mask).unsqueeze(0).to(device),
        "keypose_values": torch.from_numpy(sample.target).unsqueeze(0).to(device),
    }
    if bool(getattr(model, "uses_temporal_features", False)):
        temporal = np.concatenate([sample.phase, sample.segment_len, sample.dist_prev, sample.dist_next], axis=-1)
        kwargs["temporal_features"] = torch.from_numpy(temporal).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(
            torch.from_numpy(sample.input_seq).unsqueeze(0).to(device),
            torch.from_numpy(sample.prev_pose).unsqueeze(0).to(device),
            torch.from_numpy(sample.next_pose).unsqueeze(0).to(device),
            **kwargs,
        )
    return out["pred"].squeeze(0).detach().cpu().numpy()


def _clip_row(dataset: str, model_name: str, clip: AnimajClip, pred: np.ndarray, unmasked: np.ndarray) -> dict[str, Any]:
    unmasked = np.asarray(sorted(set(int(i) for i in unmasked)), dtype=np.int64)
    missing_mask = np.ones(len(clip.vectors), dtype=bool)
    missing_mask[unmasked] = False
    missing_frames = int(np.sum(missing_mask))
    return {
        "test_set": dataset,
        "model": model_name,
        "clip_id": clip.clip_id,
        "episode_id": clip.episode_id or "",
        "scene_id": clip.scene_id or "",
        "frames": int(len(clip.vectors)),
        "unmasked_frames": int(len(unmasked)),
        "missing_frames": missing_frames,
        "mask_ratio": float(missing_frames / max(1, len(clip.vectors))),
        "shifted_distance": _official_shifted_distance(
            target=clip.vectors,
            pred=pred,
            animation_keyframes=clip.animation_keyframes,
            unmasked_frames=[int(i) for i in unmasked],
        ),
        "npss": _official_npss(pred, clip.vectors),
        "full_l1": l1(pred, clip.vectors),
        "missing_l1": float(np.mean(np.abs(pred[missing_mask] - clip.vectors[missing_mask]))) if missing_frames else 0.0,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for test_set in sorted({str(r["test_set"]) for r in rows}):
        for model in sorted({str(r["model"]) for r in rows if r["test_set"] == test_set}):
            subset = [r for r in rows if r["test_set"] == test_set and r["model"] == model]
            item = {
                "test_set": test_set,
                "model": model,
                "clips": len(subset),
                "frames": int(sum(int(r["frames"]) for r in subset)),
                "mean_mask_ratio": float(np.mean([float(r["mask_ratio"]) for r in subset])),
            }
            for metric in ["shifted_distance", "npss", "full_l1", "missing_l1"]:
                values = np.asarray([float(r[metric]) for r in subset], dtype=np.float64)
                values = values[np.isfinite(values)]
                item[metric] = float(np.mean(values)) if values.size else float("nan")
                item[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
                item[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
            out.append(item)
    return out


def _write_dashboard(path: Path, *, rows: list[dict[str, Any]], summary: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    payload = json.dumps(
        {
            "clipRows": rows,
            "summaryRows": summary,
            "manifest": manifest,
            "metrics": ["shifted_distance", "npss", "missing_l1", "full_l1"],
            "metricInfo": {
                "shifted_distance": "Paper-style shifted temporal L1. Lower is better. This is the main timing-tolerant pose error metric.",
                "npss": "Normalized Power Spectrum Similarity. Lower is better. This measures temporal frequency/rhythm mismatch.",
                "missing_l1": "Mean absolute controller error only on masked frames.",
                "full_l1": "Mean absolute controller error over the full sequence.",
            },
        }
    )
    title = "Official-Protocol AIS Benchmark"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f5ef;
      --panel: #fff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d6dccd;
      --accent: #0f766e;
      --red: #c2410c;
      --good: #12805c;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 22px 28px 16px; border-bottom: 1px solid var(--line); background: #ecefe4; }}
    h1 {{ margin: 0 0 10px; font-size: 26px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    h3 {{ margin: 0 0 8px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    main {{ padding: 20px 28px 32px; }}
    .controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; align-items: end; }}
    label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 700; }}
    select {{ width: 100%; border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 8px 10px; color: var(--ink); }}
    .grid {{ display: grid; gap: 14px; }}
    .top {{ grid-template-columns: 1.15fr .85fr; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(360px, .75fr); gap: 14px; margin-top: 14px; }}
    .panel, .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 1px 0 rgba(17,24,39,.04); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ margin-top: 4px; font-size: 24px; font-weight: 760; }}
    .metric .delta {{ margin-top: 6px; font-size: 12px; }}
    .metric .hint {{ margin-top: 8px; color: var(--muted); font-size: 11px; }}
    .better {{ color: var(--good); }}
    .worse {{ color: var(--bad); }}
    .muted {{ color: var(--muted); }}
    .chart {{ width: 100%; height: 330px; border: 1px solid var(--line); border-radius: 6px; background: #fbfcf8; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #edf0e8; padding: 7px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 750; }}
    tbody tr:hover {{ background: #f8faf4; }}
    .scroll {{ overflow: auto; max-height: 420px; border: 1px solid var(--line); border-radius: 6px; }}
    .note {{ border-left: 4px solid var(--accent); background: #f7faf4; padding: 10px 12px; border-radius: 6px; margin: 8px 0; color: #344054; }}
    .legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 99px; margin-right: 5px; }}
    code {{ overflow-wrap: anywhere; }}
    @media (max-width: 980px) {{ .top, .layout {{ grid-template-columns: 1fr; }} main, header {{ padding-left: 16px; padding-right: 16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="controls">
      <label>Test Set<select id="testSet"></select></label>
      <label>Model A<select id="modelA"></select></label>
      <label>Model B<select id="modelB"></select></label>
      <label>Metric<select id="metric"></select></label>
    </div>
  </header>
  <main>
    <section class="grid top">
      <div class="panel">
        <h2>Primary Metrics</h2>
        <div id="metricCards" class="cards"></div>
      </div>
      <div class="panel">
        <h2>Protocol</h2>
        <div id="context"></div>
      </div>
    </section>
    <section class="layout">
      <div class="grid">
        <div class="panel">
          <h2>Model Comparison</h2>
          <svg id="barChart" class="chart" viewBox="0 0 900 330" preserveAspectRatio="none"></svg>
          <div id="legend" class="legend"></div>
        </div>
        <div class="panel">
          <h2>Clip Delta Distribution</h2>
          <svg id="scatterChart" class="chart" viewBox="0 0 900 330" preserveAspectRatio="none"></svg>
        </div>
      </div>
      <div class="grid">
        <div class="panel">
          <h2>Per-Clip Deltas</h2>
          <div class="scroll"><table id="clipTable"></table></div>
        </div>
        <div class="panel">
          <h2>All Summary Rows</h2>
          <div class="scroll"><table id="summaryTable"></table></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = {payload};
    const colors = ["#0f766e", "#c2410c", "#334155", "#b45309", "#4f46e5", "#047857"];
    const $ = id => document.getElementById(id);
    const uniq = arr => [...new Set(arr)];
    const fmt = x => Number.isFinite(+x) ? (+x).toFixed(Math.abs(+x) < 0.01 ? 6 : 4) : "";
    const lowerIsBetter = new Set(DATA.metrics);

    function fillSelect(el, values, preferred) {{
      el.innerHTML = values.map(v => `<option value="${{v}}">${{v}}</option>`).join("");
      if (preferred && values.includes(preferred)) el.value = preferred;
    }}
    function rowFor(testSet, model) {{
      return DATA.summaryRows.find(r => r.test_set === testSet && r.model === model);
    }}
    function betterClass(metric, delta) {{
      if (!Number.isFinite(delta) || delta === 0) return "muted";
      return lowerIsBetter.has(metric) ? (delta < 0 ? "better" : "worse") : (delta > 0 ? "better" : "worse");
    }}
    function table(el, rows, cols) {{
      el.innerHTML = `<thead><tr>${{cols.map(c => `<th>${{c.label}}</th>`).join("")}}</tr></thead><tbody>` +
        rows.map(r => `<tr>${{cols.map(c => `<td>${{c.format ? c.format(r[c.key], r) : r[c.key]}}</td>`).join("")}}</tr>`).join("") + "</tbody>";
    }}
    function drawBars(svg, rows, metric) {{
      const w = 900, h = 330, l = 62, r = 20, t = 28, b = 56;
      svg.innerHTML = "";
      const vals = rows.map(r => +r[metric]).filter(Number.isFinite);
      if (!vals.length) return;
      const max = Math.max(...vals) * 1.12 || 1;
      const plotW = w - l - r, plotH = h - t - b;
      for (let i=0; i<=5; i++) {{
        const y = t + plotH - (i/5)*plotH;
        const v = max*i/5;
        svg.insertAdjacentHTML("beforeend", `<line x1="${{l}}" y1="${{y}}" x2="${{w-r}}" y2="${{y}}" stroke="#d6dccd"/><text x="${{l-8}}" y="${{y+4}}" text-anchor="end" font-size="11" fill="#667085">${{fmt(v)}}</text>`);
      }}
      svg.insertAdjacentHTML("beforeend", `<line x1="${{l}}" y1="${{h-b}}" x2="${{w-r}}" y2="${{h-b}}" stroke="#1f2933"/><line x1="${{l}}" y1="${{t}}" x2="${{l}}" y2="${{h-b}}" stroke="#1f2933"/>`);
      const step = plotW / rows.length;
      rows.forEach((row, i) => {{
        const value = +row[metric];
        const bh = value / max * plotH;
        const x = l + i*step + step*.18;
        const y = h-b-bh;
        svg.insertAdjacentHTML("beforeend", `<rect x="${{x}}" y="${{y}}" width="${{step*.64}}" height="${{bh}}" rx="4" fill="${{colors[i%colors.length]}}"><title>${{row.model}}: ${{fmt(value)}}</title></rect><text x="${{x+step*.32}}" y="${{h-b+19}}" text-anchor="middle" font-size="10" fill="#667085">${{row.model.slice(0,16)}}</text>`);
      }});
      $("legend").innerHTML = rows.map((r,i) => `<span><span class="dot" style="background:${{colors[i%colors.length]}}"></span>${{r.model}}</span>`).join("");
    }}
    function drawScatter(svg, clips, metric) {{
      const w = 900, h = 330, l = 62, r = 20, t = 26, b = 48;
      svg.innerHTML = "";
      if (!clips.length) return;
      const deltas = clips.map(c => +c.delta).filter(Number.isFinite);
      const maxAbs = Math.max(...deltas.map(Math.abs), 1e-9);
      const plotW = w-l-r, plotH = h-t-b;
      const y0 = t + plotH/2;
      svg.insertAdjacentHTML("beforeend", `<line x1="${{l}}" y1="${{y0}}" x2="${{w-r}}" y2="${{y0}}" stroke="#1f2933"/><line x1="${{l}}" y1="${{t}}" x2="${{l}}" y2="${{h-b}}" stroke="#1f2933"/><text x="${{w/2}}" y="${{h-12}}" text-anchor="middle" font-size="11" fill="#667085">clip rank</text><text x="${{l-8}}" y="${{t+10}}" text-anchor="end" font-size="11" fill="#667085">worse</text><text x="${{l-8}}" y="${{h-b-4}}" text-anchor="end" font-size="11" fill="#667085">better</text>`);
      clips.forEach((clip, i) => {{
        const x = l + (clips.length === 1 ? .5 : i/(clips.length-1)) * plotW;
        const y = y0 - (+clip.delta / maxAbs) * (plotH/2*.9);
        const color = +clip.delta < 0 ? "#12805c" : "#b42318";
        svg.insertAdjacentHTML("beforeend", `<circle cx="${{x}}" cy="${{y}}" r="4" fill="${{color}}" opacity=".78"><title>${{clip.clip_id}} ${{metric}} delta: ${{fmt(clip.delta)}}</title></circle>`);
      }});
    }}
    function render() {{
      const testSet = $("testSet").value;
      const modelA = $("modelA").value;
      const modelB = $("modelB").value;
      const metric = $("metric").value;
      const rows = DATA.summaryRows.filter(r => r.test_set === testSet);
      const a = rowFor(testSet, modelA);
      const b = rowFor(testSet, modelB);
      $("metricCards").innerHTML = DATA.metrics.map(m => {{
        const delta = a && b ? +a[m] - +b[m] : NaN;
        return `<div class="metric card" title="${{DATA.metricInfo[m]}}"><div class="label">${{m}}</div><div class="value">${{a ? fmt(a[m]) : ""}}</div><div class="delta ${{betterClass(m, delta)}}">vs ${{modelB}}: ${{fmt(delta)}}</div><div class="hint">${{DATA.metricInfo[m]}}</div></div>`;
      }}).join("");
      $("context").innerHTML = `<div class="note"><strong>Main paper-style metrics:</strong> shifted_distance and npss. Lower is better.</div>` +
        DATA.manifest.notes.map(n => `<div class="note">${{n}}</div>`).join("") +
        `<table><tbody><tr><th>test set</th><td>${{testSet}}</td></tr><tr><th>dataset role</th><td>${{DATA.manifest.dataset_roles?.[testSet] || "unknown"}}</td></tr><tr><th>random seed</th><td>${{DATA.manifest.random_seed}}</td></tr><tr><th>random mask ratio</th><td>${{DATA.manifest.random_mask_ratio}}</td></tr><tr><th>max clips</th><td>${{DATA.manifest.max_clips ?? "all"}}</td></tr></tbody></table>`;
      drawBars($("barChart"), rows, metric);
      const clips = DATA.clipRows.filter(r => r.test_set === testSet && r.model === modelA).map(r => {{
        const other = DATA.clipRows.find(x => x.test_set === testSet && x.model === modelB && x.clip_id === r.clip_id);
        return {{...r, delta: other ? +r[metric] - +other[metric] : NaN}};
      }}).sort((x,y) => Math.abs(y.delta) - Math.abs(x.delta));
      drawScatter($("scatterChart"), clips, metric);
      table($("clipTable"), clips.slice(0, 150), [
        {{key:"clip_id", label:"clip"}},
        {{key:"scene_id", label:"scene"}},
        {{key:metric, label:metric, format:fmt}},
        {{key:"delta", label:"A-B", format:v => `<span class="${{betterClass(metric,+v)}}">${{fmt(v)}}</span>`}},
        {{key:"mask_ratio", label:"mask", format:fmt}},
        {{key:"frames", label:"frames"}}
      ]);
      table($("summaryTable"), rows, [
        {{key:"model", label:"model"}},
        {{key:"clips", label:"clips"}},
        {{key:"mean_mask_ratio", label:"mask", format:fmt}},
        {{key:"shifted_distance", label:"shifted", format:fmt}},
        {{key:"npss", label:"NPSS", format:fmt}},
        {{key:"missing_l1", label:"missing L1", format:fmt}},
        {{key:"full_l1", label:"full L1", format:fmt}}
      ]);
    }}
    const testSets = uniq(DATA.summaryRows.map(r => r.test_set));
    const models = uniq(DATA.summaryRows.map(r => r.model));
    fillSelect($("testSet"), testSets, testSets.includes("production") ? "production" : testSets[0]);
    fillSelect($("modelA"), models, models[models.length - 1]);
    fillSelect($("modelB"), models, models[0]);
    fillSelect($("metric"), DATA.metrics, "shifted_distance");
    ["testSet", "modelA", "modelB", "metric"].forEach(id => $(id).addEventListener("change", render));
    render();
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def run_official_protocol_benchmark(
    *,
    config: dict[str, Any],
    checkpoints,
    output_dir: Path,
    test_set: str,
    max_clips: int | None,
    device_name: str,
    random_seed: int,
    random_mask_ratio: float,
) -> dict[str, Path]:
    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _dataset_specs(config)
    selected = TEST_SETS if test_set == "all" else (test_set,)
    models = {spec.label: load_improved_model(spec.path, device=str(device)) for spec in checkpoints}
    rng = np.random.default_rng(random_seed)

    rows: list[dict[str, Any]] = []
    for name in selected:
        spec = specs[name]
        if not spec.controller_h5.exists():
            raise FileNotFoundError(f"controller HDF5 not found for {name}: {spec.controller_h5}")
        if not spec.block_keyframes_h5.exists():
            raise FileNotFoundError(f"block keyframes HDF5 not found for {name}: {spec.block_keyframes_h5}")
        for clip in iter_animaj_hdf5(spec.controller_h5, spec.block_keyframes_h5, max_clips=max_clips):
            if spec.protocol == "random_uniform_90":
                unmasked = _random_uniform_unmasked(len(clip.vectors), ratio_masked=random_mask_ratio, rng=rng)
            else:
                unmasked = _block_unmasked(clip)
            for model_name, model in models.items():
                pred = _predict(model, clip, unmasked, device)
                rows.append(_clip_row(name, model_name, clip, pred, unmasked))

    summary = _aggregate(rows)
    clip_csv = output_dir / "official_protocol_clip_metrics.csv"
    summary_csv = output_dir / "official_protocol_summary_metrics.csv"
    manifest_json = output_dir / "official_protocol_manifest.json"
    dashboard_html = output_dir / "official_protocol_dashboard.html"
    _write_csv(clip_csv, rows)
    _write_csv(summary_csv, summary)
    manifest = {
        "test_set": test_set,
        "max_clips": max_clips,
        "random_seed": random_seed,
        "random_mask_ratio": random_mask_ratio,
        "dataset_roles": {name: specs[name].role for name in selected},
        "checkpoints": [{"label": spec.label, "path": str(spec.path)} for spec in checkpoints],
        "notes": [
            "This adapter ports the public official test protocols and metrics into this repo for local model comparisons.",
            "It is closer to the paper than the exploratory dashboard, but the exact upstream Lightning datamodule remains the final oracle.",
            "Main paper metrics map to shifted_distance and npss.",
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_dashboard(dashboard_html, rows=rows, summary=summary, manifest=manifest)
    return {"clip_metrics": clip_csv, "summary_metrics": summary_csv, "manifest": manifest_json, "dashboard": dashboard_html}


def _clip_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official-style AIS benchmark protocols for local checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/ais_repro.yaml"))
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path or label=path. Repeat for multiple models.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/official_protocol"))
    parser.add_argument("--test-set", choices=[*TEST_SETS, "all"], default="all")
    parser.add_argument("--max-clips", type=int, default=20, help="Use 0 to score every clip.")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--random-mask-ratio", type=float, default=0.9)
    args = parser.parse_args()

    outputs = run_official_protocol_benchmark(
        config=_load_yaml(args.config),
        checkpoints=[parse_checkpoint(item) for item in args.checkpoint],
        output_dir=args.output_dir,
        test_set=args.test_set,
        max_clips=_clip_limit(args.max_clips),
        device_name=args.device,
        random_seed=args.random_seed,
        random_mask_ratio=args.random_mask_ratio,
    )
    print(yaml.safe_dump({key: str(value) for key, value in outputs.items()}, sort_keys=True).strip())


if __name__ == "__main__":
    main()
