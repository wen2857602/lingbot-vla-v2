#!/usr/bin/env python3
"""为 LeRobot v3 结构化数据集生成 robot_config + norm_states(离线版)。

来源:rsystem launcher 自适应层(launcher.py _make_export_support)的独立化脚本,
语义逐句对齐(相同统计口径/相同模板)。通用训练链在 GPU pod 内自动做这件事;
自定义训练 pod 没有 rsystem,故此脚本手工补齐。

用法(容器/任何 python>=3.10,numpy+pyarrow+pyyaml 即可):
    python3 gen_norm_and_robot_config.py <数据集根目录>... <输出目录>
    # 数据集根目录含 data/chunk-*/file-*.parquet 与 meta/info.json;可给多个目录 =
    # 多数据集混训,帧 union 后算一份合并 norm(与管理训练链的合并契约一致,
    # songling 同构型参数,混训即一份 config + 一份合并归一化)
    # 输出目录将生成:
    #   <out>/robot_configs/agilex_export_0.yaml   (norm_stats 已指向绝对路径)
    #   <out>/robot_configs/norm_states.json
    #   <out>/datasets.txt                          (多数据集时每行同一个 config 前缀)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

COLS = ["observation.state.arm.position", "observation.state.effector.position",
        "action.arm.position", "action.effector.position"]

# 与训练栈(rsystem launcher _ROBOT_CONFIG_TEMPLATE)完全一致;relative_type: joint
# 必须显式给出,缺省会走四元数分支导致全量 AssertionError(2026-10-21 实测)。
TEMPLATE = {
    "states": [
        {"observation.state.arm.position": {"origin_keys": "observation.state.arm.position"}},
        {"observation.state.effector.position": {"origin_keys": "observation.state.effector.position"}},
    ],
    "actions": [
        {"action.arm.position": {"origin_keys": "action.arm.position",
                                 "subtract_state": True, "relative_type": "joint"}},
        {"action.effector.position": {"origin_keys": "action.effector.position",
                                       "subtract_state": False}},
    ],
    "images": [
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
        "observation.images.camera_wrist_right",
    ],
}


def stats_of(arr):
    if arr.size == 0:
        z = arr.reshape(0)
        return {k: z.tolist() for k in ("mean", "std", "min", "max", "q01", "q99", "q02", "q98")}
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        "q02": np.quantile(arr, 0.02, axis=0).tolist(),
        "q98": np.quantile(arr, 0.98, axis=0).tolist(),
    }


def main():
    if len(sys.argv) < 3:
        sys.exit("用法: gen_norm_and_robot_config.py <数据集目录>... <输出目录>")
    ds_dirs = [Path(a).resolve() for a in sys.argv[1:-1]]
    out_dir = Path(sys.argv[-1]).resolve()
    cfg_dir = out_dir / "robot_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    buckets = {c: [] for c in COLS}
    widths: dict[str, int | None] = {c: None for c in COLS}
    for ds_dir in ds_dirs:
        info = ds_dir / "meta" / "info.json"
        if not info.is_file():
            sys.exit(f"meta/info.json 不存在: {ds_dir}")
        feats = json.loads(info.read_text()).get("features") or {}
        missing = [c for c in COLS if c not in feats]
        if missing:
            sys.exit(f"features 缺键 {missing}——不是本模板覆盖的结构化构型,勿硬套: {ds_dir}")
        files = sorted((ds_dir / "data").glob("chunk-*/file-*.parquet")) \
            or sorted((ds_dir / "data").glob("chunk-*/*.parquet"))
        if not files:
            sys.exit(f"数据 parquet 不存在: {ds_dir/'data'}")
        for f in files:
            tbl = pq.read_table(f, columns=COLS)
            for c in COLS:
                arr = np.asarray(tbl.column(c).to_pylist(), dtype=np.float64)
                w = arr.shape[1] if arr.ndim > 1 else 0
                if widths[c] is None:
                    widths[c] = w
                elif w != widths[c]:
                    sys.exit(f"数据集状态空间不一致,拒绝合并: {c} 宽度 "
                             f"{ds_dir}={w} != 前序={widths[c]}")
                buckets[c].append(arr)
    data = {c: np.concatenate(v, axis=0) for c, v in buckets.items()}

    norm_stats = {
        # state 按原始值统计;action.arm 按 delta=action-state 统计
        # (与训练栈 transform.apply 的 subtract→normalize 顺序严格一致)
        "observation.state.arm.position": stats_of(data[COLS[0]]),
        "observation.state.effector.position": stats_of(data[COLS[1]]),
        "action.arm.position": stats_of(data[COLS[2]] - data[COLS[0]]),
        "action.effector.position": stats_of(data[COLS[3]]),
    }
    stats_path = cfg_dir / "norm_states.json"
    stats_path.write_text(json.dumps({"norm_stats": norm_stats,
                                      "count": int(data[COLS[0]].shape[0])},
                                     ensure_ascii=False), encoding="utf-8")

    cfg = TEMPLATE.copy()
    cfg["norm_stats"] = str(stats_path)          # 训练要求:绝对路径
    cfg_path = cfg_dir / "agilex_export_0.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"norm_states: {stats_path}  (frames={data[COLS[0]].shape[0]})")
    print(f"robot_config: {cfg_path}")
    # datasets.txt:一行一数据集,多数据集共用同一 config 前缀(合并归一化的约定)
    lines = [f"agilex_export_0 {d}" for d in ds_dirs]
    (out_dir / "datasets.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"datasets.txt: {out_dir/'datasets.txt'}  ({len(lines)} dataset line(s), cfg=agilex_export_0)")


if __name__ == "__main__":
    main()
