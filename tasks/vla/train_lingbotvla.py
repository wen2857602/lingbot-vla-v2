import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from lingbotvla.utils.async_tb_writer import AsyncTBWriter
from lingbotvla.checkpoint import build_checkpointer
from lingbotvla.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    VLADataCollatorWithPacking,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
    build_vla_dataset,
)
from lingbotvla.distributed.offloading import build_activation_offloading_context
from lingbotvla.distributed.parallel_state import get_parallel_state, init_parallel_state
from lingbotvla.distributed.torch_parallelize import build_parallelize_model
from lingbotvla.models import build_foundation_model, build_processor, save_model_assets, build_tokenizer
from lingbotvla.optim import build_lr_scheduler, build_muon_optimizer, build_optimizer
from lingbotvla.utils import helper
from lingbotvla.utils.async_hf_checkpoint import AsyncHFCheckpointSaver
from lingbotvla.utils.arguments import EvalArguments, DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from lingbotvla.utils.dist_utils import all_reduce
from lingbotvla.models.config_registry import get_config_registry

from lingbotvla.models.vla.vision_models.module_utils import (
    build_depth_model,
    build_video_model,
    get_depth_target,
    get_video_target,
    log_video,
)
from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
import gc
import shutil
import threading
gc.set_threshold(50000, 50, 50)

logger = helper.create_logger(__name__)
# try:
#     from aistudio_tracking import training_tracking as wandb
# except Exception as e:
#     logger.info_rank0(f"Failed to import aistudio_tracking: {repr(e)}.")

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]


def get_moe_param_groups(model: "torch.nn.Module", args_train) -> Optional[List[Dict]]:
    """
    Build optimizer param groups with token-MoE expert LR scaling.

    Only routed expert weights (.mlp.experts.*) receive the scaled LR.
    Gate / shared-expert / all dense params keep the base LR.

    Scale formula (always auto-computed):
        token-level MoE:  scale = (token_num_experts  / token_top_k)^0.5

    Returns None when MoE is disabled so the caller can fall back to the
    default single-group behaviour.
    """
    if not getattr(args_train, 'use_moe', False):
        return None
    if not getattr(args_train, 'use_moe_expert_lr', False):
        return None

    base_lr = args_train.lr

    token_moe_layers = set(getattr(args_train, 'token_moe_layers', None) or [])

    # Auto-compute token expert scale: (num_experts / top_k)^0.5
    token_scale = (args_train.token_num_experts / args_train.token_top_k) ** 0.5 if token_moe_layers else None

    if token_scale is None:
        return None

    # Match routed-expert params by FQN: ...layers.<idx>.mlp.experts....
    # Gate / shared_expert / shared_expert_gate keep base_lr.
    layer_expert_re = re.compile(r'\.layers\.(\d+)\.mlp\.experts\.')

    lr_to_params: Dict[float, List] = {base_lr: []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        m = layer_expert_re.search(name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx in token_moe_layers and token_scale is not None:
                expert_lr = base_lr * token_scale
            else:
                expert_lr = base_lr
            lr_to_params.setdefault(expert_lr, []).append(param)
        else:
            lr_to_params[base_lr].append(param)

    return [{"params": params, "lr": lr} for lr, params in lr_to_params.items() if params]

@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    freeze_vision_encoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vision encoder in PI0 model."},
    )
    train_expert_only: bool = field(
        default=False,
        metadata={"help": "Train action expert only or not."},
    )
    train_state_proj: bool = field(
        default=True,
        metadata={"help": "Train state proj only or not."},
    )
    tokenizer_max_length: int = field(
        default=48,
        metadata={"help": "Maximum length of the tokenizer."},
    )
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension."},
    )
    max_action_dim: int = field(
        default=32,
        metadata={"help": "Action dimension after padding."},
    )
    max_state_dim: int = field(
        default=32,
        metadata={"help": "State dimension after padding."},
    )
    chunk_size: int = field(
        default=50,
        metadata={"help": "Chunk size of action."},
    )
    vlm_causal: bool = field(
        default=False,
        metadata={"help": "Whether to use causal atten for img anb lang tokens in vlm."},
    )
    loss_type: str = field(
        default='fm',
        metadata={"help": "Which loss to use."},
    )
    align_params: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "The config of vaco"},
    )
    attention_implementation: str = field(
        default="flex",
        metadata={"help": "Attention implementation: flex, flex_cached, or eager."},
    )
    my_tokenizer_max_length: int = field(
        default=72,
        metadata={"help": ""},
    )
    decayed_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum norm for the decayed gradients."},
    )
    stable_train_steps: int = field(
        default=100000,
        metadata={"help": "Training steps for stable training, after this step, the decayed_max_grad_norm will be applied."},
    )
    resume_dataloader_state: bool = field(
        default=True,
        metadata={"help": "Whether to resume dataloader."},
    )
    use_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use MoE."},
    )
    token_moe_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Layer indices for token-level routing MoE, e.g. [1,2,...,34]."},
    )
    token_num_experts: int = field(
        default=32,
        metadata={"help": "Number of experts per token-level MoE layer."},
    )
    token_top_k: int = field(
        default=1,
        metadata={"help": "Top-k for token-level routing."},
    )
    token_moe_intermediate_size: int = field(
        default=256,
        metadata={"help": "Intermediate size for token-level MoE expert FFN."},
    )
    token_shared_intermediate_size: int = field(
        default=256,
        metadata={"help": "Intermediate size for token-level shared expert FFN."},
    )
    bias_update_speed: float = field(
        default=0.001,
        metadata={"help": "Bias update speed for loss-free MoE load balancing."},
    )
    bias_centering: bool = field(
        default=False,
        metadata={"help": "If True, center the loss-free correction bias each step (subtract per-layer mean) to prevent cumulative drift. Routing-invariant; pure numerical hygiene."},
    )
    bias_update_interval: int = field(
        default=1,
        metadata={"help": "Apply the loss-free bias update once every N optimizer steps, accumulating tokens_per_expert in between. >1 averages load over more tokens so sign(load-mean) is reliable when the global batch is small (per-step load too noisy -> bias random-walks). Also makes the load monitor (maxvio/min_load/has_dead) windowed/less noisy."},
    )
    sequence_wise_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for sequence-wise MoE balance loss (DeepSeek-V3 style, complementary to loss-free bias). 0 = disabled."},
    )
    sequence_wise_mode: str = field(
        default="per_sequence",
        metadata={"help": "Granularity of the sequence-wise balance loss: 'per_sequence' (balance experts within each sample's token sequence, DeepSeek-V3 intent) or 'global' (treat the whole batch as one sequence). Ablation knob."},
    )
    router_z_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for router z-loss computed on raw router logits. 0 = disabled."},
    )
    moe_monitor_interval: int = field(
        default=50,
        metadata={"help": "Step interval for writing per-layer MoE monitor scalars (moe_maxvio/minvio/minload/entropy/topksigmoid) and expert-selection histograms to TensorBoard. moe_summary/* is always written every step. Set small (e.g. 1/10) for close debugging."},
    )
    router_activation: str = field(
        default="softmax",
        metadata={"help": "Router activation function: 'softmax' or 'sigmoid'. Default 'softmax' for backward compat."},
    )
    routed_scaling_factor: float = field(
        default=1.0,
        metadata={"help": "Scaling factor applied to routing weights after norm (DeepSeek-V3 style). 1.0 = disabled."},
    )
    use_shared_expert_gate: bool = field(
        default=True,
        metadata={"help": "Whether to use sigmoid gate on shared expert output. True = current behavior, False = direct add (DS-V3 style)."},
    )
    # Action expert architecture params
    expert_hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size for action expert."},
    )
    expert_intermediate_size: int = field(
        default=2752,
        metadata={"help": "FFN intermediate size for action expert."},
    )
    action_fp32: bool = field(
        default=False,
        metadata={"help": "Whether to use fp32 action and state."},
    )
    precompute_grid_thw: bool = field(
        default=False,
        metadata={"help": "Whether to precompute and cache grid_thw-derived tensors (rotary_pos_emb, window_index, etc.) for fixed-resolution training."},
    )
    use_moe_expert_lr: bool = field(
        default=False,
        metadata={"help": "Whether to apply scaled LR to MoE routed experts."},
    )
    split_fused_experts_from_decoder_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to exclude Qwen2FusedExperts params from Qwen2DecoderLayer FSDP2 units without wrapping the experts in FSDP2."},
    )
    vlm_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to apply FSDP2 for VLM."},
    )

@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    robot_config_root: str = field(
        default=None,
        metadata={"help": "Path to get all robot configs."},
    )
    joints: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of joints and their dim"},
    )
    cameras:Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of used images"},
    )
    norm_type: Optional[List[str]] = field(default=None, metadata={"help": "Normalization type."})
    img_size: int = field(
        default=256,
        metadata={"help": "Size of the image."},
    )
    norm_stats_file: str = field(
        default=None,
        metadata={"help": "Path to the normalization stats file."},
    )
    prompt_type: Literal["global", "subtask"] = field(
        default="global",
        metadata={"help": "Type of the prompt."},
    )
    use_future_image: bool = field(
        default=False,
        metadata={"help": "Whether to use future image."},
    )


@dataclass
class Arguments:

    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)
    eval: "EvalArguments" = field(default_factory=EvalArguments)


def _load_forge_events():
    """Forge 平台训练事件(可选插桩,best-effort)。

    rsystem 的提交端经 FORGE_EVENTS_HOOK=<events.py 绝对路径> 指向独立单文件模块:
    - 不在环境 = 官方行为零改动;在环境但无 FORGE_EVENT_FILE/RDATABUS 配置 → None
    - 事件四类: TRAIN_STARTED / TRAIN_METRICS(=TB 信息,每 N 步抽样)/ CKPT_READY /
      TRAIN_FINISHED;jsonl 兜底 + 可选 R-Databus,由该模块自管,这里只 emit。
    """
    try:
        import importlib.util
        hook = os.environ.get("FORGE_EVENTS_HOOK", "")
        if not hook or not os.path.isfile(hook):
            return None
        spec = importlib.util.spec_from_file_location("forge_events", hook)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_publisher()
    except Exception:
        return None


def _forge_emit(pub, method, *args, **kwargs):
    """事件发布绝不弄死训练:所有调用点都过这里包死。"""
    try:
        if pub is not None:
            getattr(pub, method)(*args, **kwargs)
    except Exception:
        pass


def _forge_prune_old_dcp(save_checkpoint_path, keep=None):
    """dcp 滚动清理:每个 save 点后删更旧的 dcp 大件(model/optimizer/extra_state),
    保留最近 keep 套(FORGE_KEEP_DCP env,默认 2)。best-effort,失败不影响训练。"""
    try:
        import re as _re
        import shutil as _shutil
        if not save_checkpoint_path or not os.path.isdir(save_checkpoint_path):
            return
        keep = keep or int(os.environ.get("FORGE_KEEP_DCP", "2"))
        steps = []
        for d in os.listdir(save_checkpoint_path):
            m = _re.fullmatch(r"global_step_(\d+)", d)
            if m and os.path.isdir(os.path.join(save_checkpoint_path, d, "model")):
                steps.append((int(m.group(1)), d))
        steps.sort()
        for _s, d in (steps[:-keep] if len(steps) > keep else []):
            base = os.path.join(save_checkpoint_path, d)
            for part in ("model", "optimizer", "extra_state"):
                p = os.path.join(base, part)
                if os.path.isdir(p):
                    _shutil.rmtree(p, ignore_errors=True)
                    print(f"[forge-prune] removed old dcp: {p}")
    except Exception as e:
        print(f"[forge-prune] failed (ignored): {e!r}")


def _forge_export_names() -> list:
    """Adapt manifest entries: config names generated for this job's datasets.

    Two sources, managed first:
    - managed chain: FORGE_ADAPT_MANIFEST (launcher's .forge_adapt/manifest.json)
    - custom/bare runs (2026-11-04 fix): FORGE_EXPORT_NAMES, fed by main() from
      the datasets.txt prefixes. Without this fallback the flatten loop was a
      no-op on custom pods and every published hf_ckpt shipped WITHOUT
      robot_config.yaml/norm_states.json — the exact gap seen on the ec80ce86
      order (10 checkpoints, self-contained three-piece missing)."""
    manifest = os.environ.get("FORGE_ADAPT_MANIFEST", "").strip()
    if not (manifest and os.path.isfile(manifest)):
        names = [n for n in os.environ.get("FORGE_EXPORT_NAMES", "").split(",") if n]
        return list(dict.fromkeys(names))
    try:
        import json as _json
        with open(manifest, encoding="utf-8") as f:
            return list(_json.load(f).get("export_configs") or [])
    except Exception as e:
        print(f"[forge-slim] manifest unreadable: {e!r}")
        return []


def _forge_train_config_text(args) -> str:
    """Train config shipped inside each ckpt (fork-native lingbotvla_cli.yaml).

    Same structure as the legacy save_args dump (asdict minus eval). Fills
    data.norm_stats_file with norm_states.json whenever any export config is
    present — multi-dataset orders now merge into ONE unified config+stats
    pair on the launcher side (2026-11-05, 4b8ba4ba dispatch failure), so the
    field is never null and always names a real sibling file."""
    try:
        args_dict = asdict(args)
        args_dict.pop("eval", None)
        if _forge_export_names():
            args_dict.setdefault("data", {})["norm_stats_file"] = "norm_states.json"
        import yaml as _yaml
        return _yaml.safe_dump(args_dict, default_flow_style=False)
    except Exception as e:
        print(f"[forge-slim] train config serialization failed: {e!r}")
        return ""


def _forge_publish_hf_ckpt(ckpt_dir, publish_root):
    """Copy the finished hf_ckpt/ to the mounted checkpoints dir (2026-10-29
    layering: raw dcp stays pod-local, only the slimmed hf bundle reaches the
    mount — a kill inside the save window can only burn the local layer, the
    mount never holds half-written dcp shards). config.json is copied last as
    the completeness sentinel: an interrupted publish leaves no config.json,
    so consumers never see a half artifact. Idempotent via the same sentinel.
    Returns the published hf_ckpt path, or None on failure (local copy kept)."""
    try:
        import shutil as _shutil
        hf_dir = os.path.join(ckpt_dir, "hf_ckpt")
        step_name = os.path.basename(os.path.normpath(ckpt_dir))
        dst = os.path.join(publish_root, step_name, "hf_ckpt")
        if os.path.isfile(os.path.join(dst, "config.json")):
            return dst
        os.makedirs(dst, exist_ok=True)
        for name in sorted(os.listdir(hf_dir)):
            src = os.path.join(hf_dir, name)
            if name == "config.json" or not os.path.isfile(src):
                continue
            # copyfile NOT copy2 (2026-11-03 custom-training EPERM): copy2's
            # copystat chmod/utime hit the publish mount (ossfs, no setattr
            # support) -> PermissionError(1) under the unprivileged custom
            # training uid. Managed orders run as root and never tripped it;
            # content copy is all a publish needs.
            _shutil.copyfile(src, os.path.join(dst, name))
        _shutil.copyfile(os.path.join(hf_dir, "config.json"), os.path.join(dst, "config.json"))
        print(f"[forge-slim] published hf_ckpt -> {dst}")
        return dst
    except Exception as e:
        print(f"[forge-slim] publish {ckpt_dir} failed (kept local): {e!r}")
        return None


def _forge_slim_ckpt_dir(ckpt_dir, cli_yaml_text=None, publish_root=None):
    """Self-contained ckpt for model distribution (backend contract 2026-10-26).

    Inference assets are flattened INTO hf_ckpt/:
      lingbotvla_cli.yaml   train config (data.norm_stats_file -> norm_states.json)
      robot_config.yaml     robot config of the export dataset
      norm_states.json      normalization stats                       \
      *_<i>.yaml / *_<i>.json  suffix variants when multiple datasets  /
    The robot config's internal norm_stats pointer is rewritten to the
    relative sibling file so the ckpt stays valid after it leaves the pod.
    publish_root given = layering mode: additionally publish hf_ckpt to the
    mounted checkpoints dir (see _forge_publish_hf_ckpt) and strip dcp parts
    from the LOCAL layer only. FORGE_CKPT_KEEP_DCP keeps local dcp; a step
    whose HF conversion did not finish keeps its dcp as the only surviving
    copy (pod-local). Best-effort, never kills training."""
    hf_published = None
    try:
        import shutil as _shutil
        import yaml as _yaml
        if not os.path.isdir(ckpt_dir):
            return None
        hf_dir = os.path.join(ckpt_dir, "hf_ckpt")
        if not os.path.isfile(os.path.join(hf_dir, "config.json")):
            print(f"[forge-slim] {ckpt_dir} hf_ckpt not ready, keep dcp")
            return None
        if cli_yaml_text and not os.path.exists(os.path.join(hf_dir, "lingbotvla_cli.yaml")):
            with open(os.path.join(hf_dir, "lingbotvla_cli.yaml"), "w", encoding="utf-8") as f:
                f.write(cli_yaml_text)
            print(f"[forge-slim] wrote lingbotvla_cli.yaml -> {hf_dir}")
        names = _forge_export_names()
        manifest = os.environ.get("FORGE_ADAPT_MANIFEST", "").strip()
        base = os.path.dirname(manifest) if manifest else os.environ.get(
            "FORGE_ROBOT_CONFIG_ROOT", "").strip()
        for i, name in enumerate(names):
            suffix = "" if len(names) == 1 else f"_{i}"
            cfg_dst = os.path.join(hf_dir, f"robot_config{suffix}.yaml")
            stats_dst = os.path.join(hf_dir, f"norm_states{suffix}.json")
            cfg_src = os.path.join(base, "robot_configs", f"{name}.yaml")
            if not os.path.isfile(cfg_src):
                # custom-training layout: configs sit flat in robot_config_root
                # (agilex_export_0.yaml + norm_states.json style)
                cfg_src = os.path.join(base, f"{name}.yaml")
            stats_src = os.path.join(base, "norm_stats", f"{name}.json")
            if not os.path.isfile(stats_src):
                flat = [os.path.join(base, f"{name}.json")]
                if len(names) == 1:
                    flat.append(os.path.join(base, "norm_states.json"))
                stats_src = next((p for p in flat if os.path.isfile(p)), stats_src)
            if os.path.isfile(cfg_src) and not os.path.exists(cfg_dst):
                cfg = _yaml.safe_load(open(cfg_src, encoding="utf-8")) or {}
                cfg["norm_stats"] = os.path.basename(stats_dst)   # relative sibling
                with open(cfg_dst, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            if os.path.isfile(stats_src) and not os.path.exists(stats_dst):
                _shutil.copyfile(stats_src, stats_dst)   # see publish note: no setattr on fuse targets
        if names:
            print(f"[forge-slim] flattened {len(names)} dataset asset(s) -> {hf_dir}")
        if publish_root:
            hf_published = _forge_publish_hf_ckpt(ckpt_dir, publish_root)
            if hf_published:
                # The local hf_ckpt is only a publish source: drop it once the
                # mount holds the copy, or long runs (20+ save points) pile up
                # ~24G per step inside the pod-local volume budget.
                _shutil.rmtree(hf_dir, ignore_errors=True)
                print(f"[forge-slim] local copy dropped after publish: {hf_dir}")
        if os.environ.get("FORGE_CKPT_KEEP_DCP", "").strip() not in ("", "0"):
            return hf_published
        for name in ("model", "optimizer", "extra_state"):
            p = os.path.join(ckpt_dir, name)
            if os.path.isdir(p):
                _shutil.rmtree(p, ignore_errors=True)
                print(f"[forge-slim] removed dcp part: {p}")
    except Exception as e:
        print(f"[forge-slim] {ckpt_dir} failed (ignored): {e!r}")
    return hf_published


def _forge_slim_all_checkpoints(save_checkpoint_path, cli_yaml_text=None, publish_root=None,
                                 submitted=None):
    """收尾兜底:hf_saver drain 完之后把【本次训练提交过的】save 点再核一遍(异步线程里
    slim 半途失败的补刀),保证 FINISHED 事件发出时本轮产物结构已定盘(publish 模式=同步
    补发挂载侧成品)。

    submitted(hf_saver._submitted_paths)限制只处理本轮真正 checkpoint 过的目录:
    custom-training 复用同一个 FORGE_SAVE_CKPT_PATH 起多轮训练时,目录里会留着往轮
    的 global_step_*(kept local 语义的遗产)——那不是本轮的东西,不许替用户搬上挂载
    (2026-11-03 生产实测:40 步单被 sweep 补发了往轮 60/80/100,并按用户要求收敛)。
    本轮的重名 step 目录由 DCP 覆盖写,旧轮残留由 slim 剥离时自然覆盖,无混读。"""
    try:
        import re as _re
        if not save_checkpoint_path or not os.path.isdir(save_checkpoint_path):
            return
        if submitted is not None:
            todo = sorted(p for p in submitted if os.path.isdir(p))
        else:
            todo = [os.path.join(save_checkpoint_path, d)
                    for d in sorted(os.listdir(save_checkpoint_path))
                    if _re.fullmatch(r"global_step_\d+", d)]
        for ckpt in todo:
            _forge_slim_ckpt_dir(ckpt, cli_yaml_text, publish_root=publish_root)
    except Exception as e:
        print(f"[forge-slim] sweep failed (ignored): {e!r}")


def main():
    args = parse_args(Arguments)
    # Workspace layering (2026-10-29): save_checkpoint_path is DERIVED in
    # TrainArguments.__post_init__ (output_dir/checkpoints) and is NOT a yaml
    # field — the strict parser rejects it as a config key. The launcher hands
    # the pod-local dcp layer over via env; override the derived value here.
    # All 8 ranks apply the same override (each rank writes its own shard).
    _forge_save_override = (os.environ.get("FORGE_SAVE_CKPT_PATH") or "").strip()
    if _forge_save_override:
        args.train.save_checkpoint_path = _forge_save_override
        print(f"[forge] save_checkpoint_path override -> {_forge_save_override}")
    # Flatten-fallback envs (2026-11-04 fix, custom-training gap): on managed
    # orders the launcher writes FORGE_ADAPT_MANIFEST; custom pods have no such
    # manifest, and slim's flatten loop then shipped hf_ckpt WITHOUT
    # robot_config.yaml/norm_states.json (ec80ce86 order: 10 checkpoints all
    # missing the self-contained pair). Feed the same facts straight from the
    # parsed config: dataset names from datasets.txt + the robot_config_root.
    os.environ.setdefault("FORGE_ROBOT_CONFIG_ROOT", str(args.data.robot_config_root or ""))
    try:
        with open(args.data.train_path, encoding="utf-8") as _f:
            _names = [ln.split()[0] for ln in _f if ln.strip() and len(ln.split()) >= 2]
        os.environ.setdefault("FORGE_EXPORT_NAMES", ",".join(_names))
    except (OSError, IndexError):
        pass
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        # 2026-10-23 存储收敛:不再向 output_dir 根写 lingbotvla_cli.yaml(ckpt 内
        # veomni_cli.yaml 才是正式份,根上属于重复)。内容源改在内存生成,由
        # _forge_slim_ckpt_dir 就地写进每个 save 点。
        cli_yaml_text = _forge_train_config_text(args)
        if cli_yaml_text:
            print("[forge-slim] veomni_cli.yaml 内容源就绪(每 ckpt 落盘一份)")
        else:
            cli_yaml_text = None

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    config_kwargs = {**vars(args.model), **vars(args.train)}
    config_registry = get_config_registry()

    config_key = args.model.config_key

    if config_key in config_registry.supported_configs:
        config_cls = config_registry.get_config_cls_from_config_key(config_key)
        config_cls = config_cls(**config_kwargs)
        logger.info_rank0(f"Successfully loaded: {config_cls.__class__.__name__}")
    else:
        config_cls = None
    model = build_foundation_model(
        config_path=args.model.config_path,
        config_cls=config_cls,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
        config_kwargs=config_kwargs,
        moe_implementation=getattr(args.model, 'moe_implementation', None),
    )
    use_depth_align = True if args.train.align_params != {} else False
    use_future_depth = args.train.align_params.get('depth', {}).get('use_future_depth', False)
    use_future_video = use_depth_align and args.train.align_params.get('use_future_video', False)
    depth_model_type = None
    video_teacher = None
    future_video_loss_weight = 1.0
    depth_loss_weight = 1.0
    future_depth_loss_weight = 1.0
    if use_future_video:
        if not args.data.use_future_image:
            raise ValueError("align_params.use_future_video=True requires data.use_future_image=True.")
        video_cfg = args.train.align_params.get("video", {})
        future_video_loss_weight = video_cfg.get(
            "future_video_loss_weight",
            args.train.align_params.get(
                "future_video_loss_weight",
                args.train.align_params.get("depth_loss_weight", 1.0),
            ),
        )
    if use_depth_align:
        depth_model_type = args.train.align_params['depth']['model_type']
        if depth_model_type != 'MoRGBD':
            raise ValueError(f"Only MoRGBD depth distillation is supported, got {depth_model_type!r}.")
        depth_loss_weight = args.train.align_params.get("depth_loss_weight", 1.0)
        future_depth_loss_weight = args.train.align_params.get("future_depth_loss_weight", 1.0)
        print('====Loading Depth Model====')
        moge_model, morgbd_model = build_depth_model(args.train.align_params)
        if args.train.use_compile:
            moge_model = torch.compile(moge_model)
            morgbd_model = torch.compile(morgbd_model)
        if 'visual_dir' not in args.train.align_params or not args.train.align_params['visual_dir']:
            # 2026-10-23 存储收敛:蒸馏视觉缓存是即焚中间物,不进 PVC(原先落
            # output_dir/images 空占目录);FORGE_VISUAL_DIR 可覆盖
            args.train.align_params['visual_dir'] = os.environ.get(
                "FORGE_VISUAL_DIR", "/tmp/forge_align_visuals")
        os.makedirs(args.train.align_params['visual_dir'], exist_ok=True)
        if use_future_video:
            print('====Loading Future Video Model====')
            video_teacher = build_video_model(args.train.align_params['video'])
    from lingbotvla.utils.moe_utils import log_model_param_stats
    log_model_param_stats(model)

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path) # if use build_processor,  tokenizer is processor.tokenizer

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.data.datasets_type == 'vla':
        data_collate_fn.append(VLADataCollatorWithPacking())
    else:
        if args.train.rmpad_with_pos_ids:
            data_collate_fn.append(OmniDataCollatorWithPacking()) # TODO 8.21
        else:
            data_collate_fn.append(OmniDataCollatorWithPadding())
    
    if args.data.dataloader_type == "native":
        if args.data.datasets_type == 'vla':
            args.data.chunk_size = args.train.chunk_size
            train_dataset = build_vla_dataset(dataset_config=args.data, model_config=args.model, config=model.config, processor=processor, use_depth_align=use_depth_align)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))
        
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor if args.data.num_workers > 0 else None,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_fp32=args.train.enable_fp32,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules if args.train.module_fsdp_enable else None,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        fsdp_llm_blocks=False,
        ignore_norm=False,
        use_depth_align=use_depth_align,
        split_fused_experts_from_decoder_fsdp=args.train.split_fused_experts_from_decoder_fsdp,
        vlm_fsdp=args.train.vlm_fsdp,
        use_future_image=args.data.use_future_image,
    )
    logger.info_rank0(model)
    if args.train.use_compile:
        model = torch.compile(model)

    moe_param_groups = get_moe_param_groups(model, args.train)
    if moe_param_groups is not None:
        n_expert = sum(len(g["params"]) for g in moe_param_groups if g["lr"] != args.train.lr)
        group_summary = {f"lr={g['lr']:.2e}": len(g["params"]) for g in moe_param_groups}
        logger.info_rank0(
            f"MoE expert LR scaling enabled: {n_expert} expert param tensors use scaled LR. "
            f"Groups: {group_summary}"
        )
    if args.train.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model,
            args.train,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
        )
        muon_groups, adamw_groups = (
            optimizer.optimizers[0].param_groups,
            optimizer.optimizers[1].param_groups if len(optimizer.optimizers) > 1 else [],
        )
        muon_summary = {f"lr={g['lr']:.2e}": len(g["params"]) for g in muon_groups}
        adamw_summary = {f"lr={g['lr']:.2e}": len(g["params"]) for g in adamw_groups}
        logger.info_rank0(
            f"Muon enabled. Muon groups: {muon_summary}; AdamW (1D/embed) groups: {adamw_summary}"
        )
    else:
        optimizer = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=False,
            optimizer_type=args.train.optimizer,
            post_training=args.model.post_training,
            param_groups=moe_param_groups,
        )

    # Register loss-free load balancing hook (before optimizer.step).
    # The hook also all-reduces and snapshots
    # last_tokens_per_expert (the global load used for monitoring). Setting
    # bias_update_speed=0 makes the bias update a no-op (bias frozen at 0) while
    # keeping the global load monitoring intact.
    if args.train.use_moe:
        _lb_hook = build_moe_load_balance_hook(
            model, coeff=args.train.bias_update_speed, bias_centering=args.train.bias_centering,
            update_interval=args.train.bias_update_interval,
        )
        optimizer.register_step_pre_hook(_lb_hook)
        logger.info_rank0(
            f"Registered MoE load-balance pre-hook (coeff={args.train.bias_update_speed}, "
            f"bias_centering={args.train.bias_centering}, "
            f"update_interval={args.train.bias_update_interval})"
        )

    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    if args.train.max_steps is not None:
        total_train_steps = min(total_train_steps, args.train.max_steps)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        # 2026-11-01 hang root cause: TB on the ossfs output mount stalled a
        # fuse write -> epoch-end flush() blocked rank0 for the whole 600s
        # NCCL watchdog (3/3 runs died at step 1731, run 69d9373d probe
        # verdict). Diagnostics IO must live on the disk-backed local layer —
        # launcher hands over FORGE_TB_LOG_DIR (=ckpt_local/tb); legacy/local
        # setups without that env keep the old output_dir/runs behavior.
        tb_root = os.environ.get("FORGE_TB_LOG_DIR") or args.train.output_dir
        log_dir = os.path.join(tb_root, "runs")
        writer = AsyncTBWriter(log_dir=log_dir)
        if args.train.use_wandb:
            wandb.init(
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        # 2026-10-23 存储收敛:model_assets(processor/config 序列化,eval 提交用)
        # 默认不落 PVC——开环 eval 接通后再开(FORGE_SAVE_MODEL_ASSETS=1)。
        model_assets = [model_config, processor]
        if os.environ.get("FORGE_SAVE_MODEL_ASSETS", "").strip() in ("", "0"):
            print("[forge-slim] model_assets 不落盘(eval 未接通;FORGE_SAVE_MODEL_ASSETS=1 开启)",
                  flush=True)
        else:
            save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    current_epoch_for_eval, current_epoch_step_for_eval = 1, 0
    # Forge 训练事件:rank0 加载;hf_ckpt 真正落盘后由 saver 回调发 CKPT_READY
    forge_events = _load_forge_events() if args.train.global_rank == 0 else None

    save_checkpoint_path = None
    hf_failure_log_path = (
        os.path.join(args.train.save_checkpoint_path, "async_hf_failures.jsonl")
        if args.train.save_checkpoint_path
        else None
    )

    def _forge_ckpt_ready(result):
        # saver 转换线程里调用(events.py 线程安全);失败/未产出不发 READY。
        # slim 不依赖事件配置(forge_events None 时也要做,故 on_result 无条件挂)。
        # 2026-10-29 layering: slim/publish BEFORE the emit, so consumers see
        # the mount-side path and the artifact is already there when READY lands.
        if result.hf_success and result.hf_path:
            pub_root = (os.environ.get("FORGE_PUBLISH_CKPT_ROOT", "") or "").strip()
            published = _forge_slim_ckpt_dir(result.checkpoint_path, cli_yaml_text,
                                             publish_root=pub_root or None)
            ckpt_path = os.path.dirname(published) if published else result.checkpoint_path
            hf_path = published or result.hf_path
            _forge_emit(forge_events, "emit_ckpt_ready", result.global_step, hf_path,
                        checkpointPath=ckpt_path)

    hf_saver = AsyncHFCheckpointSaver(
        enabled=args.train.async_save_hf_weights,
        max_pending=args.train.async_hf_max_pending,
        logger=logger,
        failure_log_path=hf_failure_log_path,
        eval_args=args.eval,
        on_result=_forge_ckpt_ready,
    )

    def save_hf_checkpoint_best_effort(
        checkpoint_path: str | None,
        checkpoint_state: Dict[str, Any],
        step: int,
        epoch: int | None = None,
        epoch_step: int | None = None,
    ) -> None:
        if args.train.global_rank != 0:
            return
        if not args.train.save_hf_weights or checkpoint_path is None:
            return
        hf_saver.submit(
            global_step=step,
            save_checkpoint_path=checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
            save_ema=checkpoint_state.get("ema") is not None,
            enable_fp32=args.train.enable_fp32,
            model_assets=model_assets,
            epoch=epoch,
            epoch_step=epoch_step,
        )

    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    load_checkpoint_path = None
    candidates = []
    if args.train.load_checkpoint_path or args.train.enable_resume:
        if args.train.load_checkpoint_path:
            load_checkpoint_path = args.train.load_checkpoint_path
            candidates = [load_checkpoint_path]
        elif args.train.enable_resume:
            checkpoint_dir = f'{args.train.output_dir}/checkpoints'
            if os.path.exists(checkpoint_dir):
                pattern = re.compile(r"global_step_(\d+)")
                tmp = []
                for dirname in os.listdir(checkpoint_dir):
                    match = pattern.fullmatch(dirname)
                    if match:
                        step = int(match.group(1))
                        tmp.append((step, os.path.join(checkpoint_dir, dirname)))
                tmp.sort(key=lambda x: x[0], reverse=True)
                candidates = [p for _, p in tmp]
            if candidates:
                load_checkpoint_path = candidates[0]
            else:
                logger.info_rank0(f"No checkpoints in {args.train.output_dir} now!")
    if candidates:
        last_err = None
        loaded = False
        for cp in candidates:
            state = {"model": model, "ema": None, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            try:
                Checkpointer.load(cp, state, allow_partial_load=getattr(args.train, 'allow_partial_checkpoint', False))
                global_step = state["extra_state"]["global_step"]
                start_epoch = global_step // args.train.train_steps
                start_step = global_step % args.train.train_steps
                lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                if start_step > 0 and args.train.resume_dataloader_state:
                    train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
                environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])
                if start_step == 0:  # resume at the end of epoch
                    iter(train_dataloader)  # clear resume state and prefetch data
                dist.barrier()
                logger.info_rank0(f"Load distributed checkpoint from {cp} successfully!")
                loaded = True
                break
            except Exception as e:
                last_err = e
                logger.info_rank0(f"Failed to load checkpoint {cp}: {repr(e)}. Trying older one...")
                continue
        if not loaded:
            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")
    else:
        logger.info_rank0("Starting training from scratch.")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    # Contract fields for frontend step/epoch conversion. train_steps equals
    # steps-per-epoch after compute_train_steps; report effective epoch count
    # (trainingSteps / steps_per_epoch), not the framework's default cap.
    _forge_spe = int(args.train.train_steps) or 1
    _forge_total_epochs = max(
        1, -(-int(args.train.max_steps or _forge_spe) // _forge_spe))
    _forge_emit(forge_events, "emit_started",
                maxSteps=args.train.max_steps or args.train.train_steps,
                worldSize=args.train.world_size,
                globalBatch=args.train.global_batch_size,
                stepsPerEpoch=args.train.train_steps,
                totalEpochs=_forge_total_epochs,
                outputDir=args.train.output_dir)
    # Forge 心跳守护线程(rank0):每 N 秒一条 RUNNING 信号——消息形态=TRAIN_METRICS
    # 空 metrics,current_step 取当前步。消费端"有消息=进程活着",与步进事件解耦;
    # 后端消费代码零改动(曲线画空 metrics 自然无点)。间隔 env FORGE_EVENT_HEARTBEAT_SEC
    # 默认 60,0=关;终态前由 _forge_hb_stop 停止,防止 FINISHED 后再冒 RUNNING。
    _forge_hb_stop = threading.Event()

    def _forge_heartbeat(_pub, _step_box):
        try:
            interval = max(0, int(os.environ.get("FORGE_EVENT_HEARTBEAT_SEC", "60")))
        except ValueError:
            interval = 60
        if _pub is None or interval == 0:
            return
        while not _forge_hb_stop.wait(interval):
            # Liveness only: no metrics fields polluting the loss/throughput series.
            _forge_emit(_pub, "emit_heartbeat", _step_box[0])

    if forge_events is not None:
        _forge_step_box = [0]
        threading.Thread(target=_forge_heartbeat, args=(forge_events, _forge_step_box),
                         daemon=True, name="forge-heartbeat").start()
    # create the path in advance to save loss log
    if args.train.global_rank == 0:
        os.makedirs(args.train.save_checkpoint_path, exist_ok=True)
    reached_max_steps = False
    max_steps_driven = (
        args.train.max_steps is not None
        and args.train.max_steps < args.train.train_steps * args.train.num_train_epochs
    )
    if max_steps_driven:
        data_loader_tqdm = trange(
            args.train.max_steps,
            bar_format="Step: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            initial=global_step,
            disable=args.train.local_rank != 0,
        )
    def _forge_epoch_probe(tag: str) -> None:
        # 2026-10-31 epoch-boundary forensics: two prod orders (84037b10/6dc3cd43)
        # froze for exactly the 600s NCCL watchdog at the SAME point — the epoch
        # 3->4 boundary (global 1731 = 3x577), zero NCCL/CUDA/OOM lines. Probe
        # worker/shm state at every epoch boundary so the next repro answers
        # whether we stall before set_epoch, between iter() and the probe, or
        # inside worker respawn.
        if args.train.global_rank != 0:
            return
        try:
            shm = os.popen("df -h /dev/shm 2>/dev/null | tail -1").read().strip()
            top = os.popen("ps -eo rss,comm --no-headers | sort -rnk1 | head -3").read()
            print(f"[epoch-probe] {tag} epoch-loop | shm[{shm}] | top-rss:\n{top}", flush=True)
        except Exception as e:
            print(f"[epoch-probe] {tag} failed: {e!r}", flush=True)

    def _forge_rank_probe(tag: str) -> None:
        # 2026-11-01 run #3 (forge-train-1ebff0e2, FORGE_NUM_WORKERS=4): worker
        # count did NOT change the freeze — same step 1731, same SeqNum 592022,
        # head probes for epochs 1/2/3 normal, epoch-4 "before" never printed.
        # So ALL ranks are stuck between the last collective of step 1731 and
        # the epoch-4 head: the post-inner-loop epoch tail (tqdm.close / writer
        # flush / print_device_mem_info) is the blind spot. Every rank prints a
        # one-line marker through the tail — the straggler rank is the one whose
        # marker is missing when the freeze hits.
        try:
            print(f"[rank-probe] r{args.train.global_rank} gstep={global_step} {tag}", flush=True)
        except Exception as e:
            print(f"[rank-probe] {tag} failed: {e!r}", flush=True)

    def _forge_async_tb_flush() -> None:
        # 2026-11-01 run #4 (69d9373d) VERDICT: rank0 froze exactly inside the
        # epoch-tail writer.flush() — r1..r7 printed every tail marker incl.
        # epoch-head of epoch 4, r0 stopped at tqdm-closed. TB runs/ lives on
        # the ossfs output mount, so a hung sync flush on the fuse path held
        # all 8 ranks hostage (7 ranks waited in the first forward allgather
        # of step 1732 -> 600s NCCL watchdog killed the whole job, 3/3 runs).
        # Diagnostics must never gate training: flush in a daemon thread and
        # drop it on the floor if the fuse path stalls.
        if args.train.global_rank != 0:
            return
        t = threading.Thread(target=_tb_flush_worker, daemon=True)
        t.start()

    def _tb_flush_worker() -> None:
        _forge_rank_probe("tb-flush-start")
        try:
            writer.flush()
            _forge_rank_probe("tb-flush-done")
        except Exception as e:
            _forge_rank_probe(f"tb-flush-failed: {e!r}")

    for epoch in range(start_epoch, args.train.num_train_epochs):
        _forge_epoch_probe("before")
        _forge_rank_probe("epoch-head")
        current_epoch_for_eval = epoch + 1
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        if not max_steps_driven:
            data_loader_tqdm = trange(
                args.train.train_steps,
                desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
                total=args.train.train_steps,
                initial=start_step,
                disable=args.train.local_rank != 0,
            )
        data_iterator = iter(train_dataloader)
        _forge_epoch_probe("iter-built")
        for epoch_step in range(start_step, args.train.train_steps):
            current_epoch_step_for_eval = epoch_step + 1
            global_step += 1
            try:
                _forge_step_box[0] = global_step   # 心跳线程读当前步(闭包内 box)
            except NameError:
                pass
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss, total_vla_loss, total_depth_loss, total_future_depth_loss, total_future_video_loss, total_seq_wise_loss, total_router_z_loss = 0, 0, 0, 0, 0, 0, 0
            depth_targets, depth_preds = None, None
            future_depth_targets, future_depth_preds = None, None
            future_video_targets, future_video_preds = None, None
            future_video_current_preds = None
            future_video_cls_targets = None
            future_video_current_dino = None
            future_video_current_rgb, future_video_target_rgb = None, None
            ignore_batch_num = 0
            torch.cuda.synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                future_video_targets = None
                future_video_current_preds = None
                future_video_cls_targets = None
                future_video_current_dino = None
                dataset_names = micro_batch.pop('rep_id', None)
                environ_meter.add(micro_batch)
                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                future_video_effective_fps = micro_batch.pop('future_video_effective_fps', None)
                depth_forward_time = 0
                if use_depth_align:
                    with torch.no_grad():
                        depth_start_time = time.time()
                        pil_images = micro_batch.pop('pil_images', None)
                        future_pil_images = micro_batch.pop('future_pil_images', None) if (use_future_depth or use_future_video) else None
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            depth_targets, cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), pil_images)
                            if use_future_depth:
                                future_depth_targets, future_cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), future_pil_images)
                        if use_future_video:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                future_video_target_bundle = get_video_target(
                                    video_teacher,
                                    pil_images,
                                    future_pil_images,
                                    args.train.align_params['video'],
                                    effective_fps=future_video_effective_fps,
                                )
                                if isinstance(future_video_target_bundle, dict):
                                    future_video_targets = future_video_target_bundle["patch"]
                                    future_video_cls_targets = future_video_target_bundle.get("cls")
                                    future_video_current_dino = future_video_target_bundle.get("current_patch")
                                elif isinstance(future_video_target_bundle, tuple):
                                    future_video_targets, future_video_cls_targets = future_video_target_bundle
                                else:
                                    future_video_targets = future_video_target_bundle
                            future_video_current_rgb = pil_images
                            future_video_target_rgb = future_pil_images
                        depth_forward_time = time.time() - depth_start_time

                with model_fwd_context:
                    model_outputs = model(
                        **micro_batch,
                        depth_targets=depth_targets,
                        future_depth_targets=future_depth_targets,
                        future_video_targets=future_video_targets,
                        future_video_cls_targets=future_video_cls_targets,
                        future_video_current_patch=future_video_current_dino,
                    )
                    if len(model_outputs) == 6:
                        loss, vla_loss, depth_loss, seq_wise_loss, loss_log, depth_preds = model_outputs
                        future_depth_loss = 0
                        future_video_loss = 0
                        future_depth_preds = None
                        future_video_preds = None
                        future_video_current_preds = None
                    elif len(model_outputs) == 8:
                        loss, vla_loss, depth_loss, future_depth_loss, seq_wise_loss, loss_log, depth_preds, future_depth_preds = model_outputs
                        future_video_loss = 0
                        future_video_preds = None
                        future_video_current_preds = None
                    elif len(model_outputs) == 10:
                        loss, vla_loss, depth_loss, future_depth_loss, future_video_loss, seq_wise_loss, loss_log, depth_preds, future_depth_preds, future_video_preds = model_outputs
                        future_video_current_preds = None
                    elif len(model_outputs) == 11:
                        loss, vla_loss, depth_loss, future_depth_loss, future_video_loss, seq_wise_loss, loss_log, depth_preds, future_depth_preds, future_video_preds, future_video_current_preds = model_outputs
                    else:
                        raise ValueError(f"Unexpected model output length: {len(model_outputs)}")

                    loss = loss / len(micro_batches)
                    vla_loss = vla_loss / len(micro_batches)
                    depth_loss = depth_loss / len(micro_batches)
                    future_depth_loss = future_depth_loss / len(micro_batches)
                    future_video_loss = future_video_loss / len(micro_batches)
                    seq_wise_loss = seq_wise_loss / len(micro_batches)
                    router_z_loss = loss_log.get("router_z_loss", loss_log.get("moe_zloss/weighted", 0))
                    avg_lang_length = micro_batch['lang_masks'].sum(dim=-1).float().mean()

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                total_vla_loss += vla_loss.item()
                if not (isinstance(depth_loss, int) or isinstance(depth_loss, float)):
                    total_depth_loss += depth_loss.item()
                if not (isinstance(future_depth_loss, int) or isinstance(future_depth_loss, float)):
                    total_future_depth_loss += future_depth_loss.item()
                if not (isinstance(future_video_loss, int) or isinstance(future_video_loss, float)):
                    total_future_video_loss += future_video_loss.item()
                if not (isinstance(seq_wise_loss, int) or isinstance(seq_wise_loss, float)):
                    total_seq_wise_loss += seq_wise_loss.item()
                if not (isinstance(router_z_loss, int) or isinstance(router_z_loss, float)):
                    total_router_z_loss += router_z_loss.item() / len(micro_batches)
                del micro_batch
            # --- TEMP gate-gradient probe (GATE_GRAD_PROBE=1): dump router gate grad
            # magnitude vs routed-expert grad, plus per-expert symmetry, pre-clip. ---
            if os.environ.get("GATE_GRAD_PROBE") and global_step < 12:
                from torch.distributed.tensor import DTensor
                def _full(t):
                    return t.full_tensor() if isinstance(t, DTensor) else t
                _moe_blocks = [m for m in model.modules()
                               if type(m).__name__ == "Qwen2TokenMoeBlock"]
                _lines = []
                for _li, _blk in enumerate(_moe_blocks):
                    _gw = _blk.gate.weight
                    _gg = _gw.grad
                    _gw_f = _full(_gw.detach())
                    _gg_f = _full(_gg.detach()) if _gg is not None else None
                    # routed-expert reference grad (experts[0].gate_proj), collective on all ranks
                    _eg_f = None
                    if hasattr(_blk.experts, "__getitem__"):
                        _ep = _blk.experts[0].gate_proj.weight
                        if _ep.grad is not None:
                            _eg_f = _full(_ep.grad.detach())
                    if args.train.local_rank == 0:
                        if _gg_f is None:
                            _lines.append(f"L{_li}: gate.grad=None rg={_gw.requires_grad}")
                        else:
                            _wn = _gw_f.norm().item(); _gn = _gg_f.norm().item()
                            _row = _gg_f.norm(dim=1)
                            _eref = f" exp|g|={_eg_f.norm().item():.2e}" if _eg_f is not None else ""
                            _lines.append(
                                f"L{_li}: |g|={_gn:.2e} |w|={_wn:.2e} g/w={_gn/(_wn+1e-9):.1e} "
                                f"rowμ={_row.mean():.2e} rowσ={_row.std():.2e} "
                                f"[{_row.min():.1e},{_row.max():.1e}]{_eref}")
                if args.train.local_rank == 0:
                    print(f"[GATE_GRAD_PROBE step={global_step}]\n  " + "\n  ".join(_lines), flush=True)
            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
            else:
                max_grad_norm = args.train.max_grad_norm
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(max_grad_norm).item()
            elif hasattr(model, '_ep_param_set'):
                from lingbotvla.distributed.fsdp2.clip_grad_norm import ep_fsdp2_clip_grad_norm
                grad_norm = ep_fsdp2_clip_grad_norm(model, max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, total_vla_loss, total_depth_loss, total_future_depth_loss, total_future_video_loss, total_seq_wise_loss, total_router_z_loss, avg_lang_length, grad_norm = all_reduce((total_loss, total_vla_loss, total_depth_loss, total_future_depth_loss, total_future_video_loss, total_seq_wise_loss, total_router_z_loss, avg_lang_length, grad_norm), group=get_parallel_state().fsdp_group)
            total_depth_loss = total_depth_loss / depth_loss_weight
            total_future_depth_loss = total_future_depth_loss / future_depth_loss_weight
            total_future_video_loss = total_future_video_loss / future_video_loss_weight
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            all_lrs = lr_scheduler.get_last_lr()
            if args.train.use_moe_expert_lr and len(all_lrs) > 1:
                lr = min(all_lrs)          # base (non-expert) lr
                expert_lr = max(all_lrs)   # routed-expert lr
            else:
                lr = max(all_lrs)
                expert_lr = None
            # global step only: the second counter used to be per-epoch steps
            # (dataloader size), which read as a confusing second denominator
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            data_loader_tqdm.update()
            expert_lr_str = f"Expert_LR {expert_lr:.2e}, " if expert_lr is not None else ""
            maxvio_val = loss_log.get("moe_summary/maxvio_avg", loss_log.get("token_moe/avg_maxvio", None))
            maxvio_str = f"MaxVio {maxvio_val.item() if torch.is_tensor(maxvio_val) else maxvio_val:.4f}, " if maxvio_val is not None else ""
            sigmoid_val = loss_log.get("moe_summary/topk_sigmoid_avg_rank0", loss_log.get("token_moe/avg_topk_sigmoid", None))
            sigmoid_str = f"AvgSigmoid {sigmoid_val.item() if torch.is_tensor(sigmoid_val) else sigmoid_val:.4f}, " if sigmoid_val is not None else ""
            logger.info_rank0(
                f"Step {global_step}, "
                f"Epoch {epoch+1}, "
                f"Loss {total_loss:.4f}, "
                f"VLA_Loss {total_vla_loss:.4f}, "
                f"Depth_Loss {total_depth_loss:.4f}, "
                f"Future_Depth_Loss {total_future_depth_loss:.4f}, "
                f"FutureVideo_Loss {total_future_video_loss:.4f}, "
                f"SeqWise_Loss {total_seq_wise_loss:.4f}, "
                f"RouterZ_Loss {total_router_z_loss:.4f}, "
                f"{maxvio_str}"
                f"{sigmoid_str}"
                f"GradNorm {grad_norm:.4f}, "
                f"LR {lr:.2e}, "
                f"{expert_lr_str}"
                f"StepTime {delta_time:.3f}s, "
                f"Depth_Forward_Time {depth_forward_time: .3f}s, "
                f"Ignore_Batch_Num {ignore_batch_num}"
            )


            if args.train.global_rank == 0:
                writer.add_scalar("training/loss", total_loss, global_step)
                writer.add_scalar("training/vla_loss", total_vla_loss, global_step)
                writer.add_scalar("training/depth_loss", total_depth_loss, global_step)
                writer.add_scalar("training/future_depth_loss", total_future_depth_loss, global_step)
                writer.add_scalar("training/future_video_loss", total_future_video_loss, global_step)
                writer.add_scalar("training/sequence_wise_loss", total_seq_wise_loss, global_step)
                writer.add_scalar("training/router_z_loss", total_router_z_loss, global_step)
                # MoE monitoring metrics.
                #   moe_summary/*            -> every step (cheap cross-layer health glance)
                #   moe_<metric>/layerXX     -> every moe_monitor_interval steps (per-layer, downsampled)
                moe_interval = max(1, args.train.moe_monitor_interval)
                log_moe_perlayer = (global_step % moe_interval == 0)
                moe_perlayer_prefixes = (
                    "moe_maxvio/", "moe_minvio/", "moe_minload/",
                    "moe_entropy_rank0/", "moe_topksigmoid_rank0/", "moe_bias/",
                    "moe_seqwise/layer",   # per-layer seq-wise loss -> downsampled
                    "moe_zloss/layer",     # per-layer raw router z-loss -> downsampled
                )

                def _tb_scalar(value):
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            return None
                        return value.detach().float().item()
                    return value

                for key, value in loss_log.items():
                    # every step: cross-layer summaries + seq-wise average + legacy V1 keys
                    if (key.startswith("moe_summary/")
                            or key in ("moe_seqwise/avg", "moe_zloss/avg_raw", "moe_zloss/weighted")
                            or key.startswith("token_moe/")):
                        scalar = _tb_scalar(value)
                        if scalar is not None:
                            writer.add_scalar(key, scalar, global_step)
                    elif key.startswith(moe_perlayer_prefixes) and log_moe_perlayer:
                        scalar = _tb_scalar(value)
                        if scalar is not None:
                            writer.add_scalar(key, scalar, global_step)
                    elif key.startswith("align/"):
                        scalar = _tb_scalar(value)
                        if scalar is not None:
                            writer.add_scalar(key, scalar, global_step)
                align_training_aliases = {
                    "align/current_video_loss": "training/current_video_loss",
                    "align/current_video_loss_weighted": "training/current_video_loss_weighted",
                    "align/future_video_loss": "training/future_video_patch_loss",
                    "align/future_video_loss_weighted": "training/future_video_patch_loss_weighted",
                    "align/video_loss": "training/video_loss",
                    "align/video_loss_weighted": "training/video_loss_weighted",
                }
                for src_key, dst_key in align_training_aliases.items():
                    if src_key in loss_log:
                        scalar = _tb_scalar(loss_log[src_key])
                        if scalar is not None:
                            writer.add_scalar(dst_key, scalar, global_step)
                # MoE expert-selection monitoring (per layer, every moe_monitor_interval):
                #   moe_expert_selection/      -> original add_histogram (Histograms tab; bins
                #                                 expert IDs, edges look inflated -- kept as-is).
                #   moe_expert_selection_bar/  -> exact per-expert bar chart (Images tab; one bar =
                #                                 that expert's true token count, no binning artifact).
                #                                 Aligned with VideoPretrain's bar chart.
                # Plus a global (summed-over-layers) bar + load CV (std/mean) imbalance scalar.
                if log_moe_perlayer:
                    if "_token_moe_expert_counts" in loss_log:
                        total_counts = None
                        for layer_id, counts in loss_log["_token_moe_expert_counts"]:
                            c = counts.detach().float().cpu()
                            writer.add_histogram_from_counts(
                                f"moe_expert_selection/layer{layer_id:02d}", c, global_step,
                            )
                            writer.add_expert_bar(
                                f"moe_expert_selection_bar/layer{layer_id:02d}", c, global_step,
                                title=f"Expert load layer{layer_id:02d} (step {global_step})",
                            )
                            total_counts = c.clone() if total_counts is None else total_counts + c
                        if total_counts is not None:
                            writer.add_expert_bar(
                                "moe_expert_load_bar", total_counts, global_step,
                                title=f"Global expert load, all layers (step {global_step})",
                            )
                            mean = total_counts.mean()
                            load_cv = (total_counts.std(unbiased=False) / (mean + 1e-9)).item()
                            writer.add_scalar("moe_summary/load_cv", load_cv, global_step)
                writer.add_scalar("training/grad_norm", grad_norm, global_step)
                writer.add_scalar("training/lr", lr, global_step)
                if expert_lr is not None:
                    writer.add_scalar("training/expert_lr", expert_lr, global_step)
                writer.add_scalar("training/avg_lang_length", avg_lang_length, global_step)
                writer.add_scalar("training/max_norm_batch", ignore_batch_num, global_step)
                writer.add_scalar("steptime", delta_time, global_step)
                # Forge TRAIN_METRICS(=TB 信息):TB 每步全量,这里按 metrics_every 抽样;
                # scalars 与上面 writer.add_scalar 同源(训练循环内已是 reduce 后的标量)。
                if (forge_events is not None and forge_events.metrics_every
                        and global_step % forge_events.metrics_every == 0):
                    _forge_emit(forge_events, "emit_metrics", global_step, {
                        "training/loss": total_loss,
                        "training/vla_loss": total_vla_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                        "steptime": delta_time,
                    })
                # we only log the last mini batch if grad acc is activated
                if dataset_names is not None and 'batch_mean_losses' in loss_log:
                    batch_mean_losses = loss_log['batch_mean_losses']  # shape (B,)
                    if hasattr(batch_mean_losses, "detach"):
                        batch_mean_losses = batch_mean_losses.detach().cpu()

                    group_losses = defaultdict(list)
                    for name, loss_value in zip(dataset_names, batch_mean_losses):
                        group_losses[name].append(loss_value.item() if hasattr(loss_value, "item") else float(loss_value))

                    for name, values in group_losses.items():
                        mean_loss = sum(values) / len(values)
                        writer.add_scalar(f"detailed_loss/{name}", mean_loss, global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
                        helper.upload_trace(
                            args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir
                        )

                # loss.jsonl disabled to avoid NFS write blocking rank 0
                # loss_record = {
                #     "step": global_step,
                #     "epoch": epoch + 1,
                #     "loss": total_loss,
                #     "grad_norm": grad_norm,
                #     "lr": lr,
                #     "step_time": delta_time
                # }
                # loss_file_path = os.path.join(args.train.save_checkpoint_path, "loss.jsonl")
                # try:
                #     with open(loss_file_path, "a", encoding="utf-8") as f:
                #         f.write(json.dumps(loss_record, ensure_ascii=False) + "\n")
                # except Exception as e:
                #     logger.info_rank0(f"⚠️ Failed to write loss.jsonl: {e}")

                if use_depth_align:
                    if global_step % args.train.align_params['visual_steps'] == 0:
                        with torch.no_grad():
                            if use_future_video:
                                log_video(
                                    future_video_preds,
                                    future_video_targets,
                                    steps=global_step,
                                    config=args.train.align_params,
                                    is_future=True,
                                    current_rgb_images=future_video_current_rgb,
                                    target_rgb_images=future_video_target_rgb,
                                    current_target_feats=future_video_current_dino,
                                    current_pred_feats=future_video_current_preds,
                                )

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                
                # param_to_name = {}
                # for name, param in model.named_parameters():
                #     param_to_name[name] = param
                # for group in optimizer.state.values():
                #     for param in group:
                #         if param in param_to_name:
                #             print(param_to_name[param])
                #         else:
                #             print("⚠️ Unidentified parameter in optimizer state")

                state = {
                    "model": model,
                    "ema": None,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                if args.train.global_rank == 0:
                    _forge_async_tb_flush()
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                save_hf_checkpoint_best_effort(
                    save_checkpoint_path,
                    state,
                    global_step,
                    current_epoch_for_eval,
                    current_epoch_step_for_eval,
                )
                # Forge:dcp 滚动清理——保最近 FORGE_KEEP_DCP(默认 2)套,删更旧的
                # model/optimizer/extra_state(hf_ckpt 不动)。长跑防 shm/cpfs 被 dcp
                # 堆顶穿(2026-10-13 8881717e 单 24000 步跑到 3000 Step 时 OOM 实测);
                # enableResume 只需最新一套 dcp,保 2 是双保险。
                if args.train.global_rank == 0:
                    _forge_prune_old_dcp(args.train.save_checkpoint_path)

            if args.train.max_steps is not None and global_step >= args.train.max_steps:
                logger.info_rank0(f"Reached max_steps={args.train.max_steps}, stopping training.")
                reached_max_steps = True
                break

        _forge_rank_probe("inner-loop-exit")
        if not max_steps_driven:
            data_loader_tqdm.close()
        _forge_rank_probe("tqdm-closed")
        _forge_async_tb_flush()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        _forge_rank_probe("epoch-tail-done")
        if reached_max_steps:
            already_saved = args.train.save_steps and global_step % args.train.save_steps == 0
            if not already_saved:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "ema": None,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                save_hf_checkpoint_best_effort(
                    save_checkpoint_path,
                    state,
                    global_step,
                    current_epoch_for_eval,
                    current_epoch_step_for_eval,
                )
            break
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "ema": None,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            save_hf_checkpoint_best_effort(
                save_checkpoint_path,
                state,
                global_step,
                current_epoch_for_eval,
                current_epoch_step_for_eval,
            )

    if max_steps_driven:
        data_loader_tqdm.close()
    if args.train.global_rank == 0:
        writer.close()
        # TB now lives on the local layer — archive it back onto the output
        # mount so successful runs keep their curves. Bounded: a stalled fuse
        # copy must never hang shutdown; the join gives up after the timeout
        # and the archive is simply lost (diagnostics only).
        if os.environ.get("FORGE_TB_LOG_DIR"):
            _t = threading.Thread(
                target=shutil.copytree,
                args=(log_dir, os.path.join(args.train.output_dir, "runs")),
                kwargs={"dirs_exist_ok": True}, daemon=True)
            _t.start()
            _t.join(timeout=300)
            if _t.is_alive():
                print(f"[forge-tb] runs archive copy still running after 300s, "
                      "giving up (local copy stays in ckpt_local)", flush=True)
    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # Ensure the last checkpoint has an HF conversion scheduled, then wait for async work.
    if save_checkpoint_path is not None:
        save_hf_checkpoint_best_effort(
            save_checkpoint_path,
            state,
            global_step,
            current_epoch_for_eval,
            current_epoch_step_for_eval,
        )
    hf_saver.wait_all_across_ranks()

    # Forge 收尾兜底:saver 回调线程里 slim 失败的补刀,FINISHED 发出前 ckpt 结构定盘。
    # 传入 saver 登记表:sweep 只核本轮真正 save 过的点(见函数 docstring,防跨轮补发)。
    if args.train.global_rank == 0:
        _forge_slim_all_checkpoints(args.train.save_checkpoint_path, cli_yaml_text,
                                    publish_root=(os.environ.get("FORGE_PUBLISH_CKPT_ROOT", "") or "").strip() or None,
                                    submitted=set(hf_saver._submitted_paths))

    # Forge TRAIN_FINISHED(成功路径;崩溃路径由 __main__ 的 except 发 FAILED;
    # SIGKILL 类外部强杀发不出事件,由平台侧 rc 状态兜底)
    if args.train.global_rank == 0:
        _forge_hb_stop.set()                      # 终态前停心跳,防 FINISHED 后再冒 RUNNING
        _forge_emit(forge_events, "emit_finished", "SUCCEEDED",
                    checkpointDir=str(args.train.output_dir))
        if forge_events is not None:
            try:
                forge_events.close()
            except Exception:
                pass

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as _exc:
        # Forge TRAIN_FINISHED(FAILED):异常退出也要让外界知道。
        # 只让 torchrun 的 rank0 发,否则 8 个 rank 各发一条。
        if os.environ.get("RANK", "0") == "0":
            _pub = _load_forge_events()
            try:
                if _pub is not None:
                    _pub.emit_finished("FAILED", error=repr(_exc))
                    _pub.close()
            except Exception:
                pass
        raise
