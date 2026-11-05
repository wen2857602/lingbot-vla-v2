import json
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, List, Sequence

import torch
import torch.distributed as dist

from lingbotvla.checkpoint import ckpt_to_state_dict
from lingbotvla.models import save_model_weights
from lingbotvla.utils import helper


def _log(logger: Any, level: str, message: str, *args: Any) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None) or getattr(logger, "info", None)
    if log_fn is not None:
        log_fn(message, *args)


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


@dataclass
class HFCheckpointResult:
    global_step: int
    checkpoint_path: str
    hf_path: str
    epoch: int | None = None
    epoch_step: int | None = None
    ema_hf_path: str = ""
    hf_success: bool = False
    ema_success: bool = False
    eval_submitted: bool = False
    eval_skipped: bool = False
    eval_record_id: str = ""
    eval_skip_reason: str = ""
    error: str = ""
    traceback: str = ""
    elapsed_sec: float = 0.0


class AsyncHFCheckpointSaver:
    """Rank0 HF checkpoint conversion with best-effort async mode."""

    def __init__(
        self,
        enabled: bool,
        max_pending: int = 1,
        logger: Any = None,
        failure_log_path: str | None = None,
        eval_args: Any = None,
        on_result: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_pending = max(1, int(max_pending or 1))
        self.logger = logger
        self.failure_log_path = failure_log_path
        # Forge 训练事件(CKPT_READY 等):hf_ckpt 真正落盘(或 best-effort 失败)后回调,
        # 入参 HFCheckpointResult。回调自身必须 best-effort——这里只 try/except 包死,
        # 不让事件通道问题影响保存主流程。
        self.on_result = on_result
        self._executor = ThreadPoolExecutor(max_workers=1) if self.enabled and _is_rank0() else None
        self._futures: List[tuple[str, Future]] = []
        self._results: List[HFCheckpointResult] = []
        self._submitted_paths: set[str] = set()
        # notify-once bookkeeping (2026-10-31 fix): the done-callback below and
        # the drain paths both funnel into _record_future_result; _notified
        # guarantees on_result fires exactly once per submitted checkpoint.
        self._lock = threading.Lock()
        self._notified: set[str] = set()

    def submit(
        self,
        global_step: int,
        save_checkpoint_path: str | None,
        output_dir: str,
        ckpt_manager: str,
        save_ema: bool,
        enable_fp32: bool,
        model_assets: Sequence[Any] | None,
        epoch: int | None = None,
        epoch_step: int | None = None,
    ) -> None:
        if save_checkpoint_path is None or not _is_rank0():
            return

        checkpoint_path = os.path.abspath(str(save_checkpoint_path))
        output_dir = str(output_dir)
        ckpt_manager = str(ckpt_manager)
        model_assets_snapshot = tuple(model_assets) if model_assets is not None else None

        # 去重必须两条分支都生效(2026-10-23 真实单 forge-train-92370e10 实锤修复):
        # 训练循环 save 点与末段"保证最后一个 ckpt 已转换"的 final submit 是同一
        # 个 checkpoint_path。此前 sync 分支(async_save_hf_weights=False,即默认)
        # 既不查重也不登记 _submitted_paths → final 对已转换完、且 slim 已剥掉
        # model/optimizer/extra_state 的 ckpt 二次 dcp.load → FileNotFoundError
        # → best_effort=False 直接 raise,40/40 跑完的训练崩在谢幕。
        if checkpoint_path in self._submitted_paths:
            _log(self.logger, "info", "[async_hf] skip duplicate checkpoint %s", checkpoint_path)
            return
        if self.enabled:
            self._drain_finished(block=False)
            while self._pending_count() >= self.max_pending:
                _log(
                    self.logger,
                    "info",
                    "[async_hf] max pending reached, waiting for oldest HF save before scheduling step %s",
                    global_step,
                )
                self._drain_oldest()
            future = self._executor.submit(
                self._run_hf_checkpoint,
                global_step,
                checkpoint_path,
                output_dir,
                ckpt_manager,
                save_ema,
                enable_fp32,
                model_assets_snapshot,
                epoch,
                epoch_step,
                True,
            )
            self._futures.append((checkpoint_path, future))
            self._submitted_paths.add(checkpoint_path)
            # 2026-10-31 fix: conversion-completion must publish/emit IMMEDIATELY.
            # The old flow only notified at the next drain point (next save or
            # training end) — a long run dying between save points (prod order
            # 84037b10: NCCL hang at step 1731, save=1000) never published its
            # already-finished ckpt and CKPT_READY was lost with the pod.
            future.add_done_callback(lambda f, _p=checkpoint_path: self._future_done(_p, f))
            _log(self.logger, "info", "[async_hf] scheduled HF checkpoint for %s", checkpoint_path)
        else:
            result = self._run_hf_checkpoint(
                global_step=global_step,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                ckpt_manager=ckpt_manager,
                save_ema=save_ema,
                enable_fp32=enable_fp32,
                model_assets=model_assets_snapshot,
                epoch=epoch,
                epoch_step=epoch_step,
                best_effort=False,
            )
            self._submitted_paths.add(checkpoint_path)
            self._results.append(result)
            self._notify_result(result)
            if result.error:
                self._write_failure(result)

    def wait_all_best_effort(self) -> dict[str, Any]:
        if not _is_rank0():
            return {}

        for _, future in list(self._futures):
            self._record_future_result(future)
        self._futures.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        summary = self.summary()
        _log(self.logger, "info", "[async_hf] summary: %s", json.dumps(summary, ensure_ascii=False))
        return summary

    def wait_all_across_ranks(self) -> dict[str, Any]:
        summary = self.wait_all_best_effort() if _is_rank0() else {}
        if dist.is_available() and dist.is_initialized():
            payload = [summary]
            dist.broadcast_object_list(payload, src=0)
            summary = payload[0]
        return summary

    def summary(self) -> dict[str, Any]:
        total = len(self._results)
        hf_success = sum(1 for item in self._results if item.hf_success)
        hf_failed = sum(1 for item in self._results if not item.hf_success)
        eval_submitted = sum(1 for item in self._results if item.eval_submitted)
        eval_skipped = sum(1 for item in self._results if item.eval_skipped)
        eval_failed = sum(
            1
            for item in self._results
            if item.hf_success and not item.eval_submitted and not item.eval_skipped
        )
        return {
            "total": total,
            "hf_success": hf_success,
            "hf_failed": hf_failed,
            "eval_submitted": eval_submitted,
            "eval_skipped": eval_skipped,
            "eval_failed": eval_failed,
            "failures": [asdict(item) for item in self._results if item.error],
        }

    def _pending_count(self) -> int:
        return sum(1 for _, future in self._futures if not future.done())

    def _drain_finished(self, block: bool) -> None:
        remaining: List[tuple[str, Future]] = []
        for path, future in self._futures:
            if future.done() or block:
                self._record_future_result(future, path=path)
            else:
                remaining.append((path, future))
        self._futures = remaining

    def _drain_oldest(self) -> None:
        if not self._futures:
            return
        path, future = self._futures.pop(0)
        _log(self.logger, "info", "[async_hf] waiting for %s", path)
        self._record_future_result(future, path=path)

    def _future_done(self, path: str, future: Future) -> None:
        """Executor-thread hook: fires the moment conversion completes."""
        self._record_future_result(future, path=path)

    def _record_future_result(self, future: Future, path: str | None = None) -> None:
        if path is not None:
            with self._lock:
                if path in self._notified:
                    return  # already notified via the other channel; future is done
                self._notified.add(path)
        try:
            result = future.result()
        except Exception as exc:
            result = HFCheckpointResult(
                global_step=-1,
                checkpoint_path="",
                hf_path="",
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
        self._results.append(result)
        self._notify_result(result)
        if result.error:
            self._write_failure(result)
            _log(
                self.logger,
                "warning",
                "[async_hf] best-effort HF checkpoint failed for %s: %s",
                result.checkpoint_path,
                result.error,
            )

    def _notify_result(self, result: HFCheckpointResult) -> None:
        """Forge 事件回调(见 __init__.on_result):best-effort,静默吞错保训练。"""
        if self.on_result is None:
            return
        try:
            self.on_result(result)
        except Exception as exc:
            _log(self.logger, "warning", "[async_hf] on_result callback failed: %s", repr(exc))

    def _run_hf_checkpoint(
        self,
        global_step: int,
        checkpoint_path: str,
        output_dir: str,
        ckpt_manager: str,
        save_ema: bool,
        enable_fp32: bool,
        model_assets: Sequence[Any] | None,
        epoch: int | None,
        epoch_step: int | None,
        best_effort: bool,
    ) -> HFCheckpointResult:
        start_time = time.time()
        hf_path = os.path.join(checkpoint_path, "hf_ckpt")
        ema_hf_path = os.path.join(checkpoint_path, "ema_hf_ckpt") if save_ema else ""
        result = HFCheckpointResult(
            global_step=global_step,
            checkpoint_path=checkpoint_path,
            hf_path=hf_path,
            epoch=epoch,
            epoch_step=epoch_step,
            ema_hf_path=ema_hf_path,
        )

        try:
            _log(self.logger, "info", "[async_hf] saving HF checkpoint for %s", checkpoint_path)
            self._save_one_hf_checkpoint(
                final_dir=hf_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                ckpt_manager=ckpt_manager,
                enable_fp32=enable_fp32,
                model_assets=model_assets,
                ema=False,
                global_step=global_step,
            )
            result.hf_success = True

            if save_ema:
                try:
                    self._save_one_hf_checkpoint(
                        final_dir=ema_hf_path,
                        checkpoint_path=checkpoint_path,
                        output_dir=output_dir,
                        ckpt_manager=ckpt_manager,
                        enable_fp32=enable_fp32,
                        model_assets=model_assets,
                        ema=True,
                        global_step=global_step,
                    )
                    result.ema_success = True
                except Exception as exc:
                    result.error = f"EMA HF save failed: {repr(exc)}"
                    result.traceback = traceback.format_exc()
                    if not best_effort:
                        raise

        except Exception as exc:
            result.error = repr(exc)
            result.traceback = traceback.format_exc()
            if not best_effort:
                raise
        finally:
            result.elapsed_sec = time.time() - start_time
            helper.empty_cache()

        if not result.error:
            _log(
                self.logger,
                "info",
                "[async_hf] HF checkpoint finished for %s in %.2fs",
                checkpoint_path,
                result.elapsed_sec,
            )
        return result

    @staticmethod
    def _patch_config_dtype(out_dir: str) -> None:
        """Stamp torch_dtype=bfloat16 on the converted config.json (2026-10-30):
        it was absent, so inference loaders fell back to fp32 — doubling host
        memory bandwidth for no gain. Best-effort, never fails the save."""
        try:
            cfg_path = os.path.join(out_dir, "config.json")
            if not os.path.isfile(cfg_path):
                return
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("torch_dtype"):
                return
            cfg["torch_dtype"] = "bfloat16"
            tmp = cfg_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            os.replace(tmp, cfg_path)
        except Exception as e:
            print(f"[hf-saver] patch config torch_dtype failed (ignored): {e!r}")

    def _save_one_hf_checkpoint(
        self,
        final_dir: str,
        checkpoint_path: str,
        output_dir: str,
        ckpt_manager: str,
        enable_fp32: bool,
        model_assets: Sequence[Any] | None,
        ema: bool,
        global_step: int,
    ) -> None:
        base_name = os.path.basename(final_dir)
        tmp_dir = os.path.join(
            os.path.dirname(final_dir),
            f".{base_name}.tmp.{global_step}.{os.getpid()}",
        )
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        state_dict = None
        try:
            state_dict = ckpt_to_state_dict(
                save_checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                ckpt_manager=ckpt_manager,
                ema=ema,
            )
            save_kwargs = {"model_assets": model_assets}
            # Storage contract 2026-10-30: hf artifacts are always saved bf16
            # (the model's native dtype) — fp32 doubled every hf_ckpt (24G ->
            # 12G) plus publish bandwidth/OSS cost for zero inference benefit.
            # enable_fp32 keeps its TRAINING-precision semantics untouched.
            save_kwargs["save_dtype"] = torch.bfloat16
            save_model_weights(tmp_dir, state_dict, **save_kwargs)
            self._patch_config_dtype(tmp_dir)
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
        finally:
            del state_dict
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except OSError:
                    pass
            helper.empty_cache()

    def _write_failure(self, result: HFCheckpointResult) -> None:
        if not self.failure_log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.failure_log_path), exist_ok=True)
            with open(self.failure_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        except Exception as exc:
            _log(self.logger, "warning", "[async_hf] failed to write failure log: %s", repr(exc))
