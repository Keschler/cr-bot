"""Measure native live capture and GPU extraction/policy without the frontend."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class Finished(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--yolo-image-size", type=int, default=896)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup_frames < 0:
        parser.error("seconds must be positive and warmup frames non-negative")
    if not args.capture_only and args.checkpoint is None:
        parser.error("--checkpoint is required for extraction + policy")

    import cv2
    cv2.setNumThreads(1)
    from simulator.physical_lab.workflow.prototype_controller import (
        AdbH264FrameSource, CachedAdbPhoneController, LivePrototypeRunner,
        V4Actor, _bootstrap_extractor_runtime, configure_detector_inference_size,
    )

    source = AdbH264FrameSource(
        CachedAdbPhoneController(args.serial, device_label="BENCHMARK"),
        diagnostics_enabled=True,
    )
    report = {
        "serial": args.serial, "transport": "stream", "capture_resolution": "native",
        "frontend": False, "execute": False, "yolo_image_size": args.yolo_image_size,
    }
    try:
        source.start()
        if args.capture_only:
            source.next_frame()
            start = time.monotonic()
            first = source.diagnostics().decoded_frames
            while time.monotonic() - start < args.seconds:
                time.sleep(min(1, args.seconds - (time.monotonic() - start)))
            elapsed = time.monotonic() - start
            capture = source.diagnostics()
            report.update(seconds=elapsed, capture_fps=(capture.decoded_frames - first) / elapsed)
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; refusing a CPU substitute for this GPU benchmark")
            _bootstrap_extractor_runtime()
            from cr_bot.app.pipeline import normalize_frame, process_frame
            from cr_bot.vision.yolo_runtime import build_detector

            checkpoint = args.checkpoint.expanduser().resolve()
            report.update(checkpoint=str(checkpoint), gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
            print(json.dumps({"phase": "loading", **report}), flush=True)
            detector = build_detector(device="0")
            configure_detector_inference_size(detector, args.yolo_image_size)
            actor = V4Actor(checkpoint, device="cuda")
            state = {"started": None, "ended": None, "completed": 0, "measured": 0, "in_game": 0,
                     "policy_calls": 0, "capture_start": 0, "last_print": 0.0}
            stages = {}
            last_vision = {}
            policy_times = []
            original_decide = actor.decide

            def decide(observation):
                started = time.monotonic()
                action = original_decide(observation)
                if state["started"] is not None:
                    policy_times.append((time.monotonic() - started) * 1000)
                    state["policy_calls"] += 1
                return action

            actor.decide = decide

            def extract(*a, **kw):
                nonlocal last_vision
                analysis = process_frame(*a, **kw, collect_timing=True)
                last_vision = analysis.vision_timing_ms
                return analysis

            def record(frame):
                now = time.monotonic()
                state["completed"] += 1
                if state["completed"] <= args.warmup_frames:
                    return
                if state["started"] is None:
                    state["started"] = now
                    state["capture_start"] = source.diagnostics().decoded_frames
                    state["last_print"] = now
                    print(json.dumps({"phase": "measuring"}), flush=True)
                    return
                state["measured"] += 1
                state["in_game"] += int(frame.in_game)
                for name, elapsed in last_vision.items():
                    stages.setdefault(name, []).append(elapsed)
                elapsed = now - state["started"]
                if now - state["last_print"] >= 5:
                    print(json.dumps({"phase": "measuring", "seconds": round(elapsed, 2),
                        "analysis_fps": state["measured"] / elapsed,
                        "policy_calls": state["policy_calls"],
                        "capture_recent_fps": source.diagnostics().recent_decoded_fps}), flush=True)
                    state["last_print"] = now
                if elapsed >= args.seconds:
                    state["ended"] = now
                    raise Finished()

            runner = LivePrototypeRunner(
                source, detector, actor, execute=False, poll_interval_s=0,
                process_frame_fn=extract, normalize_frame_fn=normalize_frame,
            )
            error = None
            try:
                runner.run(on_record=record)
            except Finished:
                pass
            except Exception as exc:
                error = str(exc)
            end = state["ended"] if state["ended"] is not None else time.monotonic()
            elapsed = end - state["started"] if state["started"] is not None else 0
            report.update(
                seconds=elapsed, analyzed_frames=state["measured"],
                analysis_fps=state["measured"] / elapsed if elapsed else 0,
                capture_fps=(source.diagnostics().decoded_frames - state["capture_start"]) / elapsed if elapsed else 0,
                in_game_frames=state["in_game"], policy_calls=state["policy_calls"],
                policy_median_ms=statistics.median(policy_times) if policy_times else None,
                vision_median_ms={name: statistics.median(values) for name, values in stages.items()},
                error=error,
            )
        report["capture"] = source.diagnostics().as_dict()
        print(json.dumps({"HEADLESS_BENCHMARK": report}, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return int(bool(report.get("error")))
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
