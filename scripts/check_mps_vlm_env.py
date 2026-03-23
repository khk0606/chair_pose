#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import import_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MPS-ready environment for VLM LoRA fine-tuning."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser.parse_args()


def try_import(module_name: str) -> tuple[bool, str]:
    try:
        mod = import_module(module_name)
    except Exception as exc:  # pragma: no cover
        return False, str(exc)
    version = getattr(mod, "__version__", "unknown")
    return True, str(version)


def main() -> int:
    args = parse_args()

    result: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    required_modules = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "trl",
        "peft",
    ]

    imports: dict[str, dict[str, str | bool]] = {}
    for name in required_modules:
        ok, detail = try_import(name)
        imports[name] = {"ok": ok, "detail": detail}
    result["imports"] = imports

    torch_ok = imports["torch"]["ok"] is True
    mps_info: dict[str, object] = {
        "is_built": False,
        "is_available": False,
        "tensor_test": False,
        "device": None,
        "error": None,
    }
    if torch_ok:
        import torch  # type: ignore

        try:
            mps_info["is_built"] = bool(torch.backends.mps.is_built())
            mps_info["is_available"] = bool(torch.backends.mps.is_available())
            if torch.backends.mps.is_available():
                t = torch.randn((4, 4), device="mps")
                mps_info["tensor_test"] = True
                mps_info["device"] = str(t.device)
        except Exception as exc:  # pragma: no cover
            mps_info["error"] = str(exc)
    result["mps"] = mps_info

    all_imports_ok = all(v["ok"] for v in imports.values())
    ready = bool(all_imports_ok and mps_info["is_available"] and mps_info["tensor_test"])
    result["ready_for_mps_vlm_lora"] = ready

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"python: {result['python_version']} ({result['machine']})")
        for name, info in imports.items():
            status = "OK" if info["ok"] else "MISSING"
            print(f"- {name:<12} {status}  {info['detail']}")
        print(
            f"- mps built={mps_info['is_built']} available={mps_info['is_available']} "
            f"tensor_test={mps_info['tensor_test']}"
        )
        if mps_info["error"]:
            print(f"- mps error: {mps_info['error']}")
        print(f"\nREADY: {ready}")

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
