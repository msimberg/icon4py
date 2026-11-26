#!/usr/bin/env python3
"""Run the weekly-review orchestrator inside the AWF sandbox."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)

    pi = shutil.which("pi")
    if not pi:
        raise RuntimeError("pi is not available in the sandbox")
    api_url = os.environ.get("AWF_API_PROXY_OPENAI_URL")
    if not api_url:
        raise RuntimeError("AWF_API_PROXY_OPENAI_URL is not set")

    config_dir = Path(config["pi_config_dir"])
    model_id = config["model_id"]
    with (config_dir / "models.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "providers": {
                    "local-openai-proxy": {
                        "api": "openai-completions",
                        "baseUrl": f"{api_url.rstrip('/')}/v1",
                        "apiKey": "awf-placeholder",
                        "models": [
                            {
                                "id": model_id,
                                "name": model_id,
                                "contextWindow": 1048576,
                                "maxTokens": 131072,
                            }
                        ],
                    }
                }
            },
            handle,
        )
    settings = {
        "defaultProvider": "local-openai-proxy",
        "defaultModel": model_id,
        "packages": config["packages"],
    }
    with (config_dir / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    findings_dir = Path(config["findings_dir"])
    reports_dir = Path(config["reports_dir"])
    findings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reviewers = ", ".join(
        f"{item['name']} (output {item['output_path']})" for item in config["reviewers"]
    )
    prompt = (
        "Run the weekly icon4py review. "
        f"icon4py_checkout={config['checkout']}. icon4py_commit={config['commit']}. "
        f"review_date={config['review_date']}. run_id={config['run_id']}. "
        f"requested_severity={config['severity']}. panel_size={config['panel_size']}. "
        f"reviewers={reviewers}. "
        f"findings_dir={findings_dir}. reports_dir={reports_dir}. "
        "See the skill for the workflow."
    )
    env = dict(os.environ)
    env.update(
        {
            "PI_CODING_AGENT_DIR": str(config_dir),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "ICON4PY_CHECKOUT": config["checkout"],
            "TZ": config.get("tz", "Europe/Zurich"),
        }
    )
    # PI_OFFLINE disables package downloads. Packages are prepared on the host
    # before entering AWF, whose API proxy is the only inference connection.
    (config_dir / "sessions").mkdir(exist_ok=True)
    return subprocess.run(
        [
            pi,
            "-p",
            "--approve",
            "--mode",
            "json",
            "--tools",
            "subagent,get_subagent_result,read,ls,write,edit",
            "--skill",
            str(config_dir / "SKILL.md"),
            prompt,
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
