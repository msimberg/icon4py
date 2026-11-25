#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Weekly icon4py review published as GitHub issues.

This host-side driver reads and writes the tracker, computes the diff
baseline, and starts the sandboxed pi review in scripts/data/weekly-review/.
The sandbox does not receive GitHub credentials or network access.

Commands: `run` for the pipeline, `labels` to create missing labels.

Human authority rules:
- Issues labeled `ai-review:invalid` are ignored.
- Issues closed by a human are not reopened.
- Issues closed by the pipeline are included for 30 days so that re-detection
  can reopen them.

There is no committed baseline. The head commit of the previous successful
workflow run provides the baseline. Local runs default to `msimberg/icon4py`.
GITHUB_REPOSITORY takes precedence. Use `--repo` to select another repository.
"""

from __future__ import annotations

import base64
import json
import os
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer


cli = typer.Typer(
    name="weekly-review",
    no_args_is_help=True,
    help="Weekly automated review of icon4py, published as GitHub issues.",
)


TZ = ZoneInfo("Europe/Zurich")
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = _SCRIPTS_DIR.parent
DATA_DIR = _SCRIPTS_DIR / "data" / "weekly-review"
SKILL_PATH = DATA_DIR / "SKILL.md"

# Local runs default to the fork so a stray --publish cannot touch upstream.
# CI always sets GITHUB_REPOSITORY, which takes precedence.
DEFAULT_REPO = "msimberg/icon4py"
UPSTREAM_REPO = "C2SM/icon4py"
WORKFLOW_FILENAME = "weekly-review.yml"

EGRESS_HOSTS = ["api.inference.cscs.ch", "registry.npmjs.org"]
NETNS = "weekly-review"
SANDBOX_TOOLS = "subagent,get_subagent_result,read,ls,write,edit"
FINDINGS_MOUNT = "/tmp/weekly-review-findings"
REPORTS_MOUNT = "/tmp/review-reports"

MARKER_LABEL = "ai-review"
NEEDS_TRIAGE_LABEL = "ai-review:needs-triage"
CONFIRMED_LABEL = "ai-review:confirmed"
INVALID_LABEL = "ai-review:invalid"

LABELS = {
    MARKER_LABEL: {
        "color": "1d76db",
        "description": "Filed by the automated weekly review",
    },
    NEEDS_TRIAGE_LABEL: {
        "color": "fbca04",
        "description": "Automated finding no human has checked yet",
    },
    CONFIRMED_LABEL: {
        "color": "0e8a16",
        "description": "A human checked this finding and confirmed it (human-set)",
    },
    INVALID_LABEL: {
        "color": "e99695",
        "description": "A human rejected this finding; the pipeline ignores it (human-set)",
    },
}

REOPEN_WINDOW_DAYS = 30
FIXEDNESS_VERDICTS = {"fixed", "persists", "unknown", "not-detected"}
REQUIRED_FINDING_FIELDS = {
    "title",
    "severity",
    "confidence",
    "description",
    "evidence",
    "file",
    "lines",
    "symbol",
    "suggested_fix",
    "reviewer",
    "tags",
}

METADATA_BEGIN = "<!-- weekly-review:metadata"
METADATA_END = "-->"

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("weekly-review")


def log(msg: str) -> None:
    LOGGER.info(msg)


def warn(msg: str) -> None:
    LOGGER.warning(msg)


def zurich_now() -> datetime:
    return datetime.now(TZ)


def review_date() -> str:
    return zurich_now().strftime("%Y-%m-%d")


def run_id() -> str:
    return zurich_now().strftime("weekly-%G-W%V")


def load_json(path: Path, label: str) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"failed to parse {label} ({path}): {exc}")
        return []
    if not isinstance(data, list):
        warn(f"{label} ({path}) must be a JSON list")
        return []
    return data


def github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None


def github_repo(args_repo: str | None = None) -> str:
    return os.environ.get("GITHUB_REPOSITORY") or args_repo or DEFAULT_REPO


# ---------------------------------------------------------------------------
# GitHub API client (stdlib only)
# ---------------------------------------------------------------------------


class GitHub:
    def __init__(self, token: str | None, repo: str):
        self.token = token
        self.repo = repo
        self.base = f"https://api.github.com/repos/{repo}"

    @property
    def authed(self) -> bool:
        return self.token is not None

    def request(self, method: str, path: str, *, body: dict | None = None) -> dict | list | None:
        url = path if path.startswith("https://") else f"{self.base}{path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                text = resp.read().decode()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"GitHub {method} {url} failed: {exc.code} {detail}") from exc

    def paginate(self, path: str, params: dict) -> list:
        out: list = []
        page = 1
        while True:
            query = dict(params, per_page=100, page=page)
            batch = self.request("GET", f"{path}?{urllib.parse.urlencode(query)}")
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out

    def bot_login(self) -> str | None:
        if not self.authed:
            return None
        if os.environ.get("GITHUB_ACTIONS") == "true":
            # The automatic GITHUB_TOKEN cannot call GET /user. Actions writes
            # issue events with this actor identity.
            return "github-actions[bot]"
        me = self.request("GET", "https://api.github.com/user")
        return me.get("login") if isinstance(me, dict) else None

    # -- labels ------------------------------------------------------------

    def label_exists(self, name: str) -> bool:
        try:
            self.request("GET", f"/labels/{urllib.parse.quote(name)}")
            return True
        except RuntimeError as exc:
            if "404" in str(exc):
                return False
            raise

    def create_label(self, name: str, color: str, description: str) -> bool:
        if self.label_exists(name):
            return False
        self.request("POST", "/labels", body={"name": name, "color": color, "description": description})
        return True

    def ensure_labels(self) -> list[str]:
        created = []
        for name, spec in LABELS.items():
            if self.create_label(name, spec["color"], spec["description"]):
                created.append(name)
                log(f"created label {name}")
        return created

    # -- issues ------------------------------------------------------------

    def list_review_issues(self) -> list[dict]:
        return self.paginate("/issues", {"state": "all", "labels": MARKER_LABEL, "sort": "created"})

    def issue_timeline(self, number: int) -> list:
        return self.paginate(f"/issues/{number}/timeline", {})

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        issue = self.request("POST", "/issues", body={"title": title, "body": body, "labels": labels})
        return issue["number"]

    def add_comment(self, number: int, body: str) -> None:
        self.request("POST", f"/issues/{number}/comments", body={"body": body})

    def update_issue(self, number: int, **fields) -> None:
        self.request("PATCH", f"/issues/{number}", body=fields)

    # -- workflow runs (diff baseline) ---------------------------------------

    def previous_run_head_sha(self, exclude_run_id: str | None) -> str | None:
        """Head commit of the last successful run of this workflow.

        Best effort: no previous run, or an API error (e.g. unauthenticated),
        means no baseline and the reviewers see the full checkout.
        """
        try:
            return self._previous_run_head_sha(exclude_run_id)
        except RuntimeError as exc:
            warn(f"could not read workflow run history: {exc}")
            return None

    def _previous_run_head_sha(self, exclude_run_id: str | None) -> str | None:
        data = self.request(
            "GET",
            f"https://api.github.com/repos/{self.repo}/actions/workflows/"
            f"{WORKFLOW_FILENAME}/runs?status=success&per_page=10",
        )
        runs = data.get("workflow_runs", []) if isinstance(data, dict) else []
        for run in runs:
            if exclude_run_id and str(run.get("id")) == exclude_run_id:
                continue
            return run.get("head_sha")
        return None


# ---------------------------------------------------------------------------
# Issue body rendering and parsing
# ---------------------------------------------------------------------------


def commit_link(repo: str, sha: str) -> str:
    return f"https://github.com/{repo}/commit/{sha}"


def file_link(repo: str, sha: str, file: str, lines: list[int] | None) -> str:
    url = f"https://github.com/{repo}/blob/{sha}/{file}"
    if lines and len(lines) == 2 and lines[1] > lines[0]:
        url += f"#L{lines[0]}-L{lines[1]}"
    elif lines:
        url += f"#L{lines[0]}"
    return url


def render_metadata(meta: dict) -> str:
    lines = [METADATA_BEGIN]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append(METADATA_END)
    return "\n".join(lines)


def render_issue_body(
    finding: dict,
    *,
    repo: str,
    commit: str,
    review_date_str: str,
    run: str,
) -> str:
    meta = {
        "file": finding["file"],
        "lines": f"{finding['lines'][0]}-{finding['lines'][1]}"
        if isinstance(finding.get("lines"), list) and len(finding["lines"]) == 2
        else "",
        "symbol": finding.get("symbol", ""),
        "severity": finding.get("severity", ""),
        "first_seen": review_date_str,
        "last_seen": review_date_str,
        "commit": commit,
        "run_id": run,
    }
    tags = ", ".join(finding.get("tags") or [])
    where = file_link(repo, commit, finding["file"], finding.get("lines"))
    body_parts = [
        render_metadata(meta),
        "## Summary",
        "",
        finding["description"],
        "",
        "## Evidence",
        "",
        finding["evidence"],
        "",
        "## Suggested fix",
        "",
        finding["suggested_fix"],
        "",
        "## Details",
        "",
        f"- Found by: `{finding['reviewer']}` (skeptic panel: 3/3 PASS)",
        f"- Location at detection: [{finding['file']} @ `{commit[:9]}`]({where})",
        f"- Tags: {tags}",
        f"- Run: `{run}`",
        "",
        "<sub>Filed automatically by the weekly review. Maintainers: add"
        f" `{CONFIRMED_LABEL}` after checking, or `{INVALID_LABEL}` / close the issue if the"
        " finding is wrong. Closed issues are not re-reported.</sub>",
        "",
    ]
    return "\n".join(body_parts)


def parse_section(body: str, heading: str) -> str:
    marker = f"## {heading}\n"
    start = body.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    nxt = body.find("\n## ", start)
    return body[start:nxt].strip() if nxt != -1 else body[start:].strip()


def parse_issue_body(issue: dict) -> dict:
    """Extract the machine-readable fields an issue body must carry."""
    body = issue.get("body") or ""
    meta: dict[str, str] = {}
    begin = body.find(METADATA_BEGIN)
    if begin != -1:
        end = body.find(METADATA_END, begin)
        if end != -1:
            for line in body[begin + len(METADATA_BEGIN) : end].strip().splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
    lines: list[int] = []
    raw_lines = meta.get("lines", "")
    if raw_lines:
        for part in re.split(r"[-,]", raw_lines):
            if part.strip().isdigit():
                lines.append(int(part.strip()))
    return {
        "file": meta.get("file", ""),
        "lines": lines,
        "symbol": meta.get("symbol", ""),
        "commit": meta.get("commit", ""),
        "description": parse_section(body, "Summary"),
        "suggested_fix": parse_section(body, "Suggested fix"),
    }


# ---------------------------------------------------------------------------
# Collecting the current issue state
# ---------------------------------------------------------------------------


def collect_open_issues(gh: GitHub) -> list[dict]:
    """Collect tracked issues, including invalid ones for duplicate suppression.

    Open issues and recently pipeline-closed issues are candidates for comments
    and fixedness. Invalid issues only prevent equivalent findings from being
    reported again. Open issues are always readable; the pipeline-closed state
    requires a token to identify the bot in the timeline.
    """
    bot = gh.bot_login()
    cutoff = zurich_now() - timedelta(days=REOPEN_WINDOW_DAYS)
    entries: list[dict] = []
    for issue in gh.list_review_issues():
        if "pull_request" in issue:
            continue
        labels = {label["name"] for label in issue.get("labels", [])}
        state = issue.get("state")
        if INVALID_LABEL in labels:
            entry_state = "invalid"
        elif state == "open":
            entry_state = "open"
        else:
            if bot is None:
                continue
            close_events = [
                event
                for event in gh.issue_timeline(issue["number"])
                if isinstance(event, dict)
                and event.get("event") == "closed"
                and (event.get("actor") or {}).get("login") == bot
            ]
            if not close_events:
                continue
            closed_at = close_events[-1].get("created_at")
            try:
                closed_dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                continue
            if closed_dt < cutoff:
                continue
            entry_state = "pipeline-closed"
        parsed = parse_issue_body(issue)
        if entry_state == "open" and not parsed["file"]:
            warn(f"issue #{issue['number']} has no location metadata; fixedness checks will be weak")
        entries.append(
            {
                "number": issue["number"],
                "title": issue["title"],
                "state": entry_state,
                **parsed,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Orchestrator output validation
# ---------------------------------------------------------------------------


def validate_accepted(accepted: list) -> list[dict]:
    valid: list[dict] = []
    for finding in accepted:
        if not isinstance(finding, dict):
            warn("skipping non-dict accepted entry")
            continue
        missing = REQUIRED_FINDING_FIELDS - set(finding.keys())
        if missing:
            warn(f"skipping accepted entry missing fields: {sorted(missing)}")
            continue
        if finding.get("severity") not in {"high", "medium", "low"}:
            warn(f"skipping accepted entry with bad severity: {finding.get('severity')!r}")
            continue
        valid.append(finding)
    return valid


def merge_fixedness_votes(fixedness: list) -> dict[int, str]:
    """Merge checker votes into one verdict per issue; fixed requires 3/3."""
    votes: dict[int, list[str]] = {}
    for entry in fixedness:
        if not isinstance(entry, dict):
            continue
        number = entry.get("number")
        verdict = entry.get("verdict")
        if not isinstance(number, int) or verdict not in FIXEDNESS_VERDICTS:
            warn(f"skipping fixedness entry {entry!r}")
            continue
        votes.setdefault(number, []).append(verdict)
    merged: dict[int, str] = {}
    for number, issue_votes in votes.items():
        if issue_votes.count("fixed") == 3:
            merged[number] = "fixed"
        elif issue_votes.count("persists") >= 2:
            merged[number] = "persists"
        else:
            merged[number] = "unknown"
    return merged


# ---------------------------------------------------------------------------
# Reconciliation: apply orchestrator output to the GitHub tracker
# ---------------------------------------------------------------------------


def reconcile(
    gh: GitHub,
    *,
    accepted: list[dict],
    duplicates: list,
    fixedness: list,
    collected: list[dict],
    repo: str,
    commit: str,
    review_date_str: str,
    run: str,
    publish: bool,
) -> list[dict]:
    """Apply the review verdicts to the tracker. Returns an actions list for the report."""
    actions: list[dict] = []
    by_number = {entry["number"]: entry for entry in collected}

    def act(action: str, number: int | None, note: str, url: str | None = None) -> None:
        actions.append({"action": action, "number": number, "note": note, "url": url})
        log(f"  {action}" + (f" #{number}" if number else "") + f": {note}")

    # 1. Duplicate merges: comment on (and maybe reopen) the existing issue.
    for entry in duplicates:
        if not isinstance(entry, dict):
            continue
        target = entry.get("duplicate_of")
        finding = entry.get("finding") or {}
        confidence = entry.get("confidence", "unknown")
        if not isinstance(target, int) or target not in by_number:
            warn(f"duplicate entry targets unknown issue {target!r}; skipping")
            act("skipped", None, f"duplicate target {target!r} is not in the tracked set")
            continue
        existing = by_number[target]
        if existing["state"] == "invalid":
            act("ignored", target, "duplicate of a human-rejected issue")
            continue
        was_pipeline_closed = existing["state"] == "pipeline-closed"
        title = finding.get("title", "")
        if publish:
            if was_pipeline_closed:
                gh.update_issue(target, state="open", state_reason="reopened")
            commit_url = commit_link(repo, commit)
            prefix = (
                "Reopening: the review pipeline closed this issue earlier and the same problem"
                if was_pipeline_closed
                else "The weekly review re-detected this issue"
            )
            gh.add_comment(
                target,
                f"{prefix} in run `{run}` at commit [`{commit[:9]}`]({commit_url}).\n\n"
                f"**New finding:** {title}\n\n"
                f"Panel: {confidence}-confidence duplicate match.",
            )
        act("reopened" if was_pipeline_closed else "commented", target, f"duplicate: {title}")

    # 2. Accepted findings: create new issues.
    for finding in accepted:
        body = render_issue_body(finding, repo=repo, commit=commit, review_date_str=review_date_str, run=run)
        if publish:
            number = gh.create_issue(
                finding["title"],
                body,
                labels=[MARKER_LABEL, NEEDS_TRIAGE_LABEL],
            )
            url = f"https://github.com/{repo}/issues/{number}"
        else:
            number, url = None, None
        act("created", number, f"new issue: {finding['title']}", url)

    # 3. Fixedness verdicts on open issues.
    verdicts = merge_fixedness_votes(fixedness)
    for number, verdict in verdicts.items():
        if number not in by_number:
            warn(f"fixedness verdict for unknown issue #{number}; skipping")
            continue
        if by_number[number]["state"] != "open":
            continue
        if verdict == "fixed":
            if publish:
                gh.add_comment(
                    number,
                    "The fixedness panel (3/3 votes) found this issue resolved at commit "
                    f"[`{commit[:9]}`]({commit_link(repo, commit)}). Closing as completed.",
                )
                gh.update_issue(number, state="closed", state_reason="completed")
            act("closed", number, "fixedness: fixed")
        elif verdict == "persists":
            if publish:
                gh.add_comment(
                    number,
                    "Fixedness check in run `" + run + "` at commit "
                    f"[`{commit[:9]}`]({commit_link(repo, commit)}): the issue still persists "
                    "(at least 2/3 panel votes).",
                )
            act("commented", number, "fixedness: persists")
        else:
            act("no-op", number, f"fixedness: {verdict} (no comment)")

    if not publish:
        log("Dry run: no GitHub changes were made. Pass --publish to apply.")
    return actions


# ---------------------------------------------------------------------------
# Sandbox (bwrap) invocation
# ---------------------------------------------------------------------------


def build_sandbox_command(
    *,
    checkout: Path,
    findings_dir: Path,
    reports_dir: Path,
    sessions_dir: Path,
    prompt: str,
) -> tuple[list[str], Path]:
    """Build the bwrap command and a writable config to remove after the run."""
    pi = shutil.which("pi")
    if not pi:
        raise RuntimeError("pi not found on PATH.")

    # pi auto-installs packages into its config directory, which must therefore
    # be writable and separate from scripts/data/weekly-review/.
    tmp_config = Path(tempfile.mkdtemp(prefix="pi-config-"))
    shutil.copytree(DATA_DIR, tmp_config, dirs_exist_ok=True)
    (tmp_config / "sessions").mkdir(exist_ok=True)

    cmd = [
        "bwrap",
        "--die-with-parent",
        "--clearenv",
        "--ro-bind", "/", "/",
        "--tmpfs", "/tmp",
        "--bind", str(tmp_config), "/tmp/pi-config",
        "--bind", str(findings_dir), FINDINGS_MOUNT,
        "--bind", str(reports_dir), REPORTS_MOUNT,
        "--bind", str(sessions_dir), "/tmp/pi-config/sessions",
    ]
    # Read-only checkout; hide .git so no checkout credentials leak in.
    cmd += ["--ro-bind", str(checkout), str(checkout)]
    if (checkout / ".git").is_dir():
        cmd += ["--tmpfs", str(checkout / ".git")]
    cmd += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--share-net",  # pi needs the inference endpoint; egress is firewalled outside
        "--setenv", "HOME", "/tmp",
        "--setenv", "PI_CODING_AGENT_DIR", "/tmp/pi-config",
        "--setenv", "PATH", os.environ.get("PATH", "/usr/bin:/bin"),
        "--setenv", "TZ", os.environ.get("TZ", "Europe/Zurich"),
        "--setenv", "CSCS_INFERENCE_API_KEY", os.environ.get("CSCS_INFERENCE_API_KEY", ""),
        "--setenv", "ICON4PY_CHECKOUT", str(checkout),
        "--setenv", "PI_SANDBOX_REPORTS_DIR", REPORTS_MOUNT,
        "--setenv", "PI_SANDBOX_FINDINGS_DIR", FINDINGS_MOUNT,
        "--chdir", "/tmp",
        pi,
        # pi text-print mode emits only the final assistant message. JSON mode
        # emits events immediately, which the job log needs for long runs.
        "-p", "--approve", "--mode", "json",
        "--tools", SANDBOX_TOOLS,
        "--skill", str(SKILL_PATH),
        prompt,
    ]
    return cmd, tmp_config


class TranscriptWatcher:
    """Follow subagent transcripts in sessions, including newly created files."""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.stop_event = threading.Event()
        self.offsets: dict[Path, int] = {}
        self.partial: dict[Path, str] = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        # Follow only new events and files created during this run.
        for path in self.sessions_dir.glob("*/tasks/*.jsonl"):
            self.offsets[path] = path.stat().st_size
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self._read_new_events()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._read_new_events()
            self.stop_event.wait(1)

    def _read_new_events(self) -> None:
        for path in self.sessions_dir.glob("*/tasks/*.jsonl"):
            size = path.stat().st_size
            offset = self.offsets.get(path, 0)
            if size <= offset:
                continue
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = self.partial.get(path, "") + handle.read()
                self.offsets[path] = handle.tell()
            lines = chunk.split("\n")
            self.partial[path] = lines.pop()
            for line in lines:
                self._print_event(path, line)

    def _print_event(self, path: Path, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Transcript headers are not JSON.
            return
        if not isinstance(event, dict) or event.get("type") != "message":
            return
        # Task transcripts store completed message records. Their event schema
        # differs from the JSON event stream emitted by `pi --mode json`.
        name = path.stem.rsplit("_", 1)[-1][:8]
        message = event.get("message") or {}
        role = message.get("role")
        if role == "assistant":
            for content in message.get("content") or []:
                if content.get("type") == "text" and content.get("text", "").strip():
                    log(f"subagent {name}: {content['text'].rstrip()}")
                elif content.get("type") == "toolCall":
                    log(f"subagent {name}: tool: {content.get('name', '?')}")
        elif role == "toolResult":
            outcome = "tool failed" if message.get("isError") else "tool completed"
            log(f"subagent {name}: {outcome}: {message.get('toolName', '?')}")


def stream_sandbox_progress(cmd: list[str], sessions_dir: Path) -> int:
    """Run the sandbox and print events from the orchestrator and its subagents."""
    watcher = TranscriptWatcher(sessions_dir)
    watcher.start()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log(line)
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type", "")
            if event_type == "message_update":
                inner = event.get("assistantMessageEvent") or {}
                if inner.get("type") == "toolcall_start":
                    log(f"  planning tool call: {inner.get('toolName', '?')}")
            elif event_type == "tool_execution_start":
                name = event.get("toolName", "?")
                args = event.get("args") or {}
                detail = ""
                if isinstance(args, dict):
                    detail = args.get("description") or args.get("summary") or ""
                log(f"  starting {name}" + (f": {detail}" if detail else ""))
            elif event_type == "tool_execution_error":
                log(f"  tool failed: {event.get('error', event)}")
            elif event_type == "message_end":
                message = event.get("message") or {}
                if message.get("role") == "assistant":
                    for content in message.get("content") or []:
                        if content.get("type") == "text" and content.get("text", "").strip():
                            log(content["text"].rstrip())
                    if message.get("stopReason") == "error":
                        log(f"  model error: {message.get('errorMessage', '(no message)')}")
            elif isinstance(event_type, str) and "subagent" in event_type:
                detail = event.get("description") or event.get("agent_name") or event.get("agent_id") or ""
                log(f"  {event_type}" + (f": {detail}" if detail else ""))
        return process.wait()
    finally:
        watcher.stop()

def netns_exec_prefix() -> list[str]:
    """Command prefix that runs a command inside the firewalled namespace.

    Namespace entry needs root; setpriv immediately drops back to the calling
    user so the review session itself runs unprivileged.
    """
    return [
        "sudo", "--preserve-env=CSCS_INFERENCE_API_KEY,PATH,TZ",
        "ip", "netns", "exec", NETNS,
        "setpriv",
        f"--reuid={os.getuid()}", f"--regid={os.getgid()}", "--clear-groups",
        "--",
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@cli.command(name="labels")
def labels(
    repo: Annotated[str | None, typer.Option(help="GitHub repository (default: $GITHUB_REPOSITORY, else msimberg/icon4py).")] = None,
) -> None:
    """Create the ai-review:* labels if missing."""
    token = github_token()
    if not token:
        LOGGER.error("GITHUB_TOKEN (or GH_TOKEN) is required to create labels.")
        raise typer.Exit(code=1)
    target = github_repo(repo)
    gh = GitHub(token, target)
    created = gh.ensure_labels()
    missing = [name for name in LABELS if name not in created]
    log(f"Labels ensured on {target}. Created: {created or 'none'}; already existed: {missing or 'none'}.")


@cli.command(name="run")
def run(
    checkout: Annotated[Path, typer.Option(help="Path to the icon4py checkout to review.")] = REPO_ROOT,
    severity: Annotated[str, typer.Option(help="Minimum severity to report.")] = "high",
    publish: Annotated[bool, typer.Option("--publish", help="Apply changes to the GitHub tracker. Without it, everything is a dry run.")] = False,
    findings_dir: Annotated[Path | None, typer.Option(help="Override the temporary findings directory.")] = None,
    reports_dir: Annotated[Path | None, typer.Option(help="Override the temporary reports directory.")] = None,
    artifacts_dir: Annotated[Path | None, typer.Option(help="Directory for the report, findings, and session transcripts.")] = None,
    netns: Annotated[bool, typer.Option("--netns", help="Run the review inside the netns firewall (requires 'netns_firewall.py setup' first).")] = False,
    repo: Annotated[str | None, typer.Option(help="GitHub repository (default: $GITHUB_REPOSITORY, else msimberg/icon4py).")] = None,
) -> None:
    """Run the full review pipeline."""
    _run(
        checkout=checkout,
        severity=severity,
        publish=publish,
        findings_dir=findings_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        netns=netns,
        repo_arg=repo,
    )


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------


ROLE_MARKERS = {
    "You are the orchestrator.": "orchestrator",
    "You are a specialist reviewer for the C2SM/icon4py": "correctness-reviewer",
    "You are a specialist performance reviewer": "performance-reviewer",
    "You are a specialist architecture reviewer": "architecture-reviewer",
    "Your job is to assess a proposed finding": "finding-skeptic",
    "Your job is to decide whether an existing open issue is resolved": "fixedness-checker",
}


def session_statistics(path: Path, default_role: str = "unknown") -> dict:
    """Summarize a persistent pi transcript; tokens include all model turns."""
    usage_fields = ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")
    usage = {field: 0 for field in usage_fields}
    timestamps: list[datetime] = []
    role = default_role
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("timestamp"), str):
                try:
                    timestamps.append(datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")))
                except ValueError:
                    pass

            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if message.get("role") == "system" and role == "unknown":
                preamble = (message.get("sections") or {}).get("preamble", "")
                for marker, candidate in ROLE_MARKERS.items():
                    if marker in preamble:
                        role = candidate
                        break
            message_usage = message.get("usage") or {}
            for field in usage_fields:
                value = message_usage.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[field] += value
    elapsed = None
    if len(timestamps) >= 2:
        elapsed = max(0, (max(timestamps) - min(timestamps)).total_seconds())
    return {
        "file": path.name,
        "role": role,
        "elapsed_seconds": elapsed,
        "tokens": usage,
    }


def collect_session_statistics(sessions_dir: Path, elapsed_seconds: float) -> dict:
    """Aggregate orchestrator and all subagent sessions into report data."""
    main_sessions = sorted(p for p in sessions_dir.glob("*.jsonl") if p.is_file())
    subagent_sessions = sorted(p for p in sessions_dir.glob("*/tasks/*.jsonl") if p.is_file())
    orchestrator = [session_statistics(path, default_role="orchestrator") for path in main_sessions]
    subagents = [session_statistics(path) for path in subagent_sessions]
    tokens = {
        field: sum(session["tokens"][field] for session in orchestrator + subagents)
        for field in ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")
    }
    return {
        "elapsed_seconds": elapsed_seconds,
        "tokens": tokens,
        "orchestrator_sessions": orchestrator,
        "subagent_sessions": subagents,
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def render_statistics_markdown(stats: dict) -> str:
    tokens = stats["tokens"]
    lines = [
        "",
        "## Session statistics",
        "",
        f"- Elapsed time: **{format_duration(stats['elapsed_seconds'])}**",
        f"- Main agent sessions: **{len(stats['orchestrator_sessions'])}**",
        f"- Subagent sessions: **{len(stats['subagent_sessions'])}**",
        f"- Tokens processed in all model turns: **{tokens['totalTokens']:,}**",
        "",
        "Token totals include input re-sent on each model turn; they are not unique context tokens.",
        "",
        "| Session | Elapsed | Input | Cached read | Output | Reasoning | Total tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in stats["orchestrator_sessions"] + stats["subagent_sessions"]:
        item_tokens = item["tokens"]
        session_id = item["file"].rsplit("_", 1)[-1].removesuffix(".jsonl")[:8]
        label = f"{item['role']} ({session_id})"
        lines.append(
            f"| {label} | {format_duration(item['elapsed_seconds'])} | "
            f"{item_tokens['input']:,} | {item_tokens['cacheRead']:,} | "
            f"{item_tokens['output']:,} | {item_tokens['reasoning']:,} | "
            f"{item_tokens['totalTokens']:,} |"
        )
    return "\n".join(lines) + "\n"


def _run(
    *,
    checkout: Path,
    severity: str,
    publish: bool,
    findings_dir: Path | None,
    reports_dir: Path | None,
    artifacts_dir: Path | None,
    netns: bool,
    repo_arg: str | None,
) -> None:
    started_at = zurich_now()
    repo = github_repo(repo_arg)
    if repo == UPSTREAM_REPO and not publish:
        log(f"Note: targeting {UPSTREAM_REPO} for a dry run; pass --repo to choose another repo.")
    checkout = checkout.resolve()
    if not (checkout / ".git").exists():
        LOGGER.error(f"{checkout} is not a git checkout.")
        raise typer.Exit(code=1)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    date_str = review_date()
    run = run_id()
    log(f"icon4py commit: {commit}")
    log(f"review date: {date_str}  run id: {run}  repo: {repo}")

    token = github_token()
    if publish and not token:
        LOGGER.error("--publish requires GITHUB_TOKEN (or GH_TOKEN).")
        raise typer.Exit(code=1)
    gh = GitHub(token, repo)

    if publish:
        gh.ensure_labels()

    log("Collecting existing review issues from GitHub...")
    if not gh.authed:
        log("No token: reading open issues anonymously; pipeline-closed tracking is unavailable.")
    collected = collect_open_issues(gh)
    if collected:
        for entry in collected:
            log(f"  #{entry['number']} [{entry['state']}] {entry['title']}")
    else:
        log("  no existing review issues found")
    log(
        f"Tracker state: {sum(e['state'] == 'open' for e in collected)} open, "
        f"{sum(e['state'] == 'pipeline-closed' for e in collected)} pipeline-closed (recent), "
        f"{sum(e['state'] == 'invalid' for e in collected)} invalid (duplicate suppression only)."
    )

    findings_path = findings_dir if findings_dir else Path(tempfile.mkdtemp(prefix="wr-findings-"))
    reports_path = reports_dir if reports_dir else Path(tempfile.mkdtemp(prefix="wr-reports-"))
    sessions_path = (artifacts_dir / "sessions") if artifacts_dir else Path(tempfile.mkdtemp(prefix="wr-sessions-"))
    for path in (findings_path, reports_path, sessions_path):
        path.mkdir(parents=True, exist_ok=True)

    current_run_id = os.environ.get("GITHUB_RUN_ID")
    prev_commit = gh.previous_run_head_sha(current_run_id)
    if prev_commit and prev_commit != commit:
        diff = subprocess.run(
            ["git", "-C", str(checkout), "diff", f"{prev_commit}..{commit}"],
            capture_output=True,
            text=True,
        )
        if diff.returncode == 0:
            (findings_path / "changes.diff").write_text(diff.stdout, encoding="utf-8")
            log(f"Wrote changes.diff ({prev_commit[:9]}..{commit[:9]}), {len(diff.stdout.splitlines())} lines.")
        else:
            (findings_path / "changes.diff").write_text("", encoding="utf-8")
            warn(f"git diff {prev_commit[:9]}..{commit[:9]} failed; wrote empty changes.diff.")
    else:
        log("No previous review run (or same commit); reviewers see no changes.diff.")

    (findings_path / "open-issues.json").write_text(json.dumps(collected, indent=2), encoding="utf-8")

    log("Running the sandboxed review session (this can take a long time)...")
    prompt = (
        "Run the weekly icon4py review. "
        f"icon4py_checkout={checkout}. icon4py_commit={commit}. review_date={date_str}. "
        f"run_id={run}. requested_severity={severity}. "
        f"findings_dir={FINDINGS_MOUNT}. reports_dir={REPORTS_MOUNT}. "
        "See the skill for the workflow."
    )
    sandbox_cmd, tmp_config = build_sandbox_command(
        checkout=checkout,
        findings_dir=findings_path,
        reports_dir=reports_path,
        sessions_dir=sessions_path,
        prompt=prompt,
    )
    cmd = netns_exec_prefix() + sandbox_cmd if netns else sandbox_cmd
    if netns:
        log(f"Running inside the netns firewall (allowed egress: {', '.join(EGRESS_HOSTS)}).")
    try:
        returncode = stream_sandbox_progress(cmd, sessions_path)
    finally:
        shutil.rmtree(tmp_config, ignore_errors=True)
    if returncode != 0:
        raise RuntimeError(f"review session failed with exit code {returncode}")

    report_candidates = sorted(reports_path.glob("*.md"))
    if not report_candidates:
        LOGGER.error("The review session wrote no overview report.")
        raise typer.Exit(code=1)
    report_path = report_candidates[-1]
    accepted = validate_accepted(load_json(findings_path / "accepted.json", "accepted.json"))
    duplicates = load_json(findings_path / "duplicates.json", "duplicates.json")
    fixedness = load_json(findings_path / "fixedness.json", "fixedness.json")
    log(
        f"Session output: {len(accepted)} accepted, {len(duplicates)} duplicates, "
        f"{len(fixedness)} fixedness votes."
    )

    log("Reconciling with the GitHub tracker...")
    actions = reconcile(
        gh,
        accepted=accepted,
        duplicates=duplicates,
        fixedness=fixedness,
        collected=collected,
        repo=repo,
        commit=commit,
        review_date_str=date_str,
        run=run,
        publish=publish,
    )

    if actions:
        lines = ["", "## GitHub reconciliation", ""]
        if not publish:
            lines.append("Dry run: none of the actions below were applied.")
            lines.append("")
        lines.append("| Action | Issue | Note |")
        lines.append("| --- | --- | --- |")
        for item in actions:
            issue = (
                f"[#{item['number']}]({item['url'] or f'https://github.com/{repo}/issues/{item["number"]}'})"
                if item["number"]
                else "-"
            )
            lines.append(f"| {item['action']} | {issue} | {item['note']} |")
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    stats = collect_session_statistics(
        sessions_path,
        (zurich_now() - started_at).total_seconds(),
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(render_statistics_markdown(stats))
    log(
        f"Review took {format_duration(stats['elapsed_seconds'])}; "
        f"processed {stats['tokens']['totalTokens']:,} tokens across "
        f"{len(stats['subagent_sessions'])} subagent sessions."
    )

    # 7. Artifacts: report, findings, and session transcripts.
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "statistics.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(report_path, artifacts_dir / report_path.name)
        for name in ("accepted.json", "duplicates.json", "fixedness.json", "open-issues.json", "changes.diff"):
            source = findings_path / name
            if source.exists():
                shutil.copy2(source, artifacts_dir / name)
        log(f"Artifacts written to {artifacts_dir}.")

    log("Done.")


if __name__ == "__main__":
    sys.exit(cli())
