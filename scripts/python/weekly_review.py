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

The host collects tracked issues, prepares pi packages, and reconciles findings.
AWF isolates the agent. Its API proxy holds the upstream inference credential.

Use ``scripts/run weekly-review run --help`` for the required run configuration.
Local invocations are dry runs unless ``--publish`` is explicit.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer


cli = typer.Typer(name="weekly-review", no_args_is_help=True, help=__doc__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("weekly-review")
TZ = ZoneInfo("Europe/Zurich")
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "weekly-review"
WORKFLOW_FILENAME = "weekly-review.yml"
GITHUB_TIMEOUT = 60

MARKER_LABEL = "ai-review"
NEEDS_TRIAGE_LABEL = "ai-review:needs-triage"
CONFIRMED_LABEL = "ai-review:confirmed"
INVALID_LABEL = "ai-review:invalid"
LABELS = {
    MARKER_LABEL: {"color": "1d76db", "description": "Filed by the weekly review"},
    NEEDS_TRIAGE_LABEL: {"color": "fbca04", "description": "Finding not checked by a human"},
    CONFIRMED_LABEL: {"color": "0e8a16", "description": "Human-confirmed review finding"},
    INVALID_LABEL: {"color": "e99695", "description": "Human-rejected review finding"},
}
REVIEWER_SPECS = {
    "architecture": ("icon4py-architecture-reviewer", "architecture.json"),
    "correctness": ("icon4py-correctness-reviewer", "correctness.json"),
    "performance": ("icon4py-performance-reviewer", "performance.json"),
}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
FIXEDNESS_VALUES = {"fixed", "persists", "unknown", "not-detected"}
REOPEN_WINDOW = timedelta(days=30)
METADATA_BEGIN = "<!-- weekly-review:metadata"
METADATA_END = "-->"
TEXT_FINDING_FIELDS = (
    "title",
    "severity",
    "confidence",
    "description",
    "evidence",
    "file",
    "symbol",
    "suggested_fix",
    "reviewer",
)


class ReviewError(RuntimeError):
    """An explicitly surfaced pipeline failure."""


@dataclass(frozen=True)
class RunConfig:
    checkout: Path
    repo: str
    severity: str
    panel_size: int
    reviewers: tuple[str, ...]
    model_id: str
    model_base_url: str
    artifacts_dir: Path | None
    automation_actor: str | None


@dataclass(frozen=True)
class Workspace:
    root: Path
    pi_config: Path
    findings: Path
    reports: Path
    review_config: Path
    awf_config: Path


def log(message: str) -> None:
    LOGGER.info(message)


def warn(message: str) -> None:
    LOGGER.warning(message)


def now() -> datetime:
    return datetime.now(TZ)


def find_or_raise(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ReviewError(f"Required executable not found: {name}")
    return path


def require_repo(repo: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ReviewError(f"Invalid GitHub repository: {repo}")


def require_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ReviewError(f"Invalid model endpoint: {endpoint}")


def require_checkout(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or not result.stdout.strip():
        raise ReviewError(f"Cannot resolve the icon4py checkout HEAD: {checkout}\n{result.stderr.strip()}")
    return result.stdout.strip()


def new_workspace(base: Path | None, checkout: Path) -> Workspace:
    if base is None:
        local_base = checkout / "_local"
        local_base.mkdir(exist_ok=True)
        base = Path(tempfile.mkdtemp(prefix="weekly-review-", dir=local_base))
    else:
        base.mkdir(parents=True, exist_ok=True)
    root = base / now().strftime("%Y%m%d-%H%M%S")
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise ReviewError(f"Run workspace already exists: {root}") from exc
    workspace = Workspace(
        root=root,
        pi_config=root / "pi-config",
        findings=root / "findings",
        reports=root / "reports",
        review_config=root / "review.json",
        awf_config=root / "awf.json",
    )
    for path in (workspace.pi_config, workspace.findings, workspace.reports):
        path.mkdir()
    log(f"Run workspace: {root}")
    return workspace


class GitHub:
    def __init__(self, token: str | None, repo: str, automation_actor: str | None = None):
        self.token = token
        self.repo = repo
        self.automation_actor = automation_actor
        self.base = f"https://api.github.com/repos/{repo}"

    @property
    def authenticated(self) -> bool:
        return self.token is not None

    def request(self, method: str, path: str, *, body: dict | None = None) -> dict | list | None:
        url = path if path.startswith("https://") else self.base + path
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=GITHUB_TIMEOUT) as response:
                text = response.read().decode()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise ReviewError(f"GitHub {method} {url}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ReviewError(f"GitHub {method} {url}: {exc}") from exc

    def paginate(self, path: str, params: dict) -> list:
        output: list = []
        for page in range(1, 1000):
            query = urllib.parse.urlencode(dict(params, per_page=100, page=page))
            batch = self.request("GET", f"{path}?{query}")
            if not batch:
                break
            output.extend(batch)
            if len(batch) < 100:
                break
        return output

    def current_actor(self) -> str | None:
        if self.automation_actor:
            return self.automation_actor
        if not self.authenticated:
            return None
        if os.environ.get("GITHUB_ACTIONS") == "true":
            # The automatic token is represented by this identity in issue events.
            return "github-actions[bot]"
        user = self.request("GET", "https://api.github.com/user")
        return user.get("login") if isinstance(user, dict) else None

    def check_publication_ready(self) -> None:
        metadata = self.request("GET", f"https://api.github.com/repos/{self.repo}")
        if metadata.get("has_issues") is not True:
            raise ReviewError(f"Issues are disabled on {self.repo}; enable them before publishing")
        self.ensure_labels()

    def ensure_labels(self) -> None:
        for name, spec in LABELS.items():
            try:
                self.request("GET", f"/labels/{urllib.parse.quote(name)}")
            except ReviewError as exc:
                if "HTTP 404" not in str(exc):
                    raise
                self.request("POST", "/labels", body={"name": name, **spec})
                log(f"Created label {name}")

    def tracked_issues(self) -> list[dict]:
        return self.paginate("/issues", {"state": "all", "labels": MARKER_LABEL, "sort": "created"})

    def timeline(self, number: int) -> list:
        return self.paginate(f"/issues/{number}/timeline", {})

    def create_issue(self, title: str, body: str) -> int:
        issue = self.request(
            "POST", "/issues", body={"title": title, "body": body, "labels": [MARKER_LABEL, NEEDS_TRIAGE_LABEL]}
        )
        return issue["number"]

    def comment(self, number: int, body: str) -> None:
        self.request("POST", f"/issues/{number}/comments", body={"body": body})

    def update(self, number: int, **fields: object) -> None:
        self.request("PATCH", f"/issues/{number}", body=fields)


def parse_section(body: str, heading: str) -> str:
    start = body.find(f"## {heading}\n")
    if start < 0:
        return ""
    start += len(heading) + 3
    end = body.find("\n## ", start)
    return body[start:end].strip() if end >= 0 else body[start:].strip()


def parse_issue(issue: dict) -> dict:
    body = issue.get("body") or ""
    metadata: dict[str, str] = {}
    start = body.find(METADATA_BEGIN)
    if start >= 0:
        end = body.find(METADATA_END, start)
        if end >= 0:
            for line in body[start + len(METADATA_BEGIN) : end].strip().splitlines():
                key, _, value = line.partition(":")
                metadata[key.strip()] = value.strip()
    line_numbers = [int(item) for item in re.findall(r"\d+", metadata.get("lines", ""))]
    return {
        "number": issue["number"],
        "title": issue["title"],
        "file": metadata.get("file", ""),
        "lines": line_numbers[:2],
        "symbol": metadata.get("symbol", ""),
        "commit": metadata.get("commit", ""),
        "description": parse_section(body, "Summary"),
        "suggested_fix": parse_section(body, "Suggested fix"),
        "source_body": body,
    }


def collect_tracked_issues(github: GitHub) -> list[dict]:
    actor = github.current_actor()
    entries = []
    for issue in github.tracked_issues():
        if "pull_request" in issue:
            continue
        labels = {label["name"] for label in issue.get("labels", [])}
        if INVALID_LABEL in labels:
            state = "invalid"
        elif issue.get("state") == "open":
            state = "open"
        else:
            closed_events = [
                event
                for event in github.timeline(issue["number"])
                if isinstance(event, dict) and event.get("event") == "closed"
            ]
            if not closed_events:
                continue
            latest = closed_events[-1]
            if not actor:
                warn(f"Skipping #{issue['number']}: automation actor is unavailable; use --automation-actor to identify it")
                continue
            if (latest.get("actor") or {}).get("login") != actor:
                continue
            try:
                closed_at = datetime.fromisoformat(latest["created_at"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if now() - closed_at > REOPEN_WINDOW:
                continue
            state = "pipeline-closed"
        entries.append({"state": state, **parse_issue(issue)})
    return entries


def public_issue_entry(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "source_body"}


def file_link(repo: str, commit: str, file: str, lines: list[int] | None = None) -> str:
    url = f"https://github.com/{repo}/blob/{commit}/{file}"
    if not lines:
        return url
    if len(lines) == 2 and lines[1] > lines[0]:
        return f"{url}#L{lines[0]}-L{lines[1]}"
    return f"{url}#L{lines[0]}"


def workflow_run_metadata(started: datetime, repo: str) -> tuple[str, str | None]:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return f"local-{started.strftime('%Y%m%d-%H%M%S')}", None
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not run_id.isascii() or not run_id.isdigit() or int(run_id) < 1:
        raise ReviewError("Invalid GitHub run id")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    if not attempt.isascii() or not attempt.isdigit() or int(attempt) < 1:
        raise ReviewError("Invalid GitHub run attempt")
    server = (os.environ.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
    url = f"{server}/{repo}/actions/runs/{run_id}"
    if int(attempt) > 1:
        url += f"/attempts/{attempt}"
    return run_id, url


def run_reference(run_id: str, run_url: str | None) -> str:
    if run_url:
        return f"[GitHub Actions run]({run_url})"
    return f"`{run_id}`"


@lru_cache(maxsize=4)
def tracked_source_files(checkout: str, commit: str) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", checkout, "ls-tree", "-r", "--name-only", "-z", commit],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    if result.returncode:
        raise ReviewError(f"Cannot load source files at {commit}: {result.stderr.strip()}")
    return tuple(path for path in result.stdout.split(chr(0)) if path)


def source_path_for_reference(
    reference: str,
    checkout: Path,
    commit: str,
    primary_file: str | None = None,
) -> str | None:
    root = checkout.resolve()
    candidate = Path(reference)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            return None
    else:
        relative = reference
        while relative.startswith("./"):
            relative = relative[2:]
    if not relative or ".." in Path(relative).parts:
        return None

    files = set(tracked_source_files(str(root), commit))
    if primary_file and "/" not in relative:
        primary = Path(primary_file)
        if relative == primary.name and primary_file in files:
            return primary_file
        local = (primary.parent / relative).as_posix()
        if local in files:
            return local
    if relative in files:
        return relative
    matches = [path for path in files if path.endswith("/" + relative)]
    return matches[0] if len(matches) == 1 else None


CODE_REFERENCE_RE = re.compile(
    r"(?<![\w/.\-])"
    r"(?P<code>`)?"
    r"(?P<path>/?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9_]+)"
    r"(?::(?P<start>\d+)(?:[-\u2013](?P<end>\d+))?)?"
    r"(?(code)`)"
    r"(?!\d)"
)

MARKDOWN_LINK_RE = re.compile(r"(\[[^\]\n]+\]\([^)\n]+\))")


def link_code_references(
    text: str,
    checkout: Path | None,
    repo: str,
    commit: str,
    primary_file: str | None = None,
) -> str:
    if not text or checkout is None or repo is None or commit is None:
        return text
    root = checkout.resolve()

    def expand_fragment(fragment: str) -> str:
        def replace(match: re.Match) -> str:
            prefix = fragment.rfind("://", 0, match.start())
            if prefix >= 0:
                space = fragment.find(" ", prefix, match.start())
                if space < 0 or match.start() < space:
                    return match.group(0)
            following = match.string[match.end():]
            if len(following) > 1 and following[0] == "." and following[1].isdigit():
                return match.group(0)
            start_text = match.group("start")
            end_text = match.group("end")
            if start_text is None:
                lines = []
            else:
                start = int(start_text)
                end = int(end_text) if end_text else start
                if start < 1 or end < start:
                    return match.group(0)
                lines = [start, end]
            resolved = source_path_for_reference(
                match.group("path"), root, commit, primary_file
            )
            if resolved is None:
                return match.group(0)
            return f"[{match.group(0)}]({file_link(repo, commit, resolved, lines)})"

        parts = MARKDOWN_LINK_RE.split(fragment)
        return "".join(
            part if MARKDOWN_LINK_RE.fullmatch(part) else CODE_REFERENCE_RE.sub(replace, part)
            for part in parts
        )

    output = []
    ordinary = []
    in_fence = False
    in_comment = False

    def flush() -> None:
        if ordinary:
            output.append(expand_fragment("".join(ordinary)))
            ordinary.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if in_comment:
            output.append(line)
            if "-->" in stripped:
                in_comment = False
            continue
        if in_fence:
            output.append(line)
            if stripped.startswith(("```", "~~~")):
                in_fence = False
            continue
        if stripped.startswith("<!--"):
            flush()
            output.append(line)
            comment_tail = stripped[4:]
            if "-->" not in comment_tail:
                in_comment = True
            continue
        if stripped.startswith(("```", "~~~")):
            flush()
            output.append(line)
            in_fence = True
        else:
            ordinary.append(line)
    flush()
    return "".join(output)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole_seconds = max(0, round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def render_issue_body(
    finding: dict,
    repo: str,
    commit: str,
    date: str,
    run_id: str,
    panel_size: int,
    run_url: str | None = None,
    checkout: Path | None = None,
) -> str:
    metadata = {
        "file": finding["file"],
        "lines": f"{finding['lines'][0]}-{finding['lines'][1]}",
        "symbol": finding["symbol"],
        "severity": finding["severity"],
        "first_seen": date,
        "last_seen": date,
        "commit": commit,
        "run_id": run_id,
    }
    if run_url:
        metadata["run_url"] = run_url
    lines = [METADATA_BEGIN]
    lines.extend(f"{key}: {value}" for key, value in metadata.items())
    description = link_code_references(finding["description"], checkout, repo, commit, finding["file"])
    evidence = link_code_references(finding["evidence"], checkout, repo, commit, finding["file"])
    suggested_fix = link_code_references(finding["suggested_fix"], checkout, repo, commit, finding["file"])
    lines.extend(
        [
            METADATA_END,
            "## Summary",
            "",
            description,
            "",
            "## Evidence",
            "",
            evidence,
            "",
            "## Suggested fix",
            "",
            suggested_fix,
            "",
            "## Details",
            "",
            f"- Found by: `{finding['reviewer']}` (skeptic panel: {panel_size}/{panel_size} PASS)",
            f"- Location at detection: [{finding['file']} @ `{commit[:9]}`]({file_link(repo, commit, finding['file'], finding['lines'])})",
            f"- Tags: {', '.join(finding['tags'])}",
            f"- Run: {run_reference(run_id, run_url)}",
            "",
            "<sub>Automatic weekly review. Add `ai-review:confirmed` to confirm the finding."
            f" Add `{INVALID_LABEL}` to reject it and suppress equivalent future findings."
            " Ordinary closure keeps this issue closed but does not suppress new findings.</sub>",
            "",
        ]
    )
    return "\n".join(lines)


def update_last_seen(entry: dict, date: str, run_id: str, run_url: str | None = None) -> str:
    body = entry["source_body"]
    start = body.find(METADATA_BEGIN)
    end = body.find(METADATA_END, start)
    if start < 0 or end < start:
        return body
    metadata = body[start + len(METADATA_BEGIN) : end]
    metadata = re.sub(r"(?m)^last_seen:.*$", f"last_seen: {date}", metadata)
    if re.search(r"(?m)^run_id:.*$", metadata):
        metadata = re.sub(r"(?m)^run_id:.*$", f"run_id: {run_id}", metadata)
    else:
        metadata = metadata.rstrip() + f"\nrun_id: {run_id}\n"
    if run_url:
        if re.search(r"(?m)^run_url:.*$", metadata):
            metadata = re.sub(r"(?m)^run_url:.*$", f"run_url: {run_url}", metadata)
        else:
            metadata = metadata.rstrip() + f"\nrun_url: {run_url}\n"
    return body[:start] + METADATA_BEGIN + metadata + body[end:]


class PanelEvidenceError(ReviewError):
    """A missing or malformed vote prevents a positive panel decision."""


def record_validation_issue(notes: list[str] | None, message: str) -> None:
    warn(message)
    if notes is not None:
        notes.append(message)


def claim_candidate_index(record: dict, claimed: set[int]) -> int:
    candidate_index = record.get("candidate_index") if isinstance(record, dict) else None
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 1
    ):
        raise PanelEvidenceError("candidate_index must be a positive integer")
    if candidate_index in claimed:
        raise PanelEvidenceError(f"candidate_index {candidate_index} is used by more than one finding")
    claimed.add(candidate_index)
    return candidate_index


def declared_candidate_indices(records: object) -> set[int]:
    if not isinstance(records, list):
        return set()
    indices = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        index = record.get("candidate_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 1:
            continue
        indices.add(index)
    return indices


def read_panel_votes(
    directory: Path,
    panel_size: int,
    expected_voter: str,
    allowed_verdicts: set[str],
    fixedness_issue: int | None = None,
) -> list[dict]:
    expected_names = {f"{index}.json" for index in range(1, panel_size + 1)}
    try:
        if directory.is_symlink():
            raise PanelEvidenceError(f"Panel directory must not be a symlink: {directory}")
        directory = directory.resolve()
        findings_root = directory.parent.parent.resolve()
        if not directory.is_relative_to(findings_root):
            raise PanelEvidenceError(f"Panel path is outside findings: {directory}")
        entries = list(directory.iterdir())
    except OSError as exc:
        raise PanelEvidenceError(f"Cannot read panel directory {directory}: {exc}") from exc

    actual_names = set()
    for entry in entries:
        if entry.suffix != ".json":
            continue
        if entry.is_symlink():
            raise PanelEvidenceError(f"Panel file must not be a symlink: {entry}")
        resolved = entry.resolve()
        if resolved.parent != directory or resolved.name != entry.name or not resolved.is_file():
            raise PanelEvidenceError(f"Unexpected panel file: {entry.name}")
        actual_names.add(entry.name)
    if actual_names != expected_names:
        raise PanelEvidenceError(
            f"Panel {directory}: expected {sorted(expected_names)}, found {sorted(actual_names)}"
        )

    votes = []
    for index in range(1, panel_size + 1):
        filename = directory / f"{index}.json"
        try:
            vote = read_json(filename)
        except ReviewError as exc:
            raise PanelEvidenceError(f"Invalid vote {filename.name}: {exc}") from exc
        if not isinstance(vote, dict):
            raise PanelEvidenceError(f"Invalid vote {filename.name}: expected an object")
        if vote.get("voter") != expected_voter:
            raise PanelEvidenceError(f"Invalid voter in {filename.name}")
        verdict = vote.get("verdict")
        if verdict not in allowed_verdicts:
            raise PanelEvidenceError(f"Invalid verdict in {filename.name}: {verdict!r}")
        if not isinstance(vote.get("reasoning"), str) or not vote["reasoning"].strip():
            raise PanelEvidenceError(f"Reasoning is required in {filename.name}")
        duplicate_of = vote.get("duplicate_of")
        if verdict == "DUPLICATE":
            if isinstance(duplicate_of, bool) or not isinstance(duplicate_of, int) or duplicate_of < 1:
                raise PanelEvidenceError(f"Invalid duplicate target in {filename.name}")
        elif duplicate_of is not None:
            raise PanelEvidenceError(f"Unexpected duplicate target in {filename.name}")
        if fixedness_issue is not None and "number" in vote:
            number = vote["number"]
            if isinstance(number, bool) or not isinstance(number, int) or number != fixedness_issue:
                raise PanelEvidenceError(f"Vote {filename.name} references the wrong issue")
        votes.append(vote)
    return votes


def read_skeptic_panel(record: dict, workspace: Workspace, panel_size: int, claimed: set[int]) -> list[dict]:
    candidate_index = claim_candidate_index(record, claimed)
    return read_panel_votes(
        workspace.findings / "votes" / str(candidate_index),
        panel_size,
        "icon4py-finding-skeptic",
        {"PASS", "REJECT", "DUPLICATE", "UNCERTAIN"},
    )


def read_fixedness_panel(number: int, workspace: Workspace, panel_size: int) -> list[dict]:
    return read_panel_votes(
        workspace.findings / "fixedness" / str(number),
        panel_size,
        "icon4py-fixedness-checker",
        {"fixed", "persists", "unknown"},
        fixedness_issue=number,
    )


def prepare_normalized_finding(finding: object, severity: str) -> dict | None:
    if not isinstance(finding, dict):
        return None
    normalized = dict(finding)
    for field in ("description", "evidence", "suggested_fix"):
        value = normalized.get(field)
        if isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
            warn(f"Normalizing list-valued {field} to a Markdown list")
            normalized[field] = "\n".join(
                item if item.startswith(("- ", "* ")) else f"- {item}"
                for item in value
            )
    valid, _ = validate_finding(normalized, severity)
    return normalized if valid else None


def validate_finding(finding: object, severity: str) -> tuple[bool, str | None]:
    if not isinstance(finding, dict):
        return False, "not an object"
    for key in TEXT_FINDING_FIELDS:
        if not isinstance(finding.get(key), str) or not finding[key].strip():
            return False, f"{key} must be a non-empty string"
    if finding["severity"] not in SEVERITY_RANK or finding["confidence"] not in SEVERITY_RANK:
        return False, "invalid severity or confidence"
    if SEVERITY_RANK[finding["severity"]] < SEVERITY_RANK[severity]:
        return False, f"below requested severity {severity}"
    lines = finding.get("lines")
    if (
        not isinstance(lines, list)
        or len(lines) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in lines)
    ):
        return False, "lines must contain two positive integers"
    if lines[0] > lines[1]:
        return False, "line range is reversed"
    source_path = Path(finding["file"])
    if source_path.is_absolute() or ".." in source_path.parts:
        return False, "file must be a path within the checkout"
    if not isinstance(finding.get("tags"), list) or not all(isinstance(tag, str) and tag.strip() for tag in finding["tags"]):
        return False, "tags must be a list of non-empty strings"
    return True, None


def validate_accepted(
    findings: object,
    severity: str,
    workspace: Workspace,
    panel_size: int,
    claimed_indices: set[int] | None = None,
    validation_notes: list[str] | None = None,
) -> list[dict]:
    if not isinstance(findings, list):
        raise ReviewError("accepted.json must contain a JSON list")
    claimed = claimed_indices if claimed_indices is not None else set()
    accepted = []
    for finding in findings:
        normalized = prepare_normalized_finding(finding, severity)
        if normalized is None:
            warn("Skipping malformed accepted finding")
            continue
        try:
            votes = read_skeptic_panel(normalized, workspace, panel_size, claimed)
            if any(vote["verdict"] != "PASS" for vote in votes):
                raise PanelEvidenceError("A new issue requires only PASS votes")
        except PanelEvidenceError as exc:
            record_validation_issue(validation_notes, f"New finding {normalized['title']!r} was not published: {exc}")
            continue
        normalized["confidence"] = "high"
        accepted.append(normalized)
    return accepted


def canonical_fixedness(
    verdicts: object,
    workspace: Workspace,
    panel_size: int,
    tracked: list[dict],
    duplicate_ids: set[int] | None = None,
    initial: list[dict] | None = None,
    icon4py_commit: str | None = None,
    validation_notes: list[str] | None = None,
) -> dict[int, str]:
    if not isinstance(verdicts, list):
        raise ReviewError("fixedness.json must contain a JSON list")
    if initial is None:
        initial = tracked
    # These targets were named as duplicates before vote validation. Their
    # corresponding fixedness panels were deliberately skipped.
    duplicate_ids = duplicate_ids or set()
    by_number = {entry["number"]: entry["state"] for entry in tracked}
    eligible = {
        entry["number"] for entry in initial if entry["state"] == "open"
    } - duplicate_ids
    merged: dict[int, str] = {}
    for verdict in verdicts:
        if (
            not isinstance(verdict, dict)
            or isinstance(verdict.get("number"), bool)
            or not isinstance(verdict.get("number"), int)
            or verdict.get("verdict") not in FIXEDNESS_VALUES
        ):
            raise ReviewError(f"Invalid fixedness entry: {verdict!r}")
        number = verdict["number"]
        if number in duplicate_ids:
            continue
        if number in merged:
            raise ReviewError(f"Duplicate final fixedness verdict for issue {number}")
        if number not in by_number:
            raise ReviewError(f"Fixedness references a missing issue #{number}")
        if by_number.get(number) != "open":
            warn(f"Ignoring fixedness for a non-open issue #{number}: {by_number.get(number)}")
            continue

        claimed_verdict = verdict["verdict"]
        if claimed_verdict == "unknown":
            merged[number] = "unknown"
            continue
        if claimed_verdict == "not-detected":
            entry_commit = next(
                (entry.get("commit", "") for entry in initial if entry["number"] == number),
                None,
            )
            if entry_commit == icon4py_commit:
                merged[number] = "not-detected"
            else:
                merged[number] = "unknown"
                record_validation_issue(
                    validation_notes,
                    f"Issue #{number}: not-detected changed to unknown because the issue commit differs from {icon4py_commit}",
                )
            continue

        resolved = claimed_verdict
        failure = None
        try:
            votes = read_fixedness_panel(number, workspace, panel_size)
            if claimed_verdict == "fixed":
                resolved = "fixed" if all(vote["verdict"] == "fixed" for vote in votes) else "unknown"
            else:
                support = sum(vote["verdict"] == "persists" for vote in votes)
                resolved = "persists" if 2 * support > panel_size else "unknown"
        except PanelEvidenceError as exc:
            resolved = "unknown"
            failure = str(exc)
        if resolved != claimed_verdict:
            suffix = f" ({failure})" if failure else ""
            record_validation_issue(validation_notes, f"Issue #{number}: {claimed_verdict} changed to unknown{suffix}")
        merged[number] = resolved

    unexpected = merged.keys() - eligible
    if unexpected:
        raise ReviewError(f"Fixedness verdicts for ineligible issues: {sorted(unexpected)}")
    missing = eligible - merged.keys()
    if missing:
        raise ReviewError(f"Missing fixedness verdicts for eligible open issues: {sorted(missing)}")
    return merged


def canonical_duplicates(
    duplicates: object,
    severity: str,
    workspace: Workspace,
    panel_size: int,
    claimed_indices: set[int] | None = None,
    validation_notes: list[str] | None = None,
) -> list[dict]:
    if not isinstance(duplicates, list):
        raise ReviewError("duplicates.json must contain a JSON list")
    claimed = claimed_indices if claimed_indices is not None else set()
    valid = []
    for duplicate in duplicates:
        number = duplicate.get("duplicate_of") if isinstance(duplicate, dict) else None
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            warn(f"Skipping duplicate with invalid issue number: {duplicate!r}")
            continue
        normalized = prepare_normalized_finding(duplicate.get("finding"), severity)
        if normalized is None:
            warn(f"Skipping duplicate #{number}: malformed finding")
            continue
        duplicate = dict(duplicate, finding=normalized)
        if duplicate.get("confidence") not in {"high", "medium", "low"}:
            warn(f"Skipping duplicate #{number}: invalid confidence {duplicate.get('confidence')!r}")
            continue
        if not isinstance(duplicate.get("reasoning"), str) or not duplicate["reasoning"].strip():
            warn(f"Skipping duplicate #{number}: reasoning is required")
            continue
        try:
            votes = read_skeptic_panel(duplicate, workspace, panel_size, claimed)
            if any(vote["verdict"] not in {"PASS", "DUPLICATE"} for vote in votes):
                raise PanelEvidenceError("A duplicate requires only PASS or DUPLICATE votes")
            targets = {
                vote["duplicate_of"]
                for vote in votes
                if vote["verdict"] == "DUPLICATE"
            }
            if not targets:
                raise PanelEvidenceError("No DUPLICATE vote supports this duplicate")
            if targets != {number}:
                raise PanelEvidenceError(f"Duplicate targets {sorted(targets)} do not match issue #{number}")
        except PanelEvidenceError as exc:
            record_validation_issue(validation_notes, f"Duplicate for issue #{number} was not applied: {exc}")
            continue

        count = sum(vote["verdict"] == "DUPLICATE" for vote in votes)
        if count == panel_size:
            confidence = "high"
        elif count >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        if duplicate["confidence"] != confidence:
            record_validation_issue(
                validation_notes,
                f"Issue #{number}: duplicate confidence {duplicate['confidence']} changed to {confidence}",
            )
        duplicate["confidence"] = confidence
        valid.append(duplicate)
    return valid


def reconcile(
    github: GitHub,
    accepted: list[dict],
    duplicates: list[dict],
    fixedness: dict[int, str],
    tracked: list[dict],
    repo: str,
    commit: str,
    date: str,
    run_id: str,
    panel_size: int,
    publish: bool,
    actions: list[dict] | None = None,
    checkout: Path | None = None,
    run_url: str | None = None,
) -> list[dict]:
    if actions is None:
        actions = []
    by_number = {entry["number"]: entry for entry in tracked}
    run_ref = run_reference(run_id, run_url)
    if actions is None:
        actions = []

    def record(action: str, number: int | None, note: str, url: str | None = None) -> None:
        actions.append({"action": action, "number": number, "note": note, "url": url})
        log(f"  {action}" + (f" #{number}" if number else "") + f": {note}")

    for duplicate in duplicates:
        target = duplicate["duplicate_of"]
        entry = by_number.get(target)
        if entry is None:
            warn(f"Duplicate target missing from the current tracker: #{target}; skipping")
            continue
        if entry["state"] == "invalid":
            record("ignored", target, "matching finding is human-rejected")
            continue
        if entry["state"] == "pipeline-closed":
            if publish:
                github.update(target, state="open", state_reason="reopened")
            record("reopened", target, "re-detected by the review panel")
            entry["state"] = "open"
        if publish:
            github.update(target, body=update_last_seen(entry, date, run_id, run_url))
            github.comment(
                target,
                f"Re-detected in {run_ref} at commit "
                f"[`{commit[:9]}`](https://github.com/{repo}/commit/{commit}).\n\n"
                f"**Finding:** {duplicate['finding']['title']}\n\n"
                f"Panel: {duplicate.get('confidence', 'unspecified')}-confidence duplicate match.",
            )
        record("commented", target, f"duplicate: {duplicate['finding']['title']}")

    for finding in accepted:
        if publish:
            number = github.create_issue(
                finding["title"],
                render_issue_body(finding, repo, commit, date, run_id, panel_size, run_url, checkout),
            )
            record("created", number, finding["title"], f"https://github.com/{repo}/issues/{number}")
        else:
            record("new", None, finding["title"])

    for number, verdict in fixedness.items():
        if not publish:
            record("fixedness", number, verdict)
            continue
        if verdict == "fixed":
            github.comment(
                number,
                f"The fixedness panel resolved this issue in {run_ref} at commit "
                f"[`{commit[:9]}`](https://github.com/{repo}/commit/{commit}). Closing as completed.",
            )
            github.update(number, state="closed", state_reason="completed")
            record("closed", number, "fixed by panel reopening later if re-detected")
        elif verdict == "persists":
            github.comment(number, f"The fixedness panel finds this still present in {run_ref} at commit `{commit[:9]}`.")
            record("commented", number, "fixedness: persists")
        else:
            record("no-op", number, f"fixedness: {verdict}")

    if not publish:
        log("Dry run: no GitHub changes were applied.")
    return actions


def read_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ReviewError(f"Missing required output: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Cannot parse required output {path}: {exc}") from exc


def overview_path(workspace: Workspace) -> Path:


    reports = sorted(workspace.reports.glob("*.md"))
    if len(reports) != 1:
        raise ReviewError(f"Expected exactly one overview report in {workspace.reports}; found {len(reports)}")
    return reports[0]
def require_complete_review(config: RunConfig, workspace: Workspace) -> None:
    for reviewer in config.reviewers:
        filename = REVIEWER_SPECS[reviewer][1]
        payload = read_json(workspace.findings / filename)
        if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
            raise ReviewError(f"Incomplete review for {reviewer}: malformed {filename}")
    report = overview_path(workspace).read_text(encoding="utf-8")
    if "<!-- weekly-review:incomplete -->" in report:
        raise ReviewError("The orchestrator marked this review as incomplete; no results published")
    if "<!-- weekly-review:complete -->" not in report:
        raise ReviewError("Overview report has no weekly-review completion marker")


def configured_packages() -> list[str]:
    with (DATA_DIR / "settings.json").open(encoding="utf-8") as handle:
        settings = json.load(handle)
    packages = settings.get("packages")
    if not isinstance(packages, list) or not all(isinstance(package, str) and package for package in packages):
        raise ReviewError("settings.json must declare packages as a list of non-empty strings")
    return packages


def prepare_pi_packages(workspace: Workspace) -> None:
    shutil.copytree(DATA_DIR, workspace.pi_config, dirs_exist_ok=True)
    (workspace.pi_config / "sessions").mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY", "CSCS_INFERENCE_API_KEY"}
    }
    env["PI_CODING_AGENT_DIR"] = str(workspace.pi_config)
    env["PI_TELEMETRY"] = "0"
    pi = find_or_raise("pi")
    log("Installing configured pi packages outside AWF...")
    for package in configured_packages():
        subprocess.run([pi, "install", package], env=env, stdin=subprocess.DEVNULL, check=True, timeout=600)


def review_configuration(config: RunConfig, workspace: Workspace) -> dict:
    return {
        "checkout": str(config.checkout),
        "severity": config.severity,
        "panel_size": config.panel_size,
        "model_id": config.model_id,
        "pi_config_dir": str(workspace.pi_config),
        "findings_dir": str(workspace.findings),
        "reports_dir": str(workspace.reports),
        "reviewers": [
            {"name": REVIEWER_SPECS[name][0], "output_path": f"{workspace.findings}/{REVIEWER_SPECS[name][1]}"}
            for name in config.reviewers
        ],
        "packages": configured_packages(),
    }

def short_value(value: object, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tool_call(name: str, arguments: object, *, prefix: str) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    return f"{prefix}-> {name} {short_value(args)}"


def result_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "text" in result:
            return result_text(result["text"])
        if "content" in result:
            return result_text(result["content"])
        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))
    if isinstance(result, list):
        for item in result:
            text = result_text(item)
            if text:
                return text
    return ""


def tool_result_line(name: str, arguments: object, *, prefix: str) -> str:
    result = arguments if isinstance(arguments, dict) else {}
    duration = result.get("durationMs")
    duration_text = f" ({duration} ms)" if isinstance(duration, (int, float)) and not isinstance(duration, bool) else ""
    outcome = "failed" if result.get("isError") else "done"
    detail = ""
    if result.get("isError"):
        detail = short_value(result_text(result["result"] if "result" in result else result.get("content")))
        detail = f": {detail}" if detail else ""
    return f"{prefix}-> {name}: {outcome}{duration_text}{detail}"

def log_assistant_text(prefix: str, text: object) -> None:
    for line in str(text).splitlines():
        if line.strip():
            log(prefix + line)

def print_json_event(event: dict, prefix: str = "orchestrator: ") -> None:
    event_type = event.get("type")
    if event_type == "tool_execution_start":
        log(tool_call(event.get("toolName", "unknown"), event.get("args"), prefix=prefix))
    elif event_type == "tool_execution_end":
        log(tool_result_line(event.get("toolName", "unknown"), event, prefix=prefix))
    elif event_type == "message_end":
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            return
        for content in message.get("content") or []:
            if content.get("type") == "text" and content.get("text", "").strip():
                log_assistant_text(prefix, content["text"])
        if message.get("stopReason") == "error":
            log_assistant_text(prefix + "model error: ", message.get("errorMessage", "unknown error"))


def transcript_files(sessions_dir: Path, subagents: bool) -> list[Path]:
    files = []
    for directory, _, filenames in os.walk(sessions_dir):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                is_task = Path(directory).name == "tasks"
                if is_task == subagents:
                    files.append(Path(directory) / filename)
    return sorted(files)


class TranscriptWatcher:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.stop_event = threading.Event()
        self.offsets: dict[Path, int] = {}
        self.partial: dict[Path, str] = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.offsets = {path: path.stat().st_size for path in transcript_files(self.sessions_dir, subagents=True)}
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self._read_new_lines()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._read_new_lines()
            self.stop_event.wait(1)

    def _read_new_lines(self) -> None:
        for path in transcript_files(self.sessions_dir, subagents=True):
            if path.parent.name != "tasks" or not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            offset = self.offsets.get(path, 0)
            if size <= offset:
                continue
            with path.open(encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = self.partial.get(path, "") + handle.read()
                self.offsets[path] = handle.tell()
            lines = chunk.split("\n")
            self.partial[path] = lines.pop()
            for line in lines:
                self._print_line(path, line)

    def _print_line(self, path: Path, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict) or event.get("type") != "message":
            return
        message = event.get("message") or {}
        identifier = path.stem.rsplit("_", 1)[-1]
        prefix = f"subagent {identifier}: "
        if message.get("role") == "assistant":
            for content in message.get("content") or []:
                if content.get("type") == "text" and content.get("text", "").strip():
                    log_assistant_text(prefix, content["text"])
                elif content.get("type") == "toolCall":
                    log(tool_call(content.get("name", "unknown"), content.get("arguments"), prefix=prefix))
        elif message.get("role") == "toolResult":
            log(tool_result_line(message.get("toolName", "unknown"), message, prefix=prefix))


def process_events(process: subprocess.Popen, sessions_dir: Path) -> int:
    watcher = TranscriptWatcher(sessions_dir)
    watcher.start()
    try:
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
            if isinstance(event, dict):
                print_json_event(event)
        return process.wait()
    finally:
        watcher.stop()


def prepare_awf_mountpoints(tmp_dir: Path) -> None:
    # AWF uses these paths as bounded tmpfs overlays. With filesystem.allowWrite
    # making /tmp read-only, Docker cannot create missing mount-point parents.
    for path in (
        "gh-aw/mcp-logs",
        "gh-aw/sandbox/firewall/logs",
        "gh-aw/sandbox/firewall/audit",
    ):
        (tmp_dir / path).mkdir(parents=True, exist_ok=True)


def launch_awf(config: RunConfig, workspace: Workspace) -> int:
    # Pi uses these host-installed binaries from inside the AWF chroot.
    for tool in ("rg", "fd"):
        find_or_raise(tool)
    awf = shutil.which("awf") or os.environ.get("AWF_BIN")
    if not awf:
        raise ReviewError("awf is not installed. Set up AWF before running the review.")
    awf_config = {
        "network": {"allowDomains": []},
        "apiProxy": {
            "targets": {"openai": {"baseUrlEnv": "WEEKLY_REVIEW_OPENAI_BASE_URL"}},
            "allowedModels": [f"openai/{config.model_id}"],
            "modelCatalogue": {"enabled": False},
            "maxTurns": 80,
            "maxPermissionDenied": 5,
        },
        "rateLimiting": {"enabled": False},
        "container": {"agentTimeout": 220},
        "filesystem": {"allowWrite": [str(workspace.root)]},
    }
    workspace.awf_config.write_text(json.dumps(awf_config, indent=2), encoding="utf-8")
    for path in (workspace.root / "audit", workspace.root / "proxy-logs"):
        path.mkdir(exist_ok=True)
    prepare_awf_mountpoints(Path("/tmp"))
    env = dict(os.environ)
    env["WEEKLY_REVIEW_OPENAI_BASE_URL"] = config.model_base_url
    log(f"Launching the review with AWF for model {config.model_id}")
    process = subprocess.Popen(
        [
            awf,
            "--config",
            str(workspace.awf_config),
            "--audit-dir",
            str(workspace.root / "audit"),
            "--proxy-logs-dir",
            str(workspace.root / "proxy-logs"),
            "--container-workdir",
            str(workspace.root),
            "--",
            "/usr/bin/python3",
            str(workspace.pi_config / "agent_entry.py"),
            str(workspace.review_config),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
        bufsize=1,
    )
    return process_events(process, workspace.pi_config / "sessions")


def session_statistics(path: Path, role: str) -> dict:
    usage = {key: 0 for key in ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")}
    timestamps = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("timestamp"), str):
                created = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                timestamps.append(created)
            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if message.get("role") == "system" and role == "unknown":
                preamble = (message.get("sections") or {}).get("preamble", "")
                for marker, candidate in {
                    "You are a specialist reviewer for the C2SM/icon4py": "correctness-reviewer",
                    "You are a specialist performance reviewer": "performance-reviewer",
                    "You are a specialist architecture reviewer": "architecture-reviewer",
                    "Your job is to assess a proposed finding": "finding-skeptic",
                    "Your job is to decide whether an existing open issue is resolved": "fixedness-checker",
                }.items():
                    if marker in preamble:
                        role = candidate
                        break
            for key, value in (message.get("usage") or {}).items():
                if key in usage and isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] += value
    elapsed = None
    if len(timestamps) >= 2:
        elapsed = (max(timestamps) - min(timestamps)).total_seconds()
    return {"file": path.name, "role": role, "elapsed_seconds": elapsed, "tokens": usage}


def collect_statistics(workspace: Workspace, elapsed_seconds: float) -> dict:
    sessions = workspace.pi_config / "sessions"
    main = [session_statistics(path, "orchestrator") for path in transcript_files(sessions, subagents=False)]
    subagents = [session_statistics(path, "unknown") for path in transcript_files(sessions, subagents=True)]
    totals = {
        field: sum(session["tokens"][field] for session in main + subagents)
        for field in ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")
    }
    return {
        "elapsed_seconds": elapsed_seconds,
        "tokens": totals,
        "orchestrator_sessions": main,
        "subagent_sessions": subagents,
    }


def append_report(
    workspace: Workspace,
    actions: list[dict],
    publish: bool,
    stats: dict,
    failure: str | None = None,
    validation_notes: list[str] | None = None,
    run_id: str | None = None,
    run_url: str | None = None,
    review_date: str | None = None,
    repo: str | None = None,
    commit: str | None = None,
    checkout: Path | None = None,
) -> None:
    report_path = overview_path(workspace)
    report_text = report_path.read_text(encoding="utf-8")
    if validation_notes:
        prefix = ["**Host panel validation (overrides conflicting model panel claims):**", ""]
        prefix.extend(f"- {message}" for message in validation_notes)
        prefix.append("")
        report_text = "\n".join(prefix) + "\n" + report_text
    if run_id is not None or run_url:
        reference = run_reference(run_id, run_url)
        previous_line = r"(?m)^- \*\*Run id:\*\*.*$"
        if re.search(previous_line, report_text):
            report_text = re.sub(previous_line, f"- **Run:** {reference}", report_text, count=1)
        elif not re.search(r"(?m)^- \*\*Run:\*\*", report_text):
            date_line = r"(?m)^- \*\*Date:\*\*.*$"
            if re.search(date_line, report_text):
                report_text = re.sub(
                    date_line,
                    lambda match: match.group(0) + f"\n- **Run:** {reference}",
                    report_text,
                    count=1,
                )
            else:
                report_text = f"- **Run:** {reference}\n\n" + report_text
    if review_date and not re.search(r"(?m)^- \*\*Date:\*\*", report_text):
        report_text = f"- **Date:** {review_date}\n" + report_text
    if commit and not re.search(r"(?m)^- \*\*icon4py commit:\*\*", report_text):
        report_text = f"- **icon4py commit:** `{commit}`\n" + report_text
    report_text = link_code_references(report_text, checkout, repo, commit)
    lines = ["", "## GitHub reconciliation", ""]
    if failure:
        lines.extend([f"**Run failed:** {failure}", ""])
    if not publish:
        lines.extend(["Dry run: planned actions only.", ""])
    lines.extend(["| Action | Issue | Note |", "| --- | --- | --- |"])
    for action in actions:
        number = action["number"]
        if number:
            url = action["url"] or f"#{number}"
            issue_cell = f"[#{number}]({url})" if action["url"] else f"#{number}"
        else:
            issue_cell = "-"
        lines.append(f"| {action['action']} | {issue_cell} | {action['note']} |")
    lines.extend(
        [
            "",
            "## Session statistics",
            "",
            f"- Elapsed: **{format_duration(stats['elapsed_seconds'])}**",
            f"- Main sessions: **{len(stats['orchestrator_sessions'])}**",
            f"- Subagent sessions: **{len(stats['subagent_sessions'])}**",
            f"- Tokens across all model turns: **{stats['tokens']['totalTokens']:,}**",
            "",
            "Total tokens include repeated input and cache reads, not unique context tokens.",
            "",
            "| Session | Elapsed | Input | Cached read | Output | Reasoning | Total tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in stats["orchestrator_sessions"] + stats["subagent_sessions"]:
        identifier = item["file"].rsplit("_", 1)[-1].removesuffix(".jsonl")
        usage = item["tokens"]
        lines.append(
            f"| {item['role']} ({identifier}) | {format_duration(item['elapsed_seconds'])} | "
            f"{usage['input']:,} | {usage['cacheRead']:,} | {usage['output']:,} | "
            f"{usage['reasoning']:,} | {usage['totalTokens']:,} |"
        )
    report_path.write_text(report_text + "\n".join(lines) + "\n", encoding="utf-8")


def previous_run_commit(github: GitHub, branch: str | None) -> str | None:
    if not branch:
        warn("No branch identified; no diff baseline")
        return None
    runs = github.request(
        "GET",
        f"https://api.github.com/repos/{github.repo}/actions/workflows/{WORKFLOW_FILENAME}/runs"
        f"?status=success&branch={urllib.parse.quote(branch, safe='')}&per_page=10",
    )
    for run in runs.get("workflow_runs", []) if isinstance(runs, dict) else []:
        if str(run.get("id")) == os.environ.get("GITHUB_RUN_ID"):
            continue
        if run.get("head_branch") == branch:
            return run.get("head_sha")
    return None


def write_diff(checkout: Path, baseline: str | None, commit: str, destination: Path) -> None:
    if not baseline or baseline == commit:
        log("No previous baseline to diff against.")
        return
    probe = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{baseline}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        warn(f"Baseline commit {baseline[:9]} is unavailable; using a full-checkout review")
        return
    diff = subprocess.run(
        ["git", "-C", str(checkout), "diff", f"{baseline}..{commit}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if diff.returncode:
        raise ReviewError(f"Cannot compute review diff: {diff.stderr.strip()}")
    destination.write_text(diff.stdout, encoding="utf-8")


def branch_name(checkout: Path) -> str | None:
    if os.environ.get("GITHUB_REF_NAME"):
        return os.environ["GITHUB_REF_NAME"]
    result = subprocess.run(["git", "-C", str(checkout), "branch", "--show-current"], check=False, capture_output=True, text=True)
    return result.stdout.strip() or None


def run_pipeline(config: RunConfig, publish: bool) -> None:
    started = now()
    require_repo(config.repo)
    require_endpoint(config.model_base_url)
    if config.severity not in SEVERITY_RANK:
        raise ReviewError(f"Invalid severity: {config.severity}")
    if config.panel_size < 1:
        raise ReviewError("Panel size must be a positive integer")
    if not config.reviewers:
        raise ReviewError("At least one reviewer is required")
    if not config.model_id.strip():
        raise ReviewError("Model id must not be empty")
    commit = require_checkout(config.checkout)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if publish and not token:
        raise ReviewError("Publishing requires GITHUB_TOKEN or GH_TOKEN")
    github = GitHub(token, config.repo, config.automation_actor)
    if publish:
        github.check_publication_ready()
    else:
        log("Dry run: reading the tracker but not publishing issues")

    workspace = new_workspace(config.artifacts_dir, config.checkout)
    log(f"Review commit {commit} on {config.repo}")
    tracked = collect_tracked_issues(github)
    for entry in tracked:
        log(f"  #{entry['number']} [{entry['state']}] {entry['title']}")
    published_entries = [public_issue_entry(entry) for entry in tracked]
    for entry in published_entries:
        if entry.get("state") == "open" and entry.get("commit") == commit:
            entry["fixedness"] = "not-detected"
    (workspace.findings / "open-issues.json").write_text(
        json.dumps(published_entries, indent=2), encoding="utf-8"
    )
    baseline = previous_run_commit(github, branch_name(config.checkout))
    write_diff(config.checkout, baseline, commit, workspace.findings / "changes.diff")
    review_date = started.strftime("%Y-%m-%d")
    run_id, run_url = workflow_run_metadata(started, config.repo)
    workspace.review_config.write_text(
        json.dumps(review_configuration(config, workspace), indent=2), encoding="utf-8"
    )

    failure = None
    validation_notes = []
    actions = []
    try:
        prepare_pi_packages(workspace)
        return_code = launch_awf(config, workspace)
        if return_code:
            raise ReviewError(f"The AWF review failed with exit code {return_code}")
        raw_accepted = read_json(workspace.findings / "accepted.json")
        raw_duplicates = read_json(workspace.findings / "duplicates.json")
        conflicting_indices = declared_candidate_indices(raw_accepted) & declared_candidate_indices(raw_duplicates)
        if conflicting_indices:
            raise ReviewError(f"Candidates are both new and duplicate: {sorted(conflicting_indices)}")
        claimed_indices = set()
        accepted = validate_accepted(
            raw_accepted,
            config.severity,
            workspace,
            config.panel_size,
            claimed_indices=claimed_indices,
            validation_notes=validation_notes,
        )
        duplicates = canonical_duplicates(
            raw_duplicates,
            config.severity,
            workspace,
            config.panel_size,
            claimed_indices=claimed_indices,
            validation_notes=validation_notes,
        )
        raw_fixedness = read_json(workspace.findings / "fixedness.json")
        claimed_duplicate_targets = {
            entry["duplicate_of"]
            for entry in raw_duplicates
            if isinstance(entry, dict)
            and isinstance(entry.get("duplicate_of"), int)
            and not isinstance(entry["duplicate_of"], bool)
            and entry["duplicate_of"] > 0
        }

        # Re-evaluate the tracker immediately before applying results, while
        # making no claims of atomicity with the panel's reasoning.
        latest = collect_tracked_issues(github)
        fixedness = canonical_fixedness(
            raw_fixedness,
            workspace=workspace,
            panel_size=config.panel_size,
            tracked=latest,
            duplicate_ids=claimed_duplicate_targets,
            initial=tracked,
            icon4py_commit=commit,
            validation_notes=validation_notes,
        )
        log("Reconciling with the issue tracker...")
        actions = reconcile(
            github,
            accepted,
            duplicates,
            fixedness,
            latest,
            config.repo,
            commit,
            review_date,
            run_id,
            config.panel_size,
            publish,
            actions=actions,
            checkout=config.checkout,
            run_url=run_url,
        )
    except Exception as exc:
        failure = str(exc)
        raise
    finally:
        stats = collect_statistics(workspace, (now() - started).total_seconds())
        if len(list(workspace.reports.glob("*.md"))) == 1:
            append_report(workspace, actions, publish, stats, failure, validation_notes, run_id, run_url, review_date, config.repo, commit, config.checkout)
        (workspace.root / "statistics.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )

    log(f"Review finished. Report: {overview_path(workspace)}")


@cli.command(name="labels")
def labels(
    repo: Annotated[str, typer.Option(help="Repository as owner/name")],
) -> None:
    """Create missing ai-review labels."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise typer.BadParameter("GITHUB_TOKEN or GH_TOKEN is required")
    GitHub(token, repo).ensure_labels()


@cli.command(name="run")
def run(
    checkout: Annotated[Path, typer.Option(help="icon4py checkout to review")],
    repo: Annotated[str, typer.Option(help="Repository as owner/name")],
    severity: Annotated[str, typer.Option(help="Minimum severity: high, medium, or low")],
    panel_size: Annotated[int, typer.Option(help="Number of votes per finding")],
    reviewer: Annotated[list[str], typer.Option(
        "--reviewer",
        help="Reviewer to enable: architecture, correctness, or performance. Repeat the option.",
    )],
    model_id: Annotated[str, typer.Option(help="Model id, for example zai-org/GLM-5.3-Flash")],
    model_base_url: Annotated[str, typer.Option(help="Upstream OpenAI-compatible API base URL")],
    artifacts_dir: Annotated[Path | None, typer.Option(help="Parent for a fresh artifact workspace")] = None,
    automation_actor: Annotated[str | None, typer.Option("--automation-actor", help="Actor allowed to have pipeline-closed issues reopened; default: current token identity")] = None,
    publish: Annotated[bool, typer.Option("--publish", help="Create and update GitHub issues")]= False,
) -> None:
    """Run a review with explicit configuration."""
    unknown = set(reviewer) - REVIEWER_SPECS.keys()
    if unknown:
        raise typer.BadParameter(f"Unknown reviewers: {', '.join(sorted(unknown))}")
    config = RunConfig(
        checkout=checkout.resolve(),
        repo=repo,
        severity=severity,
        panel_size=panel_size,
        reviewers=tuple(dict.fromkeys(reviewer)),
        model_id=model_id,
        model_base_url=model_base_url,
        artifacts_dir=artifacts_dir.resolve() if artifacts_dir else None,
        automation_actor=automation_actor,
    )
    try:
        run_pipeline(config, publish)
    except ReviewError as exc:
        LOGGER.error(str(exc))
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    sys.exit(cli())
