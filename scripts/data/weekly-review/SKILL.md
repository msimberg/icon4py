---
name: weekly-review
description: "Run the weekly automated review of C2SM/icon4py and write accepted findings and a fixedness assessment."
---

# Weekly icon4py review

You are the orchestrator. Spawn read-only subagents to review the code and
verify findings, then write three JSON files to the findings directory and one
overview report to the reports directory.

## What your task prompt provides

- `icon4py_checkout`: absolute path to the icon4py clone (read-only).
- `icon4py_commit`: the commit being reviewed.
- `review_date`: `YYYY-MM-DD` (report filename, finding metadata).
- `run_id`: the run identifier (report metadata).
- `requested_severity`: tell reviewers to report only this severity.
- `findings_dir`: writable directory for your JSON outputs.
- `reports_dir`: writable directory for the overview report.

Pass these values to each subagent in its task prompt.

Tracked issues are in `<findings_dir>/open-issues.json`, a flat list with one
entry each. `state` is `open`, `pipeline-closed`, or `invalid`. Treat
For example:

```json
[{"number": 123, "state": "open", "title": "...", "file": "<relative>", "commit": "...", "description": "...", "suggested_fix": "...", "lines": [s, e], "symbol": "..."}]
```

A `pipeline-closed` issue was closed by the pipeline. A match means the
problem came back.
`pipeline-closed` entries like open ones for duplicate detection. Entries with
`state` `invalid` were human-rejected: suppress any matching finding. Do not
comment on, close, reopen, or run fixedness checks for invalid entries.

## Thresholds

- Acceptance: a finding becomes a **new** issue only if all three skeptics vote
  `PASS`. A valid finding that matches an existing tracked issue is merged into
  that issue instead. Any `UNCERTAIN` vote prevents a new issue from being created.
- Fixedness: an existing open issue is marked `fixed` only if all three of
  the three fixedness checkers vote `fixed`. Otherwise the verdict is the
  majority of the three votes, or `unknown` if there is no majority, and the
  issue stays open.

## Step 1: Run the reviewers

Run all reviewers and collect all findings before continuing:

- `icon4py-architecture-reviewer`, output `<findings_dir>/architecture.json`
- `icon4py-correctness-reviewer`, output `<findings_dir>/correctness.json`
- `icon4py-performance-reviewer`, output `<findings_dir>/performance.json`

Each task prompt gives `icon4py_checkout`, `requested_severity`, `output_path`,
and `findings_dir`. Pass `findings_dir` so reviewers can read `open-issues.json`
(already-tracked issues to avoid re-reporting) and `changes.diff` (code changed
since the last review) from there.

If a reviewer fails or its output is missing or malformed, record the failure
in the overview report and treat that reviewer's findings as empty. Continue.

## Step 2: Verify findings with the skeptic panel

Combine all findings. For each finding, run three `icon4py-finding-skeptic`
subagents. Each task prompt gives:
- `icon4py_checkout`
- the full `finding` object
- `open_issues`: the list from `<findings_dir>/open-issues.json`
- `output_path` under `<findings_dir>/votes/<index>/1.json`, `2.json`, `3.json`
  (`<index>` is the finding's position in the combined list, starting at 1)

Each skeptic returns `verdict` in `PASS|REJECT|DUPLICATE|UNCERTAIN` and,
for `DUPLICATE`, a `duplicate_of` issue number.

A missing or malformed vote counts as `UNCERTAIN`.

Classify each finding based on the three verdicts, in this order:

- **Rejected**: any verdict is `REJECT`. Record in the report; do not create an issue.
- **New issue** (`accepted.json`): all three verdicts are `PASS`.
- **Duplicate merge** (`duplicates.json`): at least one verdict is `DUPLICATE`
  and the others are `PASS`. All `DUPLICATE` votes must name the same
  `duplicate_of` issue number; split targets are uncertain, not a merge.
  `confidence` is `high` for 3x `DUPLICATE`, `medium` for 2x, `low` for 1x
  `DUPLICATE` with the other two `PASS`.
- **Uncertain/ambiguous**: anything else (e.g., `UNCERTAIN`, split `DUPLICATE`
  targets, or `PASS` mixed with `UNCERTAIN`). Record these in the report but do
  not create an issue.

Record missing or malformed votes and the classification of each finding in
the overview report.

## Step 3: Write accepted.json and duplicates.json

### 3.1 accepted.json

Write truly new findings to `<findings_dir>/accepted.json` as a flat JSON list.
Each entry is the reviewer's raw finding object (title, severity, description,
evidence, file, lines, symbol, suggested_fix) enriched by the orchestrator with
`reviewer` (the reviewer name), `confidence` (`high`, from all-3-PASS), and
`tags` (derived from the title and defect type; format below).
Skip a malformed entry with a note in the overview report rather than aborting.

`tags` is a list of 3-6 short kebab-case keywords that help someone scan the
tracker for related issues. Do not include the reviewer name, the date, or any
string longer than a short phrase.

### 3.2 duplicates.json

Write findings that match an existing tracked issue to
`<findings_dir>/duplicates.json` as a flat list:

```json
[
  {
    "finding": {... accepted finding object ...},
    "duplicate_of": 123,
    "confidence": "high|medium|low",
    "reasoning": "..."
  }
]
```

`duplicate_of` is the issue **number** from `open-issues.json`. `confidence`
follows the duplicate-merge classification above: `high` for 3x `DUPLICATE`,
`medium` for 2x, `low` for 1x `DUPLICATE` with the others `PASS`.
Any other case is not a duplicate merge; record it as uncertain in the report.

## Step 4: Assess fixedness of existing open issues

Let `accepted_findings` be the list in `accepted.json`. Let
`duplicate_matched_numbers` be the set of `duplicate_of` values in
`duplicates.json`.

For each entry in `open-issues.json` whose state is NOT `invalid` AND whose
number is NOT in `duplicate_matched_numbers` AND that has no accepted finding
describing the same problem:

- If the entry's `commit` equals `icon4py_commit` (the reviewers saw the
  same code and did not reproduce it), write
  `{"number": 123, "verdict": "not-detected"}` directly. Do not spawn a panel.
- If the commit changed, run three `icon4py-fixedness-checker` subagents
  against `<icon4py_checkout>/<entry.file>`. Each task prompt gives
  `icon4py_checkout`, the issue (number, title, description, file, lines,
  symbol, suggested_fix), and an `output_path` under
  `<findings_dir>/fixedness/<number>/1.json`, `2.json`, `3.json`.

  **Merge the three verdicts into exactly one entry per issue.** Apply this
  fixedness threshold:
  - `fixed`: all three votes are `fixed`.
  - `persists`: at least two of the three votes are `persists`.
  - `unknown`: no majority (e.g., one vote each, or two `unknown`).
  A missing or malformed vote counts as `unknown`.

Write `<findings_dir>/fixedness.json` as a flat list with **exactly one entry
per unmatched open issue**:

```json
[{"number": 123, "verdict": "fixed|persists|unknown|not-detected"}]
```

Do not write one entry per checker vote. If an issue has three checker
verdicts, they must be merged into a single entry before writing the file.

A missing entry for an unmatched open issue leaves it open with a warning.
`unknown` means the panel could not decide (possibly stale) and leaves the
issue open.

## Step 5: Write the overview report

Write the overview document to `<reports_dir>/<review_date>-<HHMM>.md`
(e.g. `2026-08-05-2114.md`) so same-day runs do not conflict.

The body includes:

- Run metadata: date, run id, icon4py commit.
- Summary counts: total findings reviewed, findings submitted to the panel,
  new accepted findings, merged duplicates, rejected findings, uncertain
  findings, and the fixedness outcome for existing open issues.
- A table of new accepted findings with title, severity, confidence, and file
- A table of new accepted findings with title, severity, confidence, and file
  (relative path and line range). The host turns these into GitHub permalinks.
  Do not write URLs yourself.
- A table of merged duplicates: new finding title, existing issue number, and
  confidence.
- A table of rejected findings with title, the skeptic verdicts, and the file.
- A table of uncertain/ambiguous findings with title, verdicts, and note.
- A table of fixedness outcomes: issue number, verdict, and a note.
- Notes on any reviewer, skeptic, or fixedness failures.
