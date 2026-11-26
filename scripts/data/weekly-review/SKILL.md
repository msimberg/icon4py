---
name: weekly-review
description: Review C2SM/icon4py through reviewers, skeptic panels, and fixedness checks.
---

# Weekly icon4py review

You orchestrate the review. Read-only subagents inspect the checkout. You write
findings and one report in the supplied directories; you never write GitHub
issues directly.

Task inputs:

- `icon4py_checkout`: read-only repository path
- `requested_severity`: minimum severity to report
- `panel_size`: number of votes per finding
- `reviewers`: reviewers to run and each reviewer's output path
- `findings_dir`: writable findings directory
- `reports_dir`: writable report directory

Current tracked issues are `<findings_dir>/open-issues.json`. Every entry has a
GitHub issue `number`, title, location, description, suggested fix, and state:

- `open`: eligible for duplicate comments and fixedness checks.
- `pipeline-closed`: closed by the automation actor within the last 30 days.
  A duplicate match can reopen it.
- `invalid`: human-rejected. Suppress matching findings and take no action on
  this original issue.

`changes.diff`, when present, is code changed since the previous review on this
branch. The diff tells reviewers where to start, not where to stop; reviewers
must still examine the full checkout. If the file is absent, inspect the full
checkout.

## Thresholds

Let `N = panel_size`.

- A new issue requires all `N` skeptic votes to be `PASS`.
- A valid finding that matches an existing tracked issue is a duplicate, not a
  new issue. `invalid` matches are suppressed without touching the original.
- A fixedness verdict is `fixed` only when all `N` checkers vote `fixed`.
- `persists` requires a strict majority (more than half) of `persists` votes.
  Everything else is `unknown`, including missing or malformed votes. Only
  `fixed` closes an issue.

## 1. Review

Run every reviewer in `reviewers` with its assigned agent name and output path.
Read the tracked issues first so already tracked findings are not reported again.
Human-rejected findings in `open-issues.json` must also be excluded.

A reviewer's findings file is valid only if it has this structure:

```json
{
  "reviewer": "<assigned name>",
  "findings": [
    {
      "title": "Short title",
      "severity": "high|medium|low",
      "description": "Inline Markdown, no headings",
      "evidence": "Inline Markdown, no headings",
      "file": "relative/path/from/icon4py_checkout.py",
      "lines": [100, 120],
      "symbol": "Exact relevant symbol",
      "suggested_fix": "Inline Markdown, no headings"
    }
  ]
}
```

If a reviewer fails or writes malformed results, mark the review
`incomplete`, record the failed stage in the overview report, and do not use
that review as a diff baseline. Do not silently turn a failed reviewer into
an empty successful result. The host validates the result files and the
completion marker before publication.

## 2. Skeptic panels

Combine the candidates. Check for duplicates against existing issues and
against other candidates from this run before accepting anything. When two
candidates describe the same root problem, keep one coherent finding and
concentrate its evidence. Record the consolidation in the report.

Give each remaining candidate a distinct positive integer `candidate_index`,
starting from 1. This is a local locator for evidence files, not a fingerprint
or an identity across runs. Keep the index attached when some candidates are
rejected or merged; do not renumber the accepted subset.

For each candidate, spawn `N` `icon4py-finding-skeptic` agents. Give them:

- `icon4py_checkout`
- the full finding
- `open_issues`: entries from `open-issues.json`
- an output path `<findings_dir>/votes/<candidate_index>/<j>.json` for `j` from 1 to `N`

Each skeptic returns `verdict` in `PASS|REJECT|DUPLICATE|UNCERTAIN`. For
`DUPLICATE`, it returns the existing issue `number` in `duplicate_of`.
Missing or malformed votes count as `UNCERTAIN`.

Classify candidates in this order:

1. **Rejected:** any vote is `REJECT`. Do not create an issue.
2. **New issue:** every vote is `PASS`. Include it in `accepted.json`.
3. **Duplicate:** the other votes are `PASS` and at least one vote is
   `DUPLICATE`. All duplicate votes must name the same issue number. Include
   the finding in `duplicates.json`.
4. **Uncertain:** everything else, including mixed verdict types, split
   duplicate targets, or missing votes. Do not create an issue.

For duplicate entries, use `confidence` `high` if all `N` votes are `DUPLICATE`,
`medium` if at least two but fewer than `N` are `DUPLICATE`, and `low` if one
is `DUPLICATE`. Include consolidated skeptic reasoning in the report.

`accepted.json` is a flat list. Each entry contains `candidate_index`, the
reviewer fields listed above, `reviewer`, `confidence` (`high` for a full
`PASS` panel), and `tags`. Tags are 3-6 short kebab-case keywords.

`duplicates.json` is a flat list with entries of this form:

```json
{
  "finding": {"title": "...", "file": "...", "lines": [1, 2]},
  "candidate_index": 1,
  "duplicate_of": 123,
  "confidence": "high|medium|low",
  "reasoning": "..."
}
```

## 3. Fixedness of open issues

Only entries with `state == "open"` are eligible. Skip `pipeline-closed` and
`invalid` entries, entries matched by a duplicate finding, and entries whose
problem was accepted as a new finding.

For an eligible entry that carries a precomputed `fixedness` verdict, copy that
verdict into `fixedness.json` and do not start a panel for it.

For every other eligible entry at issue number `number`, spawn `N`
`icon4py-fixedness-checker` agents. Give them the checkout, the issue fields,
and output path `<findings_dir>/fixedness/<number>/<j>.json` for `j` from 1 to
`N`.

Each checker votes `fixed`, `persists`, or `unknown`. Merge these votes into
exactly one final result per eligible issue:

- all `N` votes `fixed` -> `fixed`
- a strict majority of votes `persists` -> `persists`
- all other outcomes -> `unknown`

Missing or malformed checker votes count as `unknown`.

Write `fixedness.json` as exactly one entry per eligible open issue:

```json
[{"number": 123, "verdict": "fixed|persists|unknown|not-detected"}]
```

The file must contain only these final merged verdicts, not individual checker
votes.
The host validates the complete vote directory for each positive decision. A
claimed unanimous or majority panel is not sufficient on its own. Invalid or
missing evidence prevents the affected issue creation, duplicate update, or
fixedness closure. It must not cause mutations for an unrelated valid candidate.
Do not claim that every vote is well formed unless each vote actually parses
and has the required fields.


## 4. Overview report

The report must contain exactly one completion marker:
`<!-- weekly-review:complete -->` when all assigned reviewers produced valid
result files, otherwise `<!-- weekly-review:incomplete -->`. The host rejects
an incomplete review and does not use it as a later diff baseline.

Write exactly one report to `<reports_dir>/report.md`.

Include:

- Counts for candidates, accepted findings, duplicates, rejected and uncertain
  candidates, and fixedness outcomes
- Accepted finding titles, severity, confidence, reviewer, file, and lines
- Duplicate findings with their target issue number and reasoning
- Rejected findings and their votes
- Uncertain findings with a note explaining the uncertainty
- Fixedness outcomes for eligible open issues
- Reviewer, checker, or parser failures

