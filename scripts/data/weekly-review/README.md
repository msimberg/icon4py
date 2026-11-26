# Weekly icon4py review

This pipeline reviews icon4py and publishes accepted findings as GitHub issues.
The enabled areas are architecture, correctness, and performance. Findings must
meet the requested severity and pass the configured skeptic panel.

## Stages

The host driver is `scripts/python/weekly_review.py`:

1. Collect issues labeled `ai-review` and compute a diff baseline for the
   selected branch.
2. Create a fresh run workspace and install pi's packages on the host.
3. Run AWF with the selected model and reviewers.
4. Validate the findings, then create or update GitHub issues from the host.
5. Upload the report, findings, statistics, and session transcripts.

The `SKILL.md` file defines the reviewer instructions. Reviewers emit JSON with
finding text, a relative source path, line numbers, and a suggested fix. The
host validates field types and severity before publication. A malformed
A list of strings in `description`, `evidence`, or `suggested_fix` is converted
into Markdown bullets. Other malformed candidates are logged and skipped
without stopping the run.

The orchestrator writes exactly one merged fixedness verdict per eligible open
issue. All panel votes of `fixed` close the issue. At least two `persists` votes
add a comment. Other verdicts leave it open. No fixedness panels run for closed
or human-rejected issues.

## Isolation and credentials

AWF runs the agent, not the host driver. It provides filesystem isolation and
an API proxy approved for the configured model.

- The host prepares pi packages before entering AWF. Network-isolated runs do
  not download packages.
- The upstream `OPENAI_API_KEY` is given to AWF's proxy on the host.
- The agent receives only the local proxy endpoint and the fixed placeholder
  `awf-placeholder`. The upstream key is never part of worker configuration.
- AWF rate limiting is disabled. Upstream request limits apply. Pi retries
  retryable model requests with backoff.
- The agent has no GitHub network path and receives no GitHub token. The host
  performs all issue reads and writes.

Checkout uses `persist-credentials: false`. Only the fresh workspace is
writable by the agent; the rest of the checkout is read-only. Using
`persist-credentials: false` does not store an automatic token in the checkout.

AWF supplies the endpoint `AWF_API_PROXY_OPENAI_URL`. The local worker
`agent_entry.py` turns it into a provider-neutral OpenAI-compatible endpoint.
The worker needs no knowledge of the CSCS endpoint or its API key.

## CLI

All review-defining options are required on `run`:

- `--checkout`: checkout to review
- `--repo`: issue repository as `owner/name`
- `--severity`: `high`, `medium`, or `low`
- `--panel-size`: positive vote count
- `--reviewer`: `architecture`, `correctness`, or `performance`; repeat for each
  enabled reviewer
- `--model-id`: exact model id
- `--model-base-url`: upstream OpenAI-compatible base URL

`--artifacts-dir` selects the parent for the fresh workspace. If omitted, the
driver creates one under `_local/` in the checkout. Each workspace gets a
unique timestamped directory, so results and statistics cannot mix across runs.
The optional `--automation-actor` identifies the account whose closures may be
reopened. The default is the current token identity. Dry runs without that
identity warn when closed issues cannot be classified.

Without `--publish`, the driver reads the tracker and prints planned actions
but does not change issues. Publishing needs `GITHUB_TOKEN` or `GH_TOKEN` and a
repository with Issues enabled. A publication check fails before the model run
if Issues are disabled.

Example, with all three reviewers enabled:

```bash
./scripts/run weekly-review run \
  --checkout /absolute/path/to/icon4py \
  --repo msimberg/icon4py \
  --severity high \
  --panel-size 3 \
  --reviewer architecture \
  --reviewer correctness \
  --reviewer performance \
  --model-id zai-org/GLM-5.3-Flash \
  --model-base-url https://api.inference.cscs.ch/v1 \
  --artifacts-dir /tmp/weekly-review-runs
```

Add `--publish` to apply reconciled changes. Create labels separately if needed:

```bash
GITHUB_TOKEN=... ./scripts/run weekly-review labels --repo msimberg/icon4py
```

`awf`, `pi`, `rg`, and `fd` must also be on `PATH` (or use `AWF_BIN`). In CI,
the setup action installs AWF at a pinned revision before the run. It also
installs `ripgrep` and `fd-find` and creates the `fd` command link.

## Issue lifecycle

- The issue tracker is the ledger. Later reviews receive existing issues and
  newly rejected findings as context.
- Invalid and open issues have a `number`, title, description, source location,
  suggested fix, and optional `pipeline-closed` state; invalid entries only
  prevent future equivalent findings.
- A duplicate against an open issue adds a comment and updates `last_seen` in
  the metadata without rewriting human text.
- A duplicate against a pipeline-closed issue reopens it once per run, then
  comments. Later duplicate matches in the same run only comment.
- A human close leaves the original issue closed. It is not a request to
  suppress equivalent findings in the future; use `ai-review:invalid` for that.
- A closure counts as pipeline-closed only when the **latest** closed event was
  made by the current automation actor and is no more than 30 days old. A later
  human close wins over earlier automation history.
- Outside the 30-day window, an independent accepted finding may create a new
  issue.

Labels are created automatically:

| Label | Meaning |
| --- | --- |
| `ai-review` | Managed by the review pipeline |
| `ai-review:needs-triage` | No human has checked the finding |
| `ai-review:confirmed` | A human confirmed it |
| `ai-review:invalid` | A human rejected it; suppress later duplicates |

## Issue format and report

An accepted finding has text fields, a source path, line range, reviewer,
confidence, and tags. The host builds Summary, Evidence, Suggested fix, and
Details sections. A hidden `weekly-review:metadata` block stores source fields
used by later reviews. Location links pin to the detected commit:

```text
https://github.com/<owner>/<repo>/blob/<commit>/<path>#L<start>-L<end>
```

Later runs recover the metadata, Summary, Suggested fix, issue number, title,
state, and labels. This is intentionally not a full reconstruction of the raw
finding JSON. The issue number is identity; there is no fingerprint.

The host reads the current tracker again before applying results. That uses
newer labels and states without claiming atomicity with the panel's reasoning.

Each report contains the orchestrator's outcome, publication actions, elapsed
time, and per-session token usage. Reports and result JSONs are kept directly
in the run workspace so failures retain diagnostics. `statistics.json` is the
machine-readable statistics file.

A report includes a completion marker. Incomplete reviews fail and are not
used as a later diff baseline. GitHub updates are rejected when any assigned
reviewer output is missing or malformed.
## Diff baseline

The baseline is the head commit of the previous successful run on the same
branch. If no such commit is available locally, the driver uses a full-checkout
review. A missing branch or no suitable previous run also means no baseline.

Repository or checkout errors are failures. Failure to run `git diff` is not
treated as an empty diff.

## Files

```
.github/workflows/
  weekly-review.yml           # CI workflow
scripts/python/
  weekly_review.py            # Host driver and required CLI
scripts/data/weekly-review/
  agent_entry.py              # Sandbox-side worker
  SKILL.md                    # Orchestrator workflow
  agents/                     # Reviewer, skeptic, and fixedness agents
  settings.json               # Pi package declarations
  README.md                   # This document
```
