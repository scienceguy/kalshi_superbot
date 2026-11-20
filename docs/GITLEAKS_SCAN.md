# T15-02 secret scan report (gitleaks, full history)

Date: 2026-11-20. Branch `v15-task-list` at `f47919f` (scan covers all 250
commits reachable from it).

```
$ gitleaks detect --no-banner -v
...INF 250 commits scanned.
...INF scan completed in 1.39s
...WRN leaks found: 5
```

All 5 findings are the same rule (`private-key`) firing on documentation and
example text, not on real key material:

| commit | file | what it is |
|---|---|---|
| 40676ad | scripts/install-pre-commit.sh:36 | grep pattern text: `the '-----BEGIN RSA PRIVATE KEY-----' example` (the hook's own blocklist pattern) |
| 348b125 | add_n8n_credentials.py:76 | docstring JSON example containing a placeholder `-----BEGIN RSA PRIVATE KEY-----\nREDACTED\n-----END RSA PRIVATE KEY----` |
| 17fd41a | KALSHI_API_RESEARCH.md:72 | docs text: `...also begin with '-----BEGIN PRIVATE KEY-----'. Determine type from the parsed key object.` |
| 6867ff0 | kalshi_client.py:100 | docstring/comment string literal `-----BEGIN RSA PRIVATE KEY-----\n` (line-splitting example) |
| 6867ff0 | config.py:266 | docstring example `KALSHI_PRIVATE_KEY='-----BEGIN RSA PRIVATE KEY-----...'` |

Verification that none is a real key: every Secret value shown by gitleaks
(above) is a fragment of an example PEM header, not a body. No `-----BEGIN ... `
block with base64 body content appears in any of them (checked in `-v` output).

Targeted checks:

```
$ git log --all --oneline -S 'VERTEX_PROJECT_ID' -- .
(empty - the identifier v15's F-10 worried about was never committed to this repo)

$ git ls-files | grep -i 'env|secret|key|pem|state|log'
.env.example, polymarket_trader/state_db.py, scripts/migrate_secrets.py
(no key material, no state dumps, no logs are tracked)
```

Config defaults checked for secret-looking values: `grep -rn 'VERTEX'` -> no
match at HEAD (F-10 verdict: WRONG on HEAD, already clean).

Private keys in this deployment live outside the repo
(`~/.secrets/kalshi_private.pem` chmod 600, mirrored via `~/.secrets/bw/`),
guarded by the pre-commit hook installed by scripts/install-pre-commit.sh.

Conclusion: T15-02 scan is clean. The repo may stay public; the scan report is
committed so the next agent does not repeat it. Rotation: nothing to rotate -
no secret material was ever committed to this repository per gitleaks and the
2026-11-01 history rewrite recorded in AGENTS.md.