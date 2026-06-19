# Security Policy

## Reporting a vulnerability

Please report security issues privately. **Do not open a public issue for
anything exploitable.**

- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" under the **Security** tab), or
- Contact the maintainers directly.

We aim to acknowledge a report within 5 business days and to agree on a
disclosure timeline with the reporter. Please give us a reasonable window to
ship a fix before any public disclosure.

## Scope

This is an anti-money-laundering platform that handles transaction and account
data and produces risk decisions. We are particularly interested in:

- Secret or credential leakage (tokens, keys, connection strings).
- Authentication/authorization gaps on the dashboard, Trino, or MinIO.
- Injection or data-exfiltration paths through the LLM explainer (prompt
  injection, leaking other accounts' data into a narrative).
- Tampering with scores, the blocklist feedback loop, or model artifacts.

## Secrets

No secret may be committed to this repository.

- The LLM key lives in `.env`, which is gitignored; only `.env.example` is
  tracked. GitHub tokens are used inline at push time and never stored in the
  repo or in git remotes.
- If a secret is ever committed, treat it as compromised: **rotate it
  immediately**, then purge it from history.
- Run a secret scan before pushing. A pre-commit hook (e.g. `gitleaks`) is
  recommended.

## Data and models

- Do not commit real customer or production data. Synthetic data and the public
  IBM AML (AMLWorld) dataset are the only sanctioned inputs.
- Model risk: scores are calibrated probabilities, not verdicts. See
  [`docs/model_card.md`](docs/model_card.md) for intended use and limitations.

## Supply chain

Dependencies are pinned to stable releases at least one week old, verified
against the package registry rather than from memory.
