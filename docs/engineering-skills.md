# Engineering skills and plugins

Checked 2026-11-14. What we installed for the code side, why, and how to use it without wasting time or tokens. Design skills for the website are handled separately.

## The short version
- Most of what you need is already built in. Use it by name: `/code-review`, `/security-review`, `/simplify`, plus the engineering skills (testing-strategy, system-design, architecture, debug, deploy-checklist, tech-debt).
- We added 4 skills to `.claude/skills` (work on the laptop and in project threads) and turned on 5 plugins in `.claude/settings.json` (laptop only, one install step below).
- We left out the big "install everything" packs on purpose. Each skill adds its description to every session, and several of them force long routines on every task, including outreach and grants.

## What to use when
| You want to... | Use | Where it comes from |
|---|---|---|
| Fix a bug or a failing test | `systematic-debugging` skill: root cause first, max 3 fix attempts before rethinking | obra/superpowers (MIT), added |
| Stop Claude saying "done" without proof | `verification-before-completion` skill: run the test, show the output, then claim | obra/superpowers (MIT), added |
| Test the IK, retargeting, task file and URScript generator over many inputs | `property-based-testing` skill (Hypothesis). Example: FK(IK(pose)) == pose, task.json load(save(x)) == x | trailofbits/skills (CC BY-SA 4.0), added |
| Package the pipeline, add CI, start the SaaS backend | `modern-python` skill (pyproject, uv, ruff, pytest, pip-audit, Dependabot). Only when you ask; it won't migrate anything by itself | trailofbits/skills (CC BY-SA 4.0), added |
| Tests for robot code (mocks, sim, replay) | `robotics-testing` skill | already in repo |
| Plan how to test something new | `engineering:testing-strategy` | built in |
| Review a PR before you merge | `/code-review` (quick, built in). For a bigger PR: `/pr-review-toolkit:review-pr errors tests` on the laptop | built in / Anthropic plugin |
| Find swallowed errors (a servo fails and nothing complains) | pr-review-toolkit's `silent-failure-hunter` agent: "check this PR for silent failures" | Anthropic plugin |
| Security check of a change | `/security-review`. On the laptop the security-guidance plugin also warns on risky code as Claude writes it (`torch.load`, `pickle`, `yaml.load`, keys in code) and reviews each commit | built in / Anthropic plugin |
| Design the SaaS (upload video, queue, GPU worker, results) | `engineering:system-design`, then `engineering:architecture` for the record | built in |
| Before the first customer uses it | `engineering:deploy-checklist` | built in |
| Upload datasets or models to Hugging Face, track training runs | `hf-cli` and `huggingface-trackio` plugins on the laptop (private by default) | huggingface/skills (Apache-2.0) |
| Clean up messy code | `/simplify` | built in |

Name the skill in your ask when you want it for sure: "Rui, use systematic-debugging on the IK failure in test_retarget".

## Install on the laptop (once, 5 minutes)
1. Pull main after this PR is merged: `cd C:\dev\cobot-teaching` then `git pull`.
2. Install the Python language server used by pyright-lsp: `pip install pyright`.
3. Start Claude Code in that folder and accept "trust this folder". It offers the plugins from `.claude/settings.json`. If it doesn't, type:
```
/plugin install pr-review-toolkit@claude-plugins-official
/plugin install security-guidance@claude-plugins-official
/plugin install pyright-lsp@claude-plugins-official
/plugin marketplace add huggingface/skills
/plugin install hf-cli@huggingface-skills
/plugin install huggingface-trackio@huggingface-skills
/reload-plugins
```
4. Check with `/plugin` (Installed tab).

## Project threads (cloud)
- The 4 added skills and the agents load in project threads already.
- Plugins in `.claude/settings.json` do not load in cloud threads. If you want the PR review agents there too, add `pr-review-toolkit` in Project settings > Plugins. Not needed for now: the built-in `/code-review` covers most of it.
- Source: https://code.claude.com/docs/en/claude-projects.md (section "What threads pick up from your repositories").

## Using them efficiently
- Review once per PR, just before merging. Not after every change.
- `/code-review` has effort levels: `low` for small PRs, `high` for the URScript generator or anything that moves a robot.
- security-guidance is set to review at commit and push, not at the end of every reply (`ENABLE_STOP_REVIEW=0` in settings). This saves a lot of usage. Turn it back on (delete that line) once the SaaS takes customer videos.
- Property tests: 10 examples while working, 200 in CI (the skill explains how).
- On the laptop, run `/fewer-permission-prompts` after a week. It allow-lists the safe commands you approve again and again.
- Keep CLAUDE.md short. Every session reads it.

## Looked at and left out (for now)
| What | Source | Why not now |
|---|---|---|
| Full Superpowers plugin (14 skills) | https://github.com/obra/superpowers, MIT, ~318k stars | Its start-up hook makes Claude brainstorm before every task, including outreach and grants, and its strict TDD deletes code written before a test. Too heavy for research prototypes. We took its 2 best skills. |
| code-review plugin | https://github.com/anthropics/claude-plugins-official, Apache-2.0 | Same as the built-in `/code-review`. |
| feature-dev plugin | same | Overlaps with Rui and the built-in skills. |
| claude-security (deep scan) | same | Long, costly scan. Run it once before the first paying customer uploads data. |
| Trail of Bits insecure-defaults, supply-chain-risk-auditor, static-analysis | https://github.com/trailofbits/skills, CC BY-SA 4.0, ~8k stars | Useful when there is a web backend. Add at the SaaS stage. |
| huggingface-vision-trainer | https://github.com/huggingface/skills, Apache-2.0, ~11k stars | Fine-tunes a detector for customer parts (to replace the open LLMDet), but needs a paid HF plan (Jobs). Add when we train our own detector. |
| Sentry skills | https://github.com/getsentry/skills, Apache-2.0, ~1k stars | Written for Sentry's own workflow. Add Sentry when the SaaS runs and needs error tracking. |
| Big community packs (100+ agents) | various | Not vetted, lots of overlap, crowd out our own agents. |

Star counts from the GitHub search API on 2026-11-14.

## Next steps toward SaaS (when you're ready)
1. One `pyproject.toml` for product/teach and product/sim, tests run with one command (`modern-python`).
2. GitHub Actions: run the tests on every PR (`modern-python` covers CI, pip-audit and Dependabot).
3. Write the SaaS architecture down (`engineering:system-design`).
4. Before the first customer: `claude-security` scan and `engineering:deploy-checklist`.

## Adding or updating a skill
- Read it first. Check the source, licence and when it was last updated.
- Copy only the skill folder into `.claude/skills/<name>/` with its LICENSE and a SOURCE.md (link, commit, what you changed).
- Prefer a plugin when it brings agents or hooks and you only use it on the laptop.
