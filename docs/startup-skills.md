# Startup skills (business side)

Checked 2026-11-15. These skills help with the parts of the startup that aren't code: selling, pricing, contracts, research and specs. They add to the agents you already have (Marco, Iris, Olivia, Greta, Pedro, Petra) and don't replace them. For the code side, see docs/engineering-skills.md.

All 7 are in `.claude/skills`, so they work in project threads and on the laptop with no install. Each folder has its LICENSE and a SOURCE.md.

## What to use when
| You want to... | Say | Skill | Works with |
|---|---|---|---|
| Decide what to charge: the "new task in a week" price now, SaaS tiers later | "use pricing: what should the week-long deployment cost, and how do we move to SaaS?" | pricing | Pedro, Petra |
| Make a one-pager, demo script, objection answers or a battle card for integrators | "make a one-pager for Growskills" or "how do I answer 'we already program URs in a day'?" | sales-enablement | Olivia, Pedro |
| Find patterns across several interviews and see what you can safely conclude | "what do the interviews so far really say about who pays?" | customer-research | Iris (one interview at a time) |
| Keep dated, sourced profiles of Micropsi, Mimic, RoboTwin and others | "profile Micropsi, facts apart from guesses" | competitor-profiling | Marco |
| Check an NDA from an integrator or plant before you sign it | "triage this NDA" plus the file | triage-nda | |
| Review a pilot or partner contract: red flags, redlines, what to push back on | "review this pilot agreement" plus the file | review-contract | Pedro |
| Turn a feature idea into a clear spec (goals, non-goals, acceptance tests) | "write a spec for the customer video upload page" | write-spec | Rui |

## Where they come from
| Skills | Source | Licence | Why we trust it |
|---|---|---|---|
| pricing, sales-enablement, customer-research, competitor-profiling | https://github.com/coreyhaines31/marketingskills | MIT | ~60k stars, the most-used marketing skill pack, updated weekly |
| triage-nda, review-contract | https://github.com/anthropics/knowledge-work-plugins (legal) | Apache-2.0 | Anthropic's official plugin, ~36k stars |
| write-spec | same (product-management) | Apache-2.0 | Anthropic's official plugin |

Each one has a short "In this repo" note at the top. It tells the skill to read docs/context.md, cite sources, save drafts in the repo and never send anything.

Star counts from the GitHub search API on 2026-11-14.

## Things to know
- The contract skills are not legal advice. They catch the usual traps (a hidden non-compete or non-solicit, one-sided IP, unlimited liability). Anything they mark YELLOW or RED goes to a lawyer. Once you know your standard positions, write them in `docs/legal-playbook.md` and both skills will use them.
- Pricing: competitor prices need a source link, as the CLAUDE.md rule says. Our earlier scan found nobody publishes a per-task price, so expect ranges, not exact figures.

## Looked at and left out
| What | Source | Why not |
|---|---|---|
| Product Manager Skills (100+ skills) | https://github.com/deanpeters/Product-Manager-Skills, ~8k stars | Licence is CC BY-NC-SA, which means **no commercial use**. A startup can't use it. |
| Founder pack (board, consumer panel, CFO...) | https://github.com/Jakeschincariol/founder-skill, MIT, ~3.7k stars | Built for consumer businesses (cafés, shops). Its "100 simulated buyers" are made-up data and cost a lot of tokens. For a market with a few dozen integrators, real interviews are better. Petra already plays the sceptical board. |
| Anthropic sales plugin (40 skills) | knowledge-work-plugins | Built for sales teams with a CRM. Too much for one person with a CSV pipeline. |
| Anthropic small-business plugin (50 skills) | same | US-style payroll, tax and invoicing. Revisit when the company has an accountant and revenue. |
| Whole marketing pack (50 skills) | coreyhaines31/marketingskills | We took the 4 that fit B2B industrial sales. SEO, launch and ads come later, once the website is public. |
| 400-skill mega packs | e.g. alirezarezvani/claude-skills | Not vetted, lots of overlap. |

## Later, when it fits
- **Website goes public:** marketing skills `seo-audit`, `launch` and `copywriting` from the same MIT pack.
- **SaaS backend:** the official `supabase` and `stripe` plugins (Project settings > Plugins). Supabase is already connected to your account.
- **Cloud credits:** the official `aws-startup-advisor` plugin, when you pick a cloud for the GPU workers.
- **First real contracts:** fill in `docs/legal-playbook.md` (NDA terms, liability cap, IP, payment terms).
