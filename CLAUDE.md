# Instructions for Claude Code sessions on this repo
Read docs/BRIEF.md and README.md first. Non-negotiables:
- Never add or edit a FRED series ID without running `python -m infcomp verify` and showing Tim the returned title/units. Never set `confirmed=TRUE` yourself.
- Never edit config/gate.yaml after validation results exist; a new gate needs a new commit and date.
- Parameters in settings.yaml are choices, not calibrated facts — say so when reporting results.
- Keep outputs from `demo` (synthetic) out of anything client-facing.
- Run `pytest -q` before committing.
