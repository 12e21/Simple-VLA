## Intro
- this is project to implement a simple structure vision language action model to help me learn the "transformers" frame

## Env
- mainly use uv, transformers, pytorch, lerobot
- use python directly to run script, do not use uv run

## Git
- develop on the "develop" branch
- pre-commit already set up
- update CHANGELOG.md after each commit with changes made

## Testing
- this project follow Test-Driven-Development
- Fake test data fixtures matching LeRobot dataset format in `tests/conftest.py`, use `pytest --fixtures` to check
- SimpleVLAPolicy tests in `tests/test_simple_policy.py` - encode_image, encode_text, forward methods

## Memo
- when say "memo", it is CLAUDE.md
- every thing updated to CLAUDE.md must be concise, better to use a short sentence to give a instruction
