PY := .venv/bin/python
.PHONY: test lint smoke
test:
	env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
smoke:
	.venv/bin/va-lab doctor
	.venv/bin/va-lab generate
	.venv/bin/va-lab train --data data/synthetic.npz --output runs/smoke
