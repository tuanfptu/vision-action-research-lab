PY ?= .venv-gpu/bin/python
.PHONY: test lint smoke
test:
	env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
smoke:
	env -u PYTHONPATH $(PY) -m va_lab.cli doctor --require-gpu
