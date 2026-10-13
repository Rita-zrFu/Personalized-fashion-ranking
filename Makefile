PYTHON := .venv/bin/python
export PYTHONPATH := src
export MPLCONFIGDIR := /tmp/ranking-matplotlib

.PHONY: prepare eda run rolling publish
prepare:
	$(PYTHON) -m ranking.cli prepare --max-users 50000
eda:
	$(PYTHON) -m ranking.cli eda
run:
	$(PYTHON) -m ranking.cli run --config configs/hm_sample.json
rolling:
	$(PYTHON) -m ranking.cli rolling --config configs/hm_sample.json
publish:
	$(PYTHON) -m ranking.cli publish
