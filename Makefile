PYTHON := .venv/bin/python
export PYTHONPATH := src
export MPLCONFIGDIR := .cache/matplotlib

.PHONY: prepare eda run rolling publish applied propensity propensity-features
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
applied:
	$(PYTHON) -m ranking.cli applied --config configs/hm_sample.json
propensity:
	$(PYTHON) -m ranking.cli propensity --config configs/propensity.json
propensity-features:
	$(PYTHON) -m ranking.cli propensity --config configs/propensity.json --features-only
