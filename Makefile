# Common tasks. Works after `pip install -e ".[dev]"`, or with only the requirements installed (src/ is put on PYTHONPATH).
# The experiment targets need the Mip-NeRF 360 data (see data/README.md) and a GPU; their shell scripts activate .venv.
PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: help install test lint format run_experiments tables figures slides report rebuild

help:
	@echo "install     pip install -e .[dev]"
	@echo "test        run the unit tests"
	@echo "lint        ruff check"
	@echo "format      ruff format and fix import order"
	@echo "run_experiments  every experiment in the report (GPU, about a day, resumable)"
	@echo "tables      rebuild docs/results.md from results/"
	@echo "figures     rebuild docs/figures from results/"
	@echo "slides      rebuild docs/slides.pptx"
	@echo "report      rebuild docs/report.pdf from docs/report.md"
	@echo "rebuild     tables, figures, slides and the report PDF"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --select I --fix .
	$(PYTHON) -m ruff format .

run_experiments:
	scripts/run_experiments.sh

# collect.py rewrites docs/results.md from results/; refuse when results/ is empty (e.g. a fresh clone),
# so the committed tables are not wiped.
tables:
	@ls results/*/*/*/metrics.json > /dev/null 2>&1 || { echo "results/ has no runs; not overwriting docs/results.md"; exit 1; }
	$(PYTHON) scripts/collect.py

figures: tables
	$(PYTHON) scripts/make_figures.py

slides: tables
	$(PYTHON) scripts/build_slides.py

report:
	$(PYTHON) scripts/build_report_pdf.py

rebuild: figures slides report
