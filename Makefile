# Variables
PYTHON=python3
VENV=venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

# Default target
all: help

# --- Create virtual environment and install dependencies ---
$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# --- Run training ---
train: $(VENV)
	$(PY) train.py

# --- Run inference ---
translate: $(VENV)
	$(PY) translate.py

# --- Clean temporary files ---
clean:
	rm -rf __pycache__ $(VENV)
	find . -name "*.pyc" -delete
	find . -name "*.pt" -delete

# --- Show help ---
help:
	@echo "Makefile commands:"
	@echo "  make train      # run training"
	@echo "  make translate  # run translation/inference"
	@echo "  make test       # run tests"
	@echo "  make clean      # remove temp files and virtual environment"
	@echo "  make all        # show this help"