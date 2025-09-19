.PHONY: help install install-dev format lint test train xai data-download dvc-push dvc-pull

help:
	@echo "Makefile targets:"
	@echo "  make install        # install runtime requirements"
	@echo "  make install-dev    # install dev requirements"
	@echo "  make format         # run black"
	@echo "  make lint           # run flake8"
	@echo "  make test           # run pytest"
	@echo "  make train          # run local smoke training (run.py)"
	@echo "  make xai            # run local xai demo"
	@echo "  make data-download  # download dataset via Kaggle (script)"
	@echo "  make dvc-push       # push tracked DVC artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

format:
	black .

lint:
	flake8 src tests

test:
	pytest -q

train:
	python run.py train

xai:
	python run.py xai

data-download:
	bash scripts/download_data.sh

dvc-push:
	dvc push
