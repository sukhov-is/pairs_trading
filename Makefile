.PHONY: install test lint format docs clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[docs]"

test:
	pytest tests/ --cov=src

lint:
	flake8 src/ tests/
	mypy src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

docs:
	cd docs && make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete