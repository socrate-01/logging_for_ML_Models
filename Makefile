.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
.PHONY: clean-pyc

venv: ## create virtual environment
	python3.9 -m venv venv
.PHONY: venv

dependencies: ## install dependencies from requirements.txt
	python -m pip install --upgrade pip
	python -m pip install --upgrade setuptools
	python -m pip install --upgrade wheel
	pip install -r requirements.txt
.PHONY: dependencies

test-dependencies: ## install dependencies from test_requirements.txt
	pip install -r test_requirements.txt
.PHONY: test-dependencies

update-dependencies:  ## Update dependency versions
	pip install pip-tools
	pip-compile requirements.in > requirements.txt
	pip-compile test_requirements.in > test_requirements.txt
	pip-compile service_requirements.in > service_requirements.txt
.PHONY: update-dependencies

clean-venv: ## remove all packages from virtual environment
	pip freeze | grep -v "^-e" | xargs pip uninstall -y
.PHONY: clean-venv

test: clean-pyc ## Run unit test suite.
	pytest --verbose --color=yes ./tests
.PHONY: test

test-reports: clean-pyc clean-test ## Run unit test suite with reporting
	mkdir -p reports
	mkdir ./reports/unit_tests
	mkdir ./reports/coverage
	mkdir ./reports/badge
	-python -m coverage run --source data_enrichment -m pytest --verbose --color=yes --html=./reports/unit_tests/report.html --junitxml=./reports/unit_tests/report.xml ./tests
	-coverage html -d ./reports/coverage
	-coverage-badge -o ./reports/badge/coverage.svg
	rm -rf .coverage
.PHONY: test-reports

clean-test:	## Remove test artifacts
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf reports
	rm -rf .pytype
.PHONY: clean-test

check-codestyle:  ## checks the style of the code against PEP8
	pycodestyle data_enrichment --max-line-length=120
.PHONY: check-codestyle

check-docstyle:  ## checks the style of the docstrings against PEP257
	pydocstyle data_enrichment
.PHONY: check-docstyle

check-security:  ## checks for common security vulnerabilities
	bandit -r data_enrichment
.PHONY: check-security

check-dependencies:  ## checks for security vulnerabilities in dependencies
	safety check -r requirements.txt
.PHONY: check-dependencies

check-codemetrics:  ## calculate code metrics of the package
	radon cc data_enrichment
.PHONY: check-codemetrics

check-pytype:  ## perform static code analysis
	pytype data_enrichment
.PHONY: check-pytype

convert-post:  ## Convert the notebook into Markdown file
	jupyter nbconvert --to markdown blog_post/blog_post.ipynb --output-dir='./blog_post'
.PHONY: convert-post

build-image:  ## Build docker image
	DATE_CREATED=(date +"%Y-%m-%d %T") && \
	REVISION=(git rev-parse HEAD) && \
	docker build \
	  --build-arg DATE_CREATED="$DATE_CREATED" \
	  --build-arg VERSION="0.1.0" \
	  --build-arg REVISION="$REVISION" \
	  -t credit_risk_model_service:0.1.0 .
.PHONY: build_image
