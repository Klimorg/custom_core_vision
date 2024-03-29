# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "install               : installs project requirements."
	@echo "install-dev           : installs development requirements."
	@echo "install-docs          : installs docs requirements."
	@echo "clean                 : cleans all unecessary files."
	@echo "docs                  : serve generated documentation from mkdocs."
	@echo "tests                 : run unit tests."
	@echo "mypy                  : run mypy in the src folder for type hinting checking."
	@echo "cc_report             : run radon in the src folder for code complexity report."
	@echo "raw_report            : run radon in the src folder for raw report."
	@echo "mi_report             : run radon in the src folder for maintainability index report."
	@echo "hal_report            : run radon in the src folder for hal report."
	@echo "install_precommit     : installs precommit."
	@echo "check_precommit       : check precommit."

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev: install
	python -m pip install -e ".[dev]" --no-cache-dir
	python -m pip install -e ".[test]" --no-cache-dir
	python -m pip install -e ".[doc]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

.PHONY: install-docs
install-docs:
	python -m pip install -e ".[docs]" --no-cache-dir

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".hypothesis" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E "*.egg-info" | xargs rm -rf
	find . | grep -E "htmlcov/*" | xargs rm -rf
	find . | grep -E "build/*" | xargs rm -rf
	find . | grep -E "dist/*" | xargs rm -rf
	rm -f .coverage

# Documentation
.PHONY: docs
docs:
	mkdocs serve

# Tests
.PHONY: tests
tests:
	python -m pytest -v --cov log_cli=true
# Reporting
.PHONY: mypy
mypy:
	mypy --show-error-codes core_vision/

.PHONY: cc_report
cc_report:
	radon cc core_vision/

.PHONY: raw_report
raw_report:
	radon raw --summary core_vision/

.PHONY: mi_report
mi_report:
	radon mi core_vision/

.PHONY: hal_report
hal_report:
	radon hal core_vision/

.PHONY: bandit
bandit:
	bandit -r core_vision/


# Precommit
.PHONY: install_precommit
install_precommit:
	pre-commit install

.PHONY: check_precommit
check_precommit:
	pre-commit run --all

# Docker
.PHONY: build_docker
build_docker:
	docker build \
	--build-arg USER_UID=$$(id -u) \
	--build-arg USER_GID=$$(id -g) \
	--rm \
	-f Dockerfile \
	-t core_vision:v1 .

.PHONY: run_docker
run_docker:
	docker run --gpus all \
	--shm-size=2g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-it \
	--rm \
	--mount type=bind,source=$(PWD),target=/home/$(USERNAME)/core_vision \
	-e TF_FORCE_GPU_ALLOW_GROWTH=true \
	-e TF_ENABLE_ONEDNN_OPTS=true \
	-e XLA_FLAGS='--xla_gpu_autotune_level=2' \
	core_vision:v1
