INSTALL_STAMP := .venv/.install.stamp
UV := $(shell command -v uv 2> /dev/null)

help:
	@echo "Please use 'make <target>' where <target> is one of the following commands.\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo "\nCheck the Makefile to know exactly what each target is doing."

$(INSTALL_STAMP): pyproject.toml uv.lock  ## Install dependencies
	@if [ -z $(UV) ]; then echo "uv could not be found. See https://docs.astral.sh/uv/"; exit 2; fi
	$(UV) --version
	$(UV) sync --locked
	touch $(INSTALL_STAMP)

.PHONY: format
format: $(INSTALL_STAMP)  ## Format code base
	$(UV) run ruff check --fix *.py ai_window_prompts_updater/
	$(UV) run ruff format *.py ai_window_prompts_updater/

.PHONY: lint
lint: $(INSTALL_STAMP)  ## Analyze code base
	$(UV) run ruff check *.py ai_window_prompts_updater/
	$(UV) run ruff format --check *.py ai_window_prompts_updater/

.PHONY: start
start: $(INSTALL_STAMP)  ## Start the script
	$(UV) run python script.py

.PHONY: test
test: $(INSTALL_STAMP)  ## Unit tests
	$(UV) run py.test test.py

