BIN := uv run
PYTHON := $(BIN) python3

install:
	uv sync

run:
	$(PYTHON) -m src.main
	
debug:
	$(BIN) python3 -m pdb main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	flake8 src/
	$(BIN) mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs