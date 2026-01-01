dev:
	pip install -e ".[dev]"

test:
	pytest -q

fmt:
	black .

lint:
	ruff .

check:
	black --check . && ruff . && pytest -q
