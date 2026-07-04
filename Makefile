.PHONY: install infra-up infra-down start test demo clean

install:
	pip install -r requirements.txt

infra-up:
	docker compose up --build

infra-down:
	docker compose down

infra-clean:
	docker compose down -v

start:
	uvicorn argus_core.main:app --port 8001 --reload

start-prod:
	uvicorn argus_core.main:app --port 8001 --workers $(ARGUS_WORKERS)

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=argus_core --cov-report=term-missing

demo:
	python demo/real_degradation_demo.py

demo-synthetic:
	python demo/synthetic_drift_demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
