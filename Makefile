.PHONY: up down logs test build install clean

install:
	pip install -r requirements.txt

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f app

test:
	pytest -v

build:
	docker compose build

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
