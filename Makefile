.PHONY: help up down logs ps restart test test-backend test-frontend dev-backend dev-frontend clean

help:
	@echo "make up            build and start backend (:8000) + frontend (:3000)"
	@echo "make down          stop the stack (data volume is kept)"
	@echo "make logs          follow logs"
	@echo "make test          run backend + frontend tests"
	@echo "make clean         stop the stack and delete the SQLite/cache volume"

.env:
	@cp .env.example .env
	@echo ">> created .env from .env.example - put your real API keys in LLM_API_KEYS"

up: .env
	docker compose up -d --build
	@echo ""
	@echo "Frontend: http://localhost:3000"
	@echo "Backend:  http://localhost:8000/health"

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

restart: down up

test: test-backend test-frontend

test-backend:
	cd backend && python -m pytest

test-frontend:
	cd frontend && npm ci --no-audit --no-fund && npm test

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

clean:
	docker compose down -v
