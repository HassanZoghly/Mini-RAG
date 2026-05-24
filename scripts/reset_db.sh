#!/bin/bash
set -e

echo "WARNING: Resetting database"

cd /app/models/db_schemes/minirag/

alembic downgrade base || true
alembic stamp base
alembic upgrade head

echo "Database reset completed"
