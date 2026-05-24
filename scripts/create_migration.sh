#!/bin/bash
set -e

MESSAGE=${1:-update}

cd /app/models/db_schemes/minirag/

echo "Creating migration: $MESSAGE"

alembic revision --autogenerate -m "$MESSAGE"
