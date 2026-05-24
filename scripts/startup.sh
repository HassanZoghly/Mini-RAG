#!/bin/bash
set -e

echo "Starting Mini-RAG..."

APP_ENV=${APP_ENV:-development}

if [ "$APP_ENV" = "production" ]; then
    echo "Production mode"

    /app/scripts/validate_migrations.sh
    /app/scripts/migrate.sh

else
    echo "Development mode"
    echo "Skipping automatic migrations"
fi

echo "Starting FastAPI server..."

exec "$@"
