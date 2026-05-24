#!/bin/bash
set -e

echo "Starting Mini-RAG..."

APP_ENV=${APP_ENV:-development}

cd /app/models/db_schemes/minirag/

if [ "$APP_ENV" = "production" ]; then

    echo "Production mode detected"
    echo "Running database migrations..."

    if alembic current > /dev/null 2>&1; then

        echo "Alembic state is valid"

        if alembic upgrade head; then
            echo "Migrations completed successfully"
        else
            echo "Migration failed!"
            exit 1
        fi

    else
        echo "WARNING: Invalid alembic revision detected"

        echo "Attempting recovery..."

        if alembic stamp head; then
            echo "Database stamped successfully"
        else
            echo "Recovery failed"
            exit 1
        fi
    fi

else

    echo "Development mode detected"
    echo "Skipping automatic migrations"

fi

cd /app

echo "Starting application..."

exec "$@"
