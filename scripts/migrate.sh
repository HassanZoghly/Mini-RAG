#!/bin/bash
set -e

cd /app/models/db_schemes/minirag/

echo "Running migrations..."

if alembic upgrade head; then
    echo "Migration completed successfully"
else
    echo "Migration failed"
    exit 1
fi
