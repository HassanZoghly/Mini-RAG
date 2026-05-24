#!/bin/bash
set -e

cd /app/models/db_schemes/minirag/

echo "Validating Alembic state..."

if alembic current > /dev/null 2>&1; then
    echo "Alembic state valid"
else
    echo "ERROR: Invalid or missing revision detected"
    exit 1
fi
