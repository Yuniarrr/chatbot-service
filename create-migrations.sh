#!/usr/bin/env bash

# Ensure the script exits on errors
set -e

echo "Name migrations:"
read -r migrations

echo "Creating migrations"
alembic revision --autogenerate -m "$migrations"

echo "Migrations created successfully"
echo

echo "################################"
echo "Run 'alembic upgrade head' to apply migrations"
echo "Run 'alembic downgrade -1' to downgrade migrations"
echo "################################"
