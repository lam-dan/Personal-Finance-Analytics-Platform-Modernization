#!/bin/bash

# Development formatting script
# This script ensures your local code matches the CI environment exactly

set -e

echo "🔧 Running development formatting script..."
echo "This will format your code to match the CI environment exactly."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Verify tool versions
echo "📋 Verifying tool versions..."
EXPECTED_BLACK="black, 24.3.0"
EXPECTED_ISORT="isort, version 5.13.0"
EXPECTED_FLAKE8="7.0.0"

ACTUAL_BLACK=$(black --version)
ACTUAL_ISORT=$(isort --version | grep -o 'VERSION [0-9]\+\.[0-9]\+\.[0-9]\+' | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
ACTUAL_FLAKE8=$(flake8 --version | head -n1 | grep -o '^[0-9]\+\.[0-9]\+\.[0-9]\+')

if [[ "$ACTUAL_BLACK" != *"$EXPECTED_BLACK"* ]]; then
    echo "❌ Black version mismatch. Expected: $EXPECTED_BLACK, Got: $ACTUAL_BLACK"
    echo "💡 Run: pip install black==24.3.0"
    exit 1
fi

if [[ "$ACTUAL_ISORT" != "5.13.0" ]]; then
    echo "❌ isort version mismatch. Expected: 5.13.0, Got: $ACTUAL_ISORT"
    echo "💡 Run: pip install isort==5.13.0"
    exit 1
fi

if [[ "$ACTUAL_FLAKE8" != "$EXPECTED_FLAKE8" ]]; then
    echo "❌ flake8 version mismatch. Expected: $EXPECTED_FLAKE8, Got: $ACTUAL_FLAKE8"
    echo "💡 Run: pip install flake8==7.0.0"
    exit 1
fi

echo "✅ Tool versions verified"

# Run formatting tools in the same order as CI
echo "🎨 Running Black..."
black --line-length=79 python_service/ graphql_api/ db_migration/ observability/ tests/

echo "📦 Running isort..."
isort python_service/ graphql_api/ db_migration/ observability/ tests/

echo "🔧 Running autopep8..."
# Skip autopep8 for now due to Python 3.13 compatibility issues
echo "⚠️  Skipping autopep8 due to Python 3.13 compatibility"

echo "✅ Formatting complete!"

# Check if there are any changes
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Formatting changes detected:"
    git diff --stat
    echo ""
    echo "💡 Run 'git add . && git commit -m \"style: Apply formatting\"' to commit these changes"
else
    echo "🎉 No formatting changes needed!"
fi

echo "🧪 Running linting checks..."
make lint

echo "✨ Development formatting script completed!" 