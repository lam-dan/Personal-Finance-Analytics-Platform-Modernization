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
EXPECTED_ISORT="5.13.0"

ACTUAL_BLACK=$(black --version)
ACTUAL_ISORT=$(isort --version | grep -o 'VERSION [0-9]\+\.[0-9]\+\.[0-9]\+' | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')

if [[ "$ACTUAL_BLACK" != *"$EXPECTED_BLACK"* ]]; then
    echo "❌ Black version mismatch. Expected: $EXPECTED_BLACK, Got: $ACTUAL_BLACK"
    echo "💡 Run: pip install black==24.3.0"
    exit 1
fi

if [[ "$ACTUAL_ISORT" != "$EXPECTED_ISORT" ]]; then
    echo "❌ isort version mismatch. Expected: $EXPECTED_ISORT, Got: $ACTUAL_ISORT"
    echo "💡 Run: pip install isort==5.13.0"
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
# Run type checking and security scans
echo "Running type checking..."
mypy python_service/ graphql_api/ db_migration/ observability/
echo "Running security scan..."
bandit -r python_service/ graphql_api/ db_migration/ observability/
echo "✅ Linting checks completed"

echo "✨ Development formatting script completed!" 