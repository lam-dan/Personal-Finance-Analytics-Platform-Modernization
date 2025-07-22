import subprocess  # Used to run the migration script as a subprocess


def test_migration_script_runs():
    """
    Test that the data migration script runs successfully without errors.
    This ensures the script exits cleanly and outputs a success message.
    """
    # Run the migration script as a subprocess and capture output
    result = subprocess.run(
        ["python", "db_migration/migrate.py"], capture_output=True, text=True
    )

    # Assert that the script exits with a success code (0)
    assert result.returncode == 0

    # Assert that the expected success message is present in the stderr
    # (logging output)
    assert "Transactions migrated successfully" in result.stderr
