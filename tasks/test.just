# run all tests with coverage
[group("test")]
test cov_fail_under="70":
    uv run python -c "from pathlib import Path; import shutil; Path('.coverage').unlink(missing_ok=True); shutil.rmtree('htmlcov', ignore_errors=True)"
    uv run pytest --cov={{SOURCES}}/pyRPC3 --cov-report=html:htmlcov --cov-report=term-missing --cov-fail-under={{cov_fail_under}} {{TESTS}} -v
