# run clean tasks
[group("clean")]
clean: clean-build clean-coverage clean-coverage-html clean-pytest clean-python clean-ruff

# clean build folders
[group("clean")]
clean-build:
    uv run python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ['dist', 'build']]"

# clean coverage files
[group("clean")]
clean-coverage:
    uv run python -c "import os, glob; [os.remove(f) for f in glob.glob('.coverage*')]"

# clean coverage html
[group("clean")]
clean-coverage-html:
    uv run python -c "import shutil; shutil.rmtree('htmlcov', ignore_errors=True)"

# clean pytest cache
[group("clean")]
clean-pytest:
    uv run python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"

# clean python caches
[group("clean")]
clean-python:
    uv run python -c "import os, shutil; [shutil.rmtree(os.path.join(root, d), ignore_errors=True) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__']; [os.remove(os.path.join(root, f)) for root, _, files in os.walk('.') for f in files if f.endswith(('.pyc', '.pyo'))]"

# clean ruff cache
[group("clean")]
clean-ruff:
    uv run python -c "import shutil; shutil.rmtree('.ruff_cache', ignore_errors=True)"
