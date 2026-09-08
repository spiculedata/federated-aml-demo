# Packaging the federated demo.
#
#   make wheel       distributable wheel - compiled extensions, no Python source
#   make binary      small executable launcher (needs the venv alongside it)
#   make standalone  117 MB single file - bundles CPython, Polars and XGBoost
#   make install     build + install the compiled package into the venv
#   make test        run the suite against the source tree
#   make verify      run the suite with sources hidden, against .so only
#   make clean       remove every build artefact

PY       := .venv/bin/python
CYTHON   := .venv/bin/cython
UV       := uv
INCDIR   := $(shell $(PY) -c "import sysconfig;print(sysconfig.get_paths()['include'])")
LIBDIR   := $(shell $(PY) -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
PYVER    := $(shell $(PY) -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
BINARY   := dist/fedxgb-demo

.PHONY: all ext wheel binary standalone install test verify run clean

all: wheel binary

## Compile fedxgb/*.py into native extension modules, in place.
ext:
	$(PY) setup.py build_ext --inplace

## A normal pip-installable wheel containing only __init__.py and the .so files.
wheel:
	rm -rf dist-wheel
	$(UV) build --wheel --out-dir dist-wheel .
	@echo "--- wheel contents ---"
	@unzip -l dist-wheel/*.whl | grep -E '\.(py|so|c)$$'

## Install the compiled package into the venv, replacing any previous copy.
install: wheel
	-$(UV) pip uninstall fedxgb-demo
	rm -rf .venv/lib/python$(PYVER)/site-packages/fedxgb
	$(UV) pip install --no-deps dist-wheel/*.whl

## Compile the entry point to C and link it into a native executable.
## Installed into .venv/bin so CPython finds the venv's site-packages via
## pyvenv.cfg - that is what lets the binary run with no PYTHONPATH set.
binary: install
	mkdir -p build dist
	$(CYTHON) --embed -3 run_demo.py -o build/run_demo.c
	cc build/run_demo.c -o $(BINARY) -O2 \
		-I$(INCDIR) -L$(LIBDIR) -lpython$(PYVER) -Wl,-rpath,$(LIBDIR)
	cp $(BINARY) .venv/bin/fedxgb-demo
	@echo "--- built ---"
	@file $(BINARY)

## Fully self-contained single file: bundles CPython, Polars and XGBoost too.
##
## Two traps, both silent:
##  1. The entry point is staged into an empty directory first, so `fedxgb`
##     resolves to the compiled package in site-packages rather than the
##     adjacent .py sources - otherwise PyInstaller bundles the source.
##  2. PyInstaller finds imports by parsing Python source, and cannot see
##     inside a compiled .so. Anything imported only by a compiled module is
##     invisible to it: our own aggregator/weights (hence --collect-submodules)
##     and xgboost itself (hence --hidden-import, which makes PyInstaller walk
##     xgboost's own sources; --collect-all trips over xgboost.testing).
standalone: install
	rm -rf build/standalone dist-standalone
	mkdir -p build/standalone
	cp run_demo.py build/standalone/
	.venv/bin/pyinstaller --onefile --name fedxgb-standalone \
		--distpath dist-standalone --workpath build/pyinstaller --specpath build \
		--collect-all polars --collect-all pyarrow \
		--hidden-import xgboost --collect-binaries xgboost --collect-data xgboost \
		--collect-submodules fedxgb \
		--exclude-module pytest --noconfirm \
		build/standalone/run_demo.py
	@echo "--- bundled fedxgb modules (expect 8 .so, never a .py) ---"
	@grep -ho "fedxgb/[^'\"]*" build/pyinstaller/fedxgb-standalone/*.toc | sort -u
	@ls -lh dist-standalone/fedxgb-standalone

test:
	$(PY) -m pytest tests -q

## Prove the extensions really work: hide the sources, then run the suite.
verify: ext
	@mkdir -p .srcstash
	@for f in fedxgb/*.py; do \
		[ "$$(basename $$f)" = "__init__.py" ] || mv "$$f" .srcstash/; \
	done
	@$(PY) -c "import fedxgb.weights as w; print('imported from', w.__file__)" || true
	@$(PY) -m pytest tests -q; status=$$?; \
		mv .srcstash/*.py fedxgb/ 2>/dev/null; rmdir .srcstash 2>/dev/null; \
		exit $$status

run:
	$(PY) run_demo.py

clean:
	rm -rf build dist dist-wheel dist-standalone *.egg-info .pytest_cache
	rm -f fedxgb/*.so fedxgb/*.c .venv/bin/fedxgb-demo
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
