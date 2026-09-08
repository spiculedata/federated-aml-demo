# Federated Transaction Monitoring — Polars + XGBoost

A small, runnable demo of **privacy-preserving financial crime detection**.

Three banks each hold a private ledger. Each has been targeted by a *different*
money-laundering typology, so none of them has enough evidence to build a good
model alone. They train together without any of them sharing a single
transaction — only **XGBoost tree weights** cross the wire.

```bash
uv run run_demo.py              # generate data (first run) and train
uv run run_demo.py --explain    # also print the Polars streaming query plan
uv run run_demo.py --weighted   # weight each bank's update by its data volume
uv run pytest -q                # tests
```

## The result

Every model below is given the **same 48-tree budget** and scored on the same
central hold-out set that contains all three typologies.

```
  northwind_bank (solo)              AUC 0.5971
  caledonia_trust (solo)             AUC 0.6635
  meridian_pcb (solo)                AUC 0.7613
  FEDERATED (weights only)           AUC 0.8690
  pooled raw data (not allowed)      AUC 0.8721

  detection rate within a 2% alert budget

  model                        card_not_present   cross_border   structuring   ALL CASES
  ---------------------------------------------------------------------------------------
  northwind_bank (solo)                    0.0%           3.3%         58.3%       20.6%
  caledonia_trust (solo)                  67.5%           0.0%          0.0%       22.5%
  meridian_pcb (solo)                      1.7%          85.0%          0.8%       29.2%
  FEDERATED (weights only)                41.7%          49.2%         37.5%       42.8%
  pooled raw data (not allowed)           32.5%          55.8%         42.5%       43.6%
```

Read the last column. Each bank alone catches roughly one typology and is
blind to the other two. The federated model catches all three and recovers
**~98% of what illegally pooling everyone's raw data would achieve** — having
transmitted 101 KB of tree weights and zero transactions.

## How it works

### 1. Polars streaming, inside each bank

Every participant runs the same feature pipeline over its own ledger as a
`LazyFrame` plan executed with `collect(engine="streaming")`. Nothing is
materialised until collection, and the engine spills to disk rather than RAM,
so a participant with a 50 GB ledger runs the identical code path as this demo.

`fedxgb/features.py` derives velocity signals (transactions per account per
day), behavioural baselines (amount versus the account's own mean) and
threshold-proximity markers — the features an AML analyst would actually reach
for. Run with `--explain` to see the physical plan.

### 2. Local boosting, then extract only the delta

Each round, the central service broadcasts the current global model. A bank
loads it as the starting point, boosts two more trees on its own data, and
returns **only the trees it grew** (`fedxgb/bank_node.py`). The payload is a
`BankUpdate`: tree structures, row counts, and a local metric. There is no
field on it that could carry a transaction.

### 3. Averaging the updates centrally

All banks boost from the same starting point, so each bank's trees encode a
*candidate update* to the global model. The central service scales every leaf
weight by that bank's share and appends all of them (`fedxgb/aggregator.py`).

This is mathematically exact: appending trees whose leaves are scaled by `1/n`
is identical to averaging the participants' updates in margin space. The demo
asserts this to within 1e-5 in `tests/test_weights.py`.

Split thresholds live in the same `split_conditions` array as leaf values, so
`weights.scale_leaf_weights` touches only nodes without children — scaling a
threshold would silently corrupt the tree.

Use `--weighted` for FedAvg-style weighting by data volume instead of one
institution, one vote.

## What actually leaves a bank

| Leaves the bank | Never leaves the bank |
| --- | --- |
| Split thresholds and leaf weights | Transactions |
| Tree structure (children, split feature index) | Account identifiers |
| Row count and positive-case count | Counterparties, amounts, dates |
| A local AUC figure | The feature matrix |

The output at `artifacts/global_model.json` is a standard XGBoost model — load
it with `xgb.Booster(model_file=...)` and score anywhere.

## Recording the demo

```bash
make present     # code walkthrough, then the run (~100 seconds)
make talk        # same, but every screen advances on Enter
make rehearse    # 0.3x pace, for checking changes
```

### The code walkthrough

The recording opens with four slides of real source, so there is no need to cut
away to an editor mid-demo:

```
  THE ONE LINE THAT MUST NOT BE WRONG
  fedxgb/weights.py:49

  49 def scale_leaf_weights(tree: dict[str, Any], factor: float) -> dict[str, Any]:
  50     scaled = copy.deepcopy(tree)
  51     for node, left_child in enumerate(scaled["left_children"]):
  52         if left_child == _LEAF_SENTINEL:
  53             scaled["split_conditions"][node] *= factor
  54             scaled["base_weights"][node] *= factor
  55     return scaled

  Split thresholds share this array with leaf values. Scale one and the tree lies.
```

The four stops are the feature plan, the update payload, the leaf scaling, and
the aggregation — local work, what leaves the building, the dangerous line, and
the merge.

**The snippets are extracted from the source files at runtime by symbol name,
never copied into the slide deck.** Line numbers are the real ones, so you can
point at `weights.py:49` and it will be there. Rename a function and the tour
raises rather than showing code that no longer exists; tests walk every stop and
also assert each slide fits inside 78 columns, so a long line cannot quietly
push the code off the edge of a projector. Docstrings are stripped, because the
speaker is the narration.

Use `--step` (or `make talk`) on the code slides — explaining takes as long as
it takes, and a stopwatch is the wrong tool.

`present.py` runs the identical code path as `run_demo.py` and shows the same
numbers — it only stages them. It computes nothing of its own.

The trade-off comparison holds for 12 seconds by default, and each code slide
for 15, because those are the screens people read rather than glance at. Tune
them per take with `--hold` and `--code-hold`; `--pace` scales every other beat
around them. `--no-tour` skips straight to the demo.

It is built for a projector rather than a desk, so it shows one idea per
screen with large bars and very few numbers visible at once:

```
  ALONE, EACH BANK CATCHES ONE THING

  Northwind Bank    ██████░░░░░░░░░░░░░░░░░░░░░░░░  20.6%
  Caledonia Trust   ███████░░░░░░░░░░░░░░░░░░░░░░░  23.1%
  Meridian PCB      █████████░░░░░░░░░░░░░░░░░░░░░  29.2%

  share of all criminal cases caught, within a 2% alert budget
```

then, after the federation rounds:

```
  TOGETHER

  Northwind Bank    ██████░░░░░░░░░░░░░░░░░░░░░░░░  20.6%
  Caledonia Trust   ███████░░░░░░░░░░░░░░░░░░░░░░░  23.1%
  Meridian PCB      █████████░░░░░░░░░░░░░░░░░░░░░  29.2%

  FEDERATED         █████████████░░░░░░░░░░░░░░░░░  43.6%
```

The final screens answer the question a risk officer asks straight away —
what did federating actually cost us?

```
  WHAT EACH BANK TRADES

                      its own typology   the ones it never saw

  Northwind Bank         57% → 37%              1% → 45%
  Caledonia Trust        69% → 32%              0% → 48%
  Meridian PCB           84% → 59%              2% → 34%

  worse at its speciality. far better at everything else.
```

**Every participant gets worse at the typology it specialises in.** Meridian
alone catches 84% of cross-border; after federating it catches 59%. This is
not a modelling failure — it is a fixed 2% alert budget being spread across
three typologies instead of concentrated on one. Overall detection still
roughly doubles for all three banks, which is the trade the federation is
actually offering. `run_demo.py` prints the same comparison as a table.

Two deliberate choices:

**No learning curve.** The obvious visual is an AUC line climbing across the
rounds, and it would be a lie by omission — that curve is flat here (see the
limitations below). The recording is built around the *contrast* between the
solo models and the federated one, which is where the real result lives.

**The banks train on real threads.** `run_federation(parallel=True)` puts each
participant on its own thread, so the three progressing at once on screen is
true rather than implied. XGBoost releases the GIL while boosting, so this is
genuine overlap: 2.60s sequential versus 1.70s parallel for eight rounds.

## Packaging

The demo is written as ordinary Python, then compiled with Cython so it can be
shipped as build artefacts rather than a source tree. This is a packaging
story, not a performance one — the heavy work already happens inside Polars
and XGBoost native code, and compiling our orchestration layer moves the
end-to-end run only from ~4.84s to ~4.55s.

```bash
make wheel       # distributable wheel: compiled extensions, no Python source
make binary      # small native executable (needs the venv alongside it)
make standalone  # 117 MB single file: bundles CPython, Polars and XGBoost too
make install     # build + install the compiled package into the venv
make verify      # run the test suite with sources hidden, against .so only
make clean       # remove every build artefact
```

Three artefacts, three different distribution stories:

| Artefact | Size | Needs a Python install? | Ships readable source? |
| --- | --- | --- | --- |
| `dist-wheel/*.whl` | 345 KB | yes, plus deps | no |
| `dist/fedxgb-demo` | 170 KB | yes, plus deps | no |
| `dist-standalone/fedxgb-standalone` | 117 MB | no | no |

### The wheel

`make wheel` compiles every module of `fedxgb` into a native extension and
packages them as a normal pip-installable wheel:

```
fedxgb/__init__.py                        53 bytes
fedxgb/aggregator.cpython-312-darwin.so
fedxgb/bank_node.cpython-312-darwin.so
fedxgb/config.cpython-312-darwin.so
fedxgb/data_gen.cpython-312-darwin.so
fedxgb/evaluation.cpython-312-darwin.so
fedxgb/features.cpython-312-darwin.so
fedxgb/server.cpython-312-darwin.so
fedxgb/weights.cpython-312-darwin.so
```

No `.py` sources and no generated `.c`. Both take deliberate effort to
achieve, and both are easy to get wrong:

- setuptools copies `.py` sources into the wheel alongside the extensions by
  default, which would leave the package fully readable. The custom `build_py`
  in `setup.py` keeps only `__init__.py`.
- Cython writes its generated `.c` next to each source, and that C **embeds the
  original Python line by line as comments**. Shipping it would hand the
  sources straight back, so `setup.py` redirects generation to `build/cython/`.

`make verify` is the check that matters: it moves every `.py` out of the
package, confirms imports resolve to `.so`, and runs all 48 tests against the
compiled modules. Dataclasses, `@property` and `vars()` all survive
compilation — which is not a given, hence the test.

### The binary

`make binary` runs `cython --embed` over `run_demo.py` to generate a C entry
point with a real `main()`, then links it against `libpython`:

```
dist/fedxgb-demo: Mach-O 64-bit executable arm64    (170 KB)
```

**Be clear about what this is.** It is a compiled launcher, not a
self-contained application. It still needs a Python runtime and the
`polars`/`xgboost` wheels present. Two things make it work without any
`PYTHONPATH` juggling:

1. It is installed into `.venv/bin/`, so CPython's start-up path logic finds
   `.venv/pyvenv.cfg` beside it and configures the venv's `site-packages`
   automatically — exactly as it does for the real `python` binary.
2. `make install` puts the *compiled* `fedxgb` into that `site-packages`, so
   the binary resolves the package without the source tree being present.

Verified by running it from an empty scratch directory with no environment
variables set and no source tree in sight.

### The standalone bundle

`make standalone` goes further and uses PyInstaller to bundle CPython, Polars,
XGBoost and the compiled `fedxgb` into one 117 MB file that runs on a machine
with no Python at all — verified with `env -i`, an empty environment.

Two traps here, both silent, and the second is the interesting one.

**Source wins over compiled.** PyInstaller adds the entry script's own
directory to the module search path, so building from the project root finds
`fedxgb/*.py` *before* the compiled package in `site-packages` — bundling the
source, having been asked for the compiled build. The Makefile stages
`run_demo.py` into an empty directory first, and prints what it actually
bundled as a check.

**Compiling hides your imports from the bundler.** PyInstaller discovers
dependencies by *parsing Python source*, and it cannot see inside a `.so`. Once
`fedxgb` is compiled, every import made from within it becomes invisible:

- `aggregator` and `weights` are imported only by `server`, so they were
  silently dropped — fixed with `--collect-submodules fedxgb`.
- `xgboost` is imported only by our compiled modules, so nothing analysable
  referenced it. `--collect-binaries`/`--collect-data` copied its files without
  ever adding it to the module graph, and the bundle failed at runtime with
  `module 'xgboost' has no attribute 'DMatrix'`. Fixed with `--hidden-import
  xgboost`, which makes PyInstaller walk xgboost's own sources.
  (`--collect-all xgboost` is the obvious move and does not work: it imports
  `xgboost.testing`, which needs `joblib`.)

Neither failure shows up at build time — the first produces a working binary
that quietly ships your source, the second a binary that dies on first use.
`make standalone` prints the bundled module list precisely so both are visible.

Cython and PyInstaller solve different halves of this: Cython decides what form
your own code ships in, PyInstaller decides whether the runtime comes along.

## Layout

```
fedxgb/config.py       federation topology, hyperparameters, domain constants
fedxgb/data_gen.py     synthetic ledgers; one typology per bank
fedxgb/features.py     Polars streaming feature plan (pure, lazy)
fedxgb/bank_node.py    a participant: local training -> weight payload
fedxgb/weights.py      the wire format: slicing and splicing tree JSON
fedxgb/aggregator.py   the central service: averaging updates into a global model
fedxgb/server.py       round orchestration, hold-out scoring, control models
fedxgb/evaluation.py   alert-budget scoring, per-typology detection rates
run_demo.py            the narrated end-to-end run
setup.py               Cython build: compiled-only wheel, generated C kept out
Makefile               wheel / binary / verify / clean
```

## Honest limitations

This is a demo, not a production federation.

- **Tree structure leaks information.** Split thresholds reveal something about
  a bank's data distribution. A real deployment adds secure aggregation or
  differential privacy on the leaf weights; neither is implemented here.
- **No transport, no identity, no attestation.** Rounds are function calls in
  one process. A real system needs mutual TLS, participant authentication and
  a tamper-evident audit log of every round.
- **The AUC curve is flat across rounds.** Gradient-boosted *ranking* converges
  within a few trees on a problem with 14 rule-like features, so rounds 2–8 add
  little. The rounds are here to exercise the protocol. A harder or
  higher-dimensional problem would show a real learning curve.
- **No dropout, stragglers, or poisoning defence.** Every bank is assumed
  present and honest each round. A malicious participant could skew the global
  model through its leaf weights.
- **The data is synthetic** and the typologies are deliberately cleaner than
  reality.
