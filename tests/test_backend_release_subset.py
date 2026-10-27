"""The published backend is closed under import, and it is the base product.

    pip install -r parlorwave_radio/backend/requirements.txt
    python3 tests/test_backend_release_subset.py

WHAT THIS IS FOR. The backend published here is a SUBSET of a larger
development tree, selected by taking the import closure of the three
entrypoints the add-on runs. A subset has exactly one interesting failure
mode -- a module that was left behind and is needed anyway -- and it is
invisible until the customer's container starts and dies on an ImportError
inside a request handler an hour later.

So this imports every published module, for real, and then builds the
service's route table. It is not the development tree's full suite: that
belongs where the code is written and runs on every change there. This is
the question only this repository can answer.

It also asserts what is NOT here, because "the first Alpha ships no local
voice engine" is a promise and promises need tests.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "parlorwave_radio" / "backend"

# The image runs with the backend as the working directory and flat on the
# path, which is how its modules import each other.
sys.path.insert(0, str(BACKEND))

# THE ORDER OF THE NEXT TWO LINES IS THE WHOLE POINT.
#
# This file imports all 72 published modules, and CPython writes a
# __pycache__ beside whatever it imports. The tree audits below then look at
# the directory and find caches THIS TEST just made -- a test failing on its
# own footprints, and reporting it as a packaging defect.
#
# So: stop writing them, and photograph the tree before anything is
# imported. The audits run against the photograph, which is the checkout as
# it arrived, not as this process left it. `PYTHONDONTWRITEBYTECODE` in the
# environment does the same job and the image sets it -- but a test that
# only passes because somebody remembered an environment variable is a test
# that fails on the one runner where nobody did, which is exactly how this
# was found.
sys.dont_write_bytecode = True
AS_CHECKED_OUT = sorted(str(p.relative_to(BACKEND)) for p in BACKEND.rglob("*"))

# A writable, disposable state directory, and no Music Assistant. The
# service is designed to start in exactly this condition and report what is
# missing, so nothing here needs a network.
_STATE = tempfile.TemporaryDirectory()
os.environ.setdefault("MUSICBRAIN_BASE_DIR", _STATE.name)
os.environ.setdefault("MUSIC_ASSISTANT_URL", "http://127.0.0.1:8095")
os.environ.setdefault("MUSIC_ASSISTANT_TOKEN", "test-token-not-a-real-one")

#: Optional subsystems that must not be in a base-product release.
FORBIDDEN_MODULES = {
    "kokoro_service",       # the optional voice engine's host process
    "index_library",        # builds a local music catalogue
    "library_compare",      # a migration harness
    "certification_run",    # the model-certification subsystem
    "certification_suite",
}

#: Third-party packages that must not be importable from the image. Most of
#: them would mean a voice engine, a model runtime or weights came along.
#: `mutagen` is there for a different reason: it is GPL-2.0-or-later, and an
#: Apache-2.0 product that imports it cannot describe itself honestly. Its
#: work -- reading a duration, writing three ID3 text frames -- is done by
#: `audio_probe` over the ffprobe boundary instead.
FORBIDDEN_IMPORTS = ("torch", "onnxruntime", "numpy", "transformers",
                     "chatterbox", "kokoro", "mutagen")


def published_modules() -> list[str]:
    top = sorted(p.stem for p in BACKEND.glob("*.py"))
    inf = sorted(f"inference.{p.stem}" for p in (BACKEND / "inference").glob("*.py")
                 if p.stem != "__init__")
    return top + ["inference"] + inf


class TheSubsetIsComplete(unittest.TestCase):
    def test_every_published_module_imports(self):
        failures = []
        for name in published_modules():
            try:
                importlib.import_module(name)
            except Exception as err:               # noqa: BLE001 -- report all
                failures.append(f"{name}: {type(err).__name__}: {err}")
        self.assertEqual(failures, [],
                         "modules that do not import:\n  " + "\n  ".join(failures))

    def test_the_service_builds_its_routes(self):
        app = importlib.import_module("app")
        paths = {r.path for r in app.app.routes}
        for required in ("/", "/health/problems", "/instance", "/dj/status",
                         "/now/playing", "/settings/dj", "/outputs"):
            self.assertIn(required, paths, f"{required} is not routed")
        self.assertGreater(len(paths), 40, len(paths))

    def test_the_entrypoints_are_here(self):
        for name in ("addon_bootstrap", "serve", "app"):
            self.assertTrue((BACKEND / f"{name}.py").is_file(), name)
        self.assertIn("addon_bootstrap", published_modules())

    def test_the_runtime_assets_are_here(self):
        self.assertTrue((BACKEND / "requirements.txt").is_file())
        self.assertTrue((BACKEND / "assets" / "retired_silence.mp3").is_file())


class TheSubsetIsTheBaseProduct(unittest.TestCase):
    def test_no_optional_subsystem_is_published(self):
        present = sorted(m for m in FORBIDDEN_MODULES
                         if (BACKEND / f"{m}.py").exists())
        self.assertEqual(present, [], f"optional subsystems published: {present}")

    def test_no_model_runtime_is_installed(self):
        found = []
        for name in FORBIDDEN_IMPORTS:
            try:
                importlib.import_module(name)
            except ImportError:
                continue
            found.append(name)
        self.assertEqual(found, [],
                         f"the image can import a model runtime: {found}")

    def test_no_weights_or_native_libraries_ship(self):
        bad = [rel for rel in AS_CHECKED_OUT
               if Path(rel).suffix.lower() in (".onnx", ".pt", ".pth", ".bin",
                                               ".so", ".safetensors", ".gguf")]
        self.assertEqual(bad, [], f"binary model or native material: {bad}")

    def test_nothing_but_python_and_the_two_declared_extras(self):
        extra = [rel for rel in AS_CHECKED_OUT
                 if not rel.endswith(".py") and (BACKEND / rel).is_file()
                 and rel not in ("requirements.txt", "assets/retired_silence.mp3")]
        self.assertEqual(extra, [], f"undeclared files in the backend: {extra}")

    def test_no_tests_documents_or_caches_ship(self):
        for rel in AS_CHECKED_OUT:
            self.assertNotIn("tests/", rel, rel)
            self.assertNotIn("__pycache__", rel, rel)
            self.assertFalse(rel.endswith(".md"), rel)
            self.assertFalse(rel.endswith(".sh"), rel)


class TheDependencySetIsPinned(unittest.TestCase):
    def test_every_requirement_is_pinned_to_an_exact_version(self):
        lines = [l.strip() for l in (BACKEND / "requirements.txt").read_text()
                 .splitlines() if l.strip() and not l.startswith("#")]
        self.assertTrue(lines)
        for line in lines:
            self.assertIn("==", line, f"{line} is not pinned")

    def test_no_shipped_module_imports_a_copyleft_library(self):
        """Not importable is one half; not imported is the other. A module
        that reaches for mutagen would fail at runtime rather than at
        build, and only on the path that needs it."""
        import ast
        offenders = []
        for path in sorted(BACKEND.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n.split(".")[0] == "mutagen" for n in names):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], f"GPL import in shipped code: {offenders}")

    def test_the_declared_dependencies_are_importable(self):
        for line in (BACKEND / "requirements.txt").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("==")[0].replace("-", "_")
            importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
