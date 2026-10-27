"""Nothing private is in this repository.

    python3 -m unittest discover -s tests

This repository is produced by an allow-list exporter. That exporter
has its own tests, which run before
anything is published -- but they run over THERE, and a pull request
arrives over HERE. So this repository checks itself, on every push, with no
knowledge of how it was made.

It walks every tracked file and fails on anything SHAPED like somebody's
house rather than somebody's product: an address off a home network, a path
on somebody's own machine, a credential, a bulk data file, a screenshot, an
internal document, or a retired brand name in text a customer reads.

It deliberately searches for no names: publishing a list of somebody's
private hostnames in order to grep for them is a guard that leaks what it
guards.

WHAT IS DELIBERATELY ALLOWED. `music_brain` and `MUSICBRAIN_*` are the
service's internal vocabulary -- a package name, environment variables,
storage keys -- and so is `jarvis` inside the backend's Python. Both are
implementation detail, visible only to somebody reading the source, and
renaming either is a data migration rather than a privacy fix. They are
allowed in code and forbidden in anything a customer reads: READMEs,
changelogs, add-on and integration metadata, translations, and the panel.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Where the product speaks to a customer. Everything here is read by
#: somebody who never opens a source file.
CUSTOMER_FACING_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".js"}
CUSTOMER_FACING_NAMES = {"README.md", "CHANGELOG.md", "LICENSE", "NOTICE",
                         "config.yaml", "repository.yaml", "manifest.json",
                         "hacs.json", "strings.json", "Dockerfile"}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".github/workflows"}


def _files() -> list[Path]:
    out = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(ROOT).parts
        if any(p in SKIP_DIRS for p in parts) or "__pycache__" in parts:
            continue
        out.append(path)
    return out


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""                                    # a binary asset


#: WHY THERE IS NO LIST OF NAMES HERE.
#:
#: A list of private names to search for -- a domain, an account, a
#: directory, a host -- is, once published, an annotated map of the very
#: environment it exists to protect: a guard that leaks what it guards.
#:
#: Nor would it catch much: no contributor is going to type somebody
#: else's domain into a pull request. What a contributor DOES do is paste
#: a log with their own LAN address in it, commit a token by accident, or
#: drop a screenshot in -- and those are shapes, which is what this file
#: looks for.
#:
#: A leak is a shape. Shapes generalise to people this project has never
#: met; a list of names only ever catches one person.

#: Somebody's filesystem. `/data`, `/app`, `/usr/share/...` are the
#: service's own runtime paths and are fine; a path under a home or a mount
#: point is a developer's machine that got written down.
DEVELOPER_PATH = re.compile(r"/(?:mnt|home|Users|media|srv)/[A-Za-z0-9._-]+/")

#: Patterns that may never appear anywhere.
#:
#: The two LAN ranges are somebody's house. Documentation addresses
#: (192.0.2.0/24, 198.51.100.0/24, RFC 5737) are reserved for exactly this
#: and are used instead wherever a test needs an address.
#:
#: 172.16/12 is deliberately NOT here: 172.30.33.x is the address the
#: Supervisor gives every add-on on every Home Assistant installation in the
#: world. It is a fact about Home Assistant, not about a maintainer.
#: Checked against the NAME, where an end-of-string anchor means what it
#: says. These used to be checked against name + content glued together,
#: where `$` could only ever match a file with no readable text in it -- so
#: a screenshot was caught and a screenshot with a text comment was not.
FORBIDDEN_NAMES = [
    ("a screenshot", re.compile(r"\.(?:png|jpe?g|gif|webp|bmp)$", re.I)),
]

#: Checked against the name AND the contents, since either can carry them.
FORBIDDEN_PATTERNS = [
    ("a private LAN address", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
    ("a private LAN address", re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("a path on somebody's own machine", DEVELOPER_PATH),
]

#: Nothing here is a catalogue, a database dump or an export. The panel is
#: the only large file the product ships, and it is source. A data file this
#: size is something somebody meant to keep locally.
MAX_DATA_FILE_BYTES = 512 * 1024
SOURCE_SUFFIXES = {".py", ".js", ".mjs", ".md", ".yml", ".yaml", ".json",
                   ".sh", ".txt", ""}

#: Things that read as a credential. Every one of these is a shape, not a
#: word: a leak is a shape.
SECRET_PATTERNS = [
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")),
    ("a JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.")),
    ("an AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("an assigned secret", re.compile(
        r"""(?ix)\b(?:api[_-]?key|secret|password|token)\b\s*[:=]\s*
            ["']([A-Za-z0-9+/_\-]{16,})["']""")),
]

#: Values the secret shape above is allowed to match: placeholders, empty
#: defaults and the names of things rather than the things.
SECRET_ALLOWED = re.compile(
    r"(?i)^(?:x{4,}|y{4,}|\.{3,}|<.*>|\$\{.*\}|your[_-].*|change[_-]?me|"
    r"example.*|placeholder.*|redacted.*|secrets\..*|"
    r"(?:smoke|test|fake|dummy|sample|not-a-real|no-such|invalid)[_-].*)$")

#: The retired product name, in text a customer reads.
RETIRED_BRAND = re.compile(r"\bjarvis\b", re.I)


class NothingPrivate(unittest.TestCase):
    def setUp(self):
        self.files = _files()
        self.assertGreater(len(self.files), 10,
                           "this test found almost no files, which means it "
                           "is passing for the wrong reason")

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(ROOT))

    def test_no_private_pattern_anywhere(self):
        """No file is exempt, including this one.

        There used to be an exemption here, because this file carried the
        literals it searched for. It carries none now -- every rule above is
        a shape, and a regex for a LAN address is not a LAN address -- so
        the scanner is scanned like everything else."""
        hits = []
        for path in self.files:
            rel = self._rel(path)
            for label, pattern in FORBIDDEN_NAMES:
                if pattern.search(rel):
                    hits.append(f"{rel}: {label}")
            haystack = rel + "\n" + _text(path)
            for label, pattern in FORBIDDEN_PATTERNS:
                found = pattern.search(haystack)
                if found:
                    hits.append(f"{rel}: {label} ({found.group(0)[:40]!r})")
        self.assertEqual(hits, [], "private content:\n  " + "\n  ".join(hits))

    def test_no_bulk_data_file_is_here(self):
        """A catalogue, a database dump or an export of somebody's library.
        Source can be large -- the panel is 170 KB -- but a DATA file this
        size was meant to stay on one machine."""
        hits = [f"{self._rel(p)} ({p.stat().st_size:,} bytes)"
                for p in self.files
                if p.suffix.lower() not in SOURCE_SUFFIXES
                and p.stat().st_size > MAX_DATA_FILE_BYTES]
        self.assertEqual(hits, [], f"bulk data files: {hits}")

    def test_nothing_looks_like_a_credential(self):
        hits = []
        for path in self.files:
            rel = self._rel(path)
            text = _text(path)
            for label, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(text):
                    value = match.group(1) if match.groups() else match.group(0)
                    if SECRET_ALLOWED.match(value):
                        continue
                    line = text[:match.start()].count("\n") + 1
                    hits.append(f"{rel}:{line}: {label}")
        self.assertEqual(hits, [], "credential-shaped values:\n  " + "\n  ".join(hits))

    def test_no_retired_brand_in_anything_a_customer_reads(self):
        """`jarvis` survives inside the backend's Python as internal
        vocabulary -- queue markers, storage keys, a provider's multipart
        boundary. It may not survive in a sentence somebody reads."""
        hits = []
        for path in self.files:
            rel = self._rel(path)
            customer_facing = (path.name in CUSTOMER_FACING_NAMES
                               or (path.suffix in CUSTOMER_FACING_SUFFIXES
                                   and "backend" not in path.parts))
            if not customer_facing:
                continue
            for n, line in enumerate(_text(path).splitlines(), 1):
                if RETIRED_BRAND.search(line):
                    hits.append(f"{rel}:{n}: {line.strip()[:90]}")
        self.assertEqual(hits, [], "the retired brand, in customer-facing "
                                   "text:\n  " + "\n  ".join(hits))

    #: The retired brand may survive in shipped backend code ONLY as these
    #: exact protocol values. Both are written into the queue-ownership
    #: table and compared against rows already on disk in every existing
    #: installation, so renaming them is a data migration with nothing to
    #: gain. Every OTHER runtime string is a sentence somebody could read.
    RETAINED_PROTOCOL_VALUES = {"jarvis_program_element", "jarvis"}

    def test_the_retired_brand_reaches_no_log_line_or_api_message(self):
        """The customer-facing scan above reads FILES. This one reads
        STRING LITERALS inside the shipped backend, because a docstring is
        private and a log line is not -- and they live in the same file.

        Docstrings and comments are excluded deliberately: they are how the
        work is explained, they ship inside the image, and nobody reads
        them by using the product. A string the code can PRINT is a
        different thing.
        """
        import ast
        hits = []
        for path in self.files:
            if path.suffix != ".py" or "backend" not in path.parts:
                continue
            if "tests" in path.parts:
                continue
            try:
                tree = ast.parse(_text(path))
            except SyntaxError:
                continue
            docs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)) and node.body:
                    first = node.body[0]
                    if (isinstance(first, ast.Expr)
                            and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str)):
                        docs.add(id(first.value))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant)
                        and isinstance(node.value, str)):
                    continue
                if id(node) in docs:
                    continue
                if node.value.strip() in self.RETAINED_PROTOCOL_VALUES:
                    continue
                if RETIRED_BRAND.search(node.value):
                    hits.append(f"{self._rel(path)}:{node.lineno}: "
                                f"{node.value.strip()[:80]}")
        self.assertEqual(hits, [], "the retired brand in a string the "
                                   "shipped service can print:\n  "
                                   + "\n  ".join(hits))

    def test_no_internal_document_ships(self):
        """Design notes, roadmaps, handoffs and audits are how the work was
        done. They are not the product, and several of them describe the
        environment they were written in."""
        allowed = {"README.md", "CHANGELOG.md", "THIRD_PARTY_NOTICES.md"}
        hits = [self._rel(p) for p in self.files
                if p.suffix.lower() in (".md", ".rst")
                and p.name not in allowed]
        self.assertEqual(hits, [], f"documents that are not the product: {hits}")

    def test_the_licence_and_notices_are_here(self):
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            self.assertTrue((ROOT / name).is_file(), f"{name} is missing")
        self.assertIn("Apache License", (ROOT / "LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
