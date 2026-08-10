from __future__ import annotations

from pathlib import Path

from ..common import SHA256_RE, sha256_bytes, sha256_file


def test_source_test_manifest_has_complete_standard_and_normalized_self_coverage():
    candidate = Path(__file__).parents[1]
    repository = Path(__file__).parents[4]
    manifest = candidate / "SOURCE_TEST_HASHES.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
    standard: dict[str, str] = {}
    self_lines: list[tuple[str, str]] = []
    for line in lines:
        if line.startswith("# SELF-NORMALIZED-SHA256: "):
            pieces = line.rstrip("\n").split()
            self_lines.append((pieces[2], pieces[3]))
            continue
        digest, relative = line.rstrip("\n").split("  ", 1)
        assert SHA256_RE.fullmatch(digest)
        assert relative not in standard
        standard[relative] = digest
    expected_files = {
        path.relative_to(repository).as_posix()
        for path in candidate.rglob("*")
        if path.is_file() and path != manifest
    }
    assert set(standard) == expected_files
    for relative, expected in standard.items():
        assert sha256_file(repository / relative) == expected
    assert len(self_lines) == 1
    recorded, relative = self_lines[0]
    assert relative == manifest.relative_to(repository).as_posix()
    assert SHA256_RE.fullmatch(recorded)
    assert manifest.read_text(encoding="utf-8").count(recorded) == 1
    normalized = manifest.read_bytes().replace(recorded.encode("ascii"), b"0" * 64)
    assert sha256_bytes(normalized) == recorded
