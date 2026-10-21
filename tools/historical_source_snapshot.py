"""Verify immutable private producers across a public identifier redaction."""
import ast
import json
from pathlib import Path
import re

from ar2_checkpoint import digest
from worker_identity import private_config_path

ROOT = Path(__file__).resolve().parents[1]


class HistoricalSources:
    def __init__(self):
        config = json.loads(private_config_path().read_text())
        self.snapshot = Path(config['source_snapshot'])
        self.host = config['expected_hostname'].split('.')[0]
        self.entries = json.loads((self.snapshot / 'manifest.json').read_text())
        self.receipts = []

    def neutral(self, text):
        return re.sub(re.escape(self.host), lambda m: 'WORKER' if m[0].isupper()
                      else 'Worker' if m[0][0].isupper() else 'worker', text, flags=re.I)

    def original(self, path, expected):
        """Check both original bytes and exactly allowed current text changes."""
        relative = Path(path).relative_to(ROOT)
        neutral = self.neutral(str(relative))
        entries = [e for e in self.entries if self.neutral(e['path']) == neutral
                   and e['sha256'] == expected]
        if len(entries) != 1:
            raise RuntimeError('No unique immutable source binding for the historical producer')
        original = self.snapshot / 'worktree' / entries[0]['path']
        current = ROOT / neutral
        if digest(original) != expected:
            raise RuntimeError('Private original source snapshot changed')
        old = self.neutral(original.read_text())
        new = current.read_text()
        if current.suffix == '.py':
            equal = ast.dump(ast.parse(old), include_attributes=False) == ast.dump(ast.parse(new), include_attributes=False)
        elif current.suffix == '.json':
            equal = json.loads(old) == json.loads(new)
        else:
            equal = old == new
        if not equal:
            raise RuntimeError('Current producer differs beyond the approved identifier redaction')
        self.receipts.append({'public_path': str(current.relative_to(ROOT)), 'original_sha256': expected,
                              'current_sha256': digest(current), 'verification': 'EXACT_AST_OR_JSON_AFTER_IDENTIFIER_REDACTION'})
        return original

    def verify(self, path, expected):
        file = Path(path)
        if file.is_file() and digest(file) == expected:
            return
        self.original(path, expected)
