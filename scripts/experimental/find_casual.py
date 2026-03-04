#!/usr/bin/env python3
"""Find casual language in IEEE paper."""

import re
from pathlib import Path

ieee = Path('research_bundle/07_papers_ieee/IEEE_SMART_NOTES_COMPLETE.md').read_text()

casual = ['amazing', 'awesome', 'cool', 'wow']
for word in casual:
    matches = list(re.finditer(word, ieee, re.IGNORECASE))
    if matches:
        print(f'Found "{word}" ({len(matches)}x):')
        for m in matches:
            start = max(0, m.start()-50)
            end = min(len(ieee), m.end()+50)
            context = ieee[start:end].replace('\n', ' ')
            line_num = ieee[:m.start()].count('\n') + 1
            print(f'  Line {line_num}: ...{context}...')
