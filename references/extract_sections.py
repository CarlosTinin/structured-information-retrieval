#!/usr/bin/env python3
"""Extract Title, Abstract, Introduction, and Conclusion from pdftotext .txt files."""

import os
import re
from pathlib import Path

REFERENCES_DIR = Path(__file__).parent

# Section heading patterns
ABSTRACT_START = re.compile(r'^\s*(Abstract|ABSTRACT)\s*$', re.MULTILINE)
INTRO_START = re.compile(r'^\s*(1\.?\s+Introduction|I\.\s+Introduction|INTRODUCTION|1\s+Introduction)\s*$', re.MULTILINE | re.IGNORECASE)
NEXT_AFTER_INTRO = re.compile(r'^\s*(2\.?\s+\w|II\.\s+\w|Related\s+Work|Background|Literature|Preliminaries|Problem)', re.MULTILINE | re.IGNORECASE)
CONCLUSION_START = re.compile(r'^\s*(\d+\.?\s+)?Conclusion[s]?\s*(and\s+Future\s+Work)?\s*$', re.MULTILINE | re.IGNORECASE)
AFTER_CONCLUSION = re.compile(r'^\s*(References|REFERENCES|Acknowledg|ACKNOWLEDG|Appendix|APPENDIX)\s*', re.MULTILINE)


def extract_title(text):
    """Extract title: first non-empty lines before Abstract."""
    m = ABSTRACT_START.search(text)
    if m:
        before = text[:m.start()].strip()
        # Take last meaningful chunk (often after header/author info)
        lines = [l.strip() for l in before.split('\n') if l.strip()]
        # Title is usually the first 1-3 lines
        if lines:
            # Heuristic: title is before authors (lines with @, university, etc.)
            title_lines = []
            for line in lines:
                if '@' in line or 'university' in line.lower() or 'department' in line.lower():
                    break
                title_lines.append(line)
            if title_lines:
                return ' '.join(title_lines[:4])  # max 4 lines
            return ' '.join(lines[:3])
    # Fallback: first 3 non-empty lines
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ' '.join(lines[:3]) if lines else "TITLE NOT FOUND"


def extract_between(text, start_pattern, end_pattern, max_chars=15000):
    """Extract text between two pattern matches."""
    m_start = start_pattern.search(text)
    if not m_start:
        return None
    start_pos = m_start.end()
    m_end = end_pattern.search(text, start_pos + 100)  # skip at least 100 chars
    if m_end:
        content = text[start_pos:m_end.start()]
    else:
        content = text[start_pos:start_pos + max_chars]
    return content.strip()[:max_chars]


def extract_abstract(text):
    m = ABSTRACT_START.search(text)
    if not m:
        return "ABSTRACT NOT FOUND"
    start = m.end()
    # End at Introduction or first section heading
    end_pat = re.compile(r'^\s*(1\.?\s+Introduction|I\.\s+Introduction|INTRODUCTION|Keywords|Key\s*words|Index\s+Terms)', re.MULTILINE | re.IGNORECASE)
    m_end = end_pat.search(text, start + 50)
    if m_end:
        return text[start:m_end.start()].strip()[:5000]
    return text[start:start + 3000].strip()


def extract_introduction(text):
    content = extract_between(text, INTRO_START, NEXT_AFTER_INTRO, max_chars=15000)
    return content if content else "INTRODUCTION NOT FOUND"


def extract_conclusion(text):
    m = CONCLUSION_START.search(text)
    if not m:
        # Try alternate patterns
        alt = re.compile(r'^\s*(\d+\.?\s+)?(Concluding\s+Remarks|Final\s+Remarks|Discussion\s+and\s+Conclusion)', re.MULTILINE | re.IGNORECASE)
        m = alt.search(text)
    if not m:
        return "CONCLUSION NOT FOUND"
    start = m.end()
    m_end = AFTER_CONCLUSION.search(text, start + 50)
    if m_end:
        return text[start:m_end.start()].strip()[:10000]
    return text[start:start + 8000].strip()


def process_folder(folder_path):
    folder_name = folder_path.name
    txt_files = sorted(folder_path.glob("*.txt"))
    # Exclude the consolidated output file itself
    txt_files = [f for f in txt_files if f.stem != folder_name and not f.stem.endswith('_summaries')]

    if not txt_files:
        return

    output_path = folder_path / f"{folder_name}.txt"
    with open(output_path, 'w', encoding='utf-8') as out:
        for txt_file in txt_files:
            text = txt_file.read_text(encoding='utf-8', errors='replace')
            if len(text.strip()) < 100:
                continue

            title = extract_title(text)
            abstract = extract_abstract(text)
            introduction = extract_introduction(text)
            conclusion = extract_conclusion(text)

            out.write(f"{'='*80}\n")
            out.write(f"PAPER: {txt_file.stem}\n")
            out.write(f"{'='*80}\n\n")
            out.write(f"--- TITLE ---\n{title}\n\n")
            out.write(f"--- ABSTRACT ---\n{abstract}\n\n")
            out.write(f"--- INTRODUCTION ---\n{introduction}\n\n")
            out.write(f"--- CONCLUSION ---\n{conclusion}\n\n\n")

    print(f"  [{folder_name}] Processed {len(txt_files)} papers -> {output_path.name}")


def main():
    folders = sorted([d for d in REFERENCES_DIR.iterdir() if d.is_dir()])
    print(f"Processing {len(folders)} folders...")
    for folder in folders:
        process_folder(folder)
    print("Done.")


if __name__ == '__main__':
    main()
