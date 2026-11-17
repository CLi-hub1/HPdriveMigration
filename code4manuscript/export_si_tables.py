"""Extract the final supplementary tables to CSV without modifying the Word file."""
import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from config import PROJECT_ROOT, OUTPUT_ROOT

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'


def cell_text(cell):
    paragraphs = []
    for paragraph in cell.findall(W + 'p'):
        text = []
        for node in paragraph.iter():
            if node.tag in (W + 't', M + 't'):
                text.append(node.text or '')
            elif node.tag in (W + 'br', W + 'cr'):
                text.append('\n')
            elif node.tag == W + 'tab':
                text.append('\t')
        paragraphs.append(''.join(text))
    return '\n'.join(paragraphs).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'manuscript_finalversion/Supplymentary information.docx')
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'si_tables')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Select an empty output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.input) as z:
        body = ET.fromstring(z.read('word/document.xml')).find(W + 'body')
    heading, count = '', 0
    for node in body:
        if node.tag == W + 'p':
            text = ''.join(t.text or '' for t in node.iter(W + 't')).strip()
            if re.match(r'^Table\s+S\d+\b', text, re.I):
                heading = text
        elif node.tag == W + 'tbl':
            count += 1
            match = re.match(r'^Table\s+S(\d+)\b', heading, re.I)
            if not match or int(match.group(1)) != count:
                raise ValueError(f'Unexpected table heading: {heading}')
            rows = []
            for tr in node.findall(W + 'tr'):
                cells = []
                for cell in tr.findall(W + 'tc'):
                    span = cell.find('./' + W + 'tcPr/' + W + 'gridSpan')
                    cells.append(cell_text(cell))
                    cells.extend([''] * (int(span.get(W + 'val')) - 1 if span is not None else 0))
                rows.append(cells)
            width = max(map(len, rows))
            with (args.output / f'Table_S{count:02d}_Word_display_values.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(row + [''] * (width-len(row)) for row in rows)
            heading = ''
    print(f'Extracted {count} tables.')


if __name__ == '__main__':
    main()
