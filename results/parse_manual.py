import zipfile, xml.etree.ElementTree as ET, csv, os, re
from fractions import Fraction

src = r'other\Ruchnye_zamery_zerna_po_ryadam.xlsx'
out = r'results\manual_reference.csv'
z = zipfile.ZipFile(src)
ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
shared = []
sr = ET.fromstring(z.read('xl/sharedStrings.xml'))
for si in sr.findall('m:si', ns):
    shared.append(''.join(t.text or '' for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')))
sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
rows = sheet.findall('.//m:sheetData/m:row', ns)

def cellval(c):
    t = c.attrib.get('t')
    v = c.find('m:v', ns)
    if v is None:
        return ''
    return shared[int(v.text)] if t == 's' else v.text

records = []
current_row = None
position = 0
for r in rows:
    rn = int(r.attrib['r'])
    if rn == 1:
        continue
    vals = {c.attrib['r'][0]: cellval(c) for c in r.findall('m:c', ns)}
    label = vals.get('A', '').strip()
    if label:
        m = re.match(r'(\d+)\s*ряд', label)
        if not m:
            raise ValueError(label)
        current_row = int(m.group(1))
        position = 0
    position += 1
    serial = vals.get('F', '').strip()
    rec = {
        'sheet_row': rn, 'row': current_row, 'pos': position,
        'serial': int(float(serial)) if serial else '',
        'length_mm': vals.get('B', ''), 'width_mm': vals.get('C', ''),
        'thickness_mm': vals.get('D', ''), 'mass_g': vals.get('E', ''),
    }
    records.append(rec)

with open(out, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader(); w.writerows(records)

print('records', len(records))
print('measured', sum(1 for r in records if r['length_mm'] != ''))
print('blank', [r for r in records if r['length_mm'] == ''])
# distribution of row sizes
from collections import Counter
print('row sizes', Counter(r['row'] for r in records))
print('serial min/max', min(r['serial'] for r in records if r['serial'] != ''), max(r['serial'] for r in records if r['serial'] != ''))
print('serial continuous', [r['serial'] for r in records] == list(range(1, 326)))
# check rounding grid
import numpy as np
for col in ('length_mm','width_mm','thickness_mm','mass_g'):
    v = np.array([float(r[col]) for r in records if r[col] != ''])
    for q in (0.1, 0.05, 0.01, 0.0001, 0.00001):
        resid = np.abs(v/q - np.round(v/q))
        print(col, 'q=',q,'max resid* q', (resid*q).max(), 'frac exact', np.mean(resid < 1e-6))
    print(col, 'unique', len(np.unique(v)), 'min', v.min(), 'max', v.max(), 'mean', v.mean())
