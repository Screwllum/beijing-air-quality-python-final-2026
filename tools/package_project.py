"""Validate final artifacts, refresh hashes and create a portable project ZIP."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys
import unittest
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={
    '.gitattributes','.gitignore','MANIFEST.sha256.json','README.md',
    'analysis.ipynb','build_report.py','download_data.py','extended_analysis.py',
    'project_config.json','requirements.txt','requirements-notebook.txt','run_analysis.py',
    'data/SOURCE.json','data/raw/beijing-air-quality.zip',
}
DELIVERY_PATTERNS=(
    ('src','*.py'),('tests','*.py'),('tools','*.py'),
    ('data/processed','*.csv'),('results','*.csv'),
    ('results','*.json'),('results/figures','*.png'),
)

def files():
    """Select deliverables explicitly, excluding temporary files and old ZIPs."""
    selected={ROOT/name for name in ROOT_FILES}
    for directory,pattern in DELIVERY_PATTERNS:
        selected.update((ROOT/directory).glob(pattern))
    cfg=json.loads((ROOT/'project_config.json').read_text(encoding='utf-8'))
    selected.add(ROOT/'report'/f"{cfg['name']}{cfg['student_id']}期末大作业.pdf")
    return sorted(p for p in selected if p.is_file())


def run_unit_tests():
    """Run the current suite and reject failed or empty test discovery."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    suite=unittest.TestLoader().discover(str(ROOT/'tests'))
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={
        'status':'passed' if outcome.wasSuccessful() and outcome.testsRun else 'failed',
        'tests_run':outcome.testsRun,'failures':len(outcome.failures),
        'errors':len(outcome.errors),'skipped':len(outcome.skipped),
        'expected_failures':len(outcome.expectedFailures),
        'unexpected_successes':len(outcome.unexpectedSuccesses),
    }
    if summary['status']!='passed':
        raise RuntimeError(f'Packaging stopped: unit tests did not pass: {summary}')
    return summary

def package(destination=None):
    # Validate before changing files or creating an archive.
    test_summary=run_unit_tests()
    import nbformat
    # Match Git's LF representation so hashes survive a fresh clone on any OS.
    for source in files():
        if source.suffix in {'.py','.json','.csv','.md','.ipynb','.txt'} or source.name in {'.gitattributes','.gitignore'}:
            original=source.read_bytes()
            normalized=original.replace(b'\r\n',b'\n')
            if normalized!=original:
                source.write_bytes(normalized)
    cfg=json.loads((ROOT/'project_config.json').read_text(encoding='utf-8'))
    pdf=ROOT/'report'/f"{cfg['name']}{cfg['student_id']}期末大作业.pdf"
    reader=PdfReader(pdf);texts=[page.extract_text() for page in reader.pages]
    page_count=len(reader.pages)
    figure_count=sum(len(page.images) for page in reader.pages)
    outline_count=len(reader.outline)
    body=''.join(texts)
    assert len(reader.pages)==15, f'Expected 15 pages, found {len(reader.pages)}'
    assert cfg['github_url'] in body
    assert all(f'{i:02d} / 15' in texts[i-1] for i in range(1,16))
    assert all(re.search(f'图{i}'+r'\s',body) for i in range(1,16))
    assert figure_count==15
    assert outline_count==19
    for value in ['46.18','66.27','30.3','78.3','45.0','51.75','3,187']:
        assert value in body,value
    nb=nbformat.read(ROOT/'analysis.ipynb',as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    executed_cells=sum(c.execution_count is not None for c in code)
    error_outputs=sum(o.output_type=='error' for c in code for o in c.outputs)
    assert executed_cells==len(code)
    assert error_outputs==0
    provenance=json.loads((ROOT/'data/SOURCE.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'data/raw/beijing-air-quality.zip').read_bytes()).hexdigest()==provenance['sha256']
    result={'unit_tests':test_summary,'verified_at':datetime.now(timezone.utc).isoformat(),
        'pdf_pages':page_count,'pdf_figures':figure_count,'pdf_outline_sections':outline_count,
        'notebook_executed_code_cells':executed_cells,
        'notebook_error_outputs':error_outputs,'original_archive_sha256_verified':True,
        'github_url':cfg['github_url'],'layout':'continuous',
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'pdf_text_characters':len(body)}
    (ROOT/'results/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files() if p.name!='MANIFEST.sha256.json'}
    (ROOT/'MANIFEST.sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    sources=files()
    dest=Path(destination) if destination else ROOT/'output/python_final_2026'
    dest.mkdir(parents=True,exist_ok=True)
    archive=dest/f"{cfg['name']}{cfg['student_id']}期末大作业_完整项目.zip"
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sources:z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read(f'{ROOT.name}/report/{pdf.name}')==pdf.read_bytes()
        expected={(Path(ROOT.name)/p.relative_to(ROOT)).as_posix() for p in sources}
        assert set(z.namelist())==expected
    print(json.dumps(result,ensure_ascii=True,indent=2))
    print('ZIP:',archive)
    return archive

if __name__=='__main__':package()
