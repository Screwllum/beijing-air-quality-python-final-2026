"""Download official data without credentials. Existing matching data are reused."""
from pathlib import Path
import hashlib
import urllib.request
import urllib.error
import json

ROOT=Path(__file__).resolve().parent
URL='https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'

def download():
    path=ROOT/'data'/'raw'/'beijing-air-quality.zip'
    path.parent.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data'/'SOURCE.json'
    expected=json.loads(manifest.read_text(encoding='utf-8')).get('sha256') if manifest.exists() else None
    if path.exists() and (not expected or hashlib.sha256(path.read_bytes()).hexdigest()==expected):
        print('Verified existing archive:',path);return
    temp=path.with_suffix('.download')
    try:
        urllib.request.urlretrieve(URL,temp)
        digest=hashlib.sha256(temp.read_bytes()).hexdigest()
        if expected and digest!=expected:
            raise ValueError('Downloaded archive changed; inspect its contents before updating SOURCE.json')
        temp.replace(path)
    except (urllib.error.URLError,OSError,ValueError) as exc:
        raise SystemExit(f'Download failed: {exc}\nDownload manually from {URL} and save to {path}') from exc
    print('Saved:',path,'SHA256:',digest)

if __name__=='__main__':
    download()
