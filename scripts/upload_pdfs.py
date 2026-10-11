"""批量上传PDF到待审核队列"""
import sys
sys.path.insert(0, r"D:\aiproject")

import requests
import os

pdf_dir = r"D:\aiproject\data\kb_pending\source"
url = "http://127.0.0.1:8001/api/v1/knowledge/upload"

for f in sorted(os.listdir(pdf_dir)):
    if not f.endswith('.pdf'):
        continue
    path = os.path.join(pdf_dir, f)
    print(f"上传: {f} ...", end=" ")
    try:
        with open(path, 'rb') as fh:
            files = {'file': (f, fh, 'application/pdf')}
            r = requests.post(url, files=files, timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"OK -> doc_id={data.get('doc_id', data.get('id', '?'))}, status={data.get('status', '?')}")
        else:
            print(f"FAIL HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n上传完成，等待Celery处理...")
