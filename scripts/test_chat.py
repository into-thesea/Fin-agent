"""Quick chat test"""
import http.client, json, sys

host = 'localhost:8001'

def test(query, timeout=60):
    conn = http.client.HTTPConnection('localhost', 8001, timeout=timeout)
    body = json.dumps({'query': query}).encode('utf-8')
    t0 = __import__('time').time()
    conn.request('POST', '/api/v1/chat/sync', body=body, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    data = resp.read().decode('utf-8')
    elapsed = __import__('time').time() - t0
    conn.close()
    result = json.loads(data)
    answer = result.get('answer', '')
    print(f'[{elapsed:.1f}s] Status {resp.status}')
    print(f'  Answer: {len(answer)} chars')
    if answer:
        print(f'  Preview: {answer[:200]}')
    else:
        print(f'  Full response: {json.dumps(result, ensure_ascii=False)[:300]}')

if __name__ == '__main__':
    q = sys.argv[1] if len(sys.argv) > 1 else 'hello'
    test(q)
