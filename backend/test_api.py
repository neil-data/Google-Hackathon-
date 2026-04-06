import urllib.request
import json

endpoints = ['shipments', 'aviation', 'roads', 'maritime']
result = {}

for ep in endpoints:
    try:
        r = urllib.request.urlopen(f'http://127.0.0.1:8000/api/v1/{ep}/')
        data = json.loads(r.read())
        if data:
            result[ep] = data[0]
        else:
            result[ep] = {"message": "No data available"}
    except Exception as e:
        result[ep] = {"error": str(e)}

print(json.dumps(result, indent=2))
