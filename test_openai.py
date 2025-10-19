import os, requests, json
base = os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1')
url = f'{base}/chat/completions'
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json",
}
body = {
    "model": os.getenv("OPENAI_MODEL","gpt-4o-mini"),
    "messages":[{"role":"user","content":"Say OK."}],
    "max_tokens":5
}
r = requests.post(url, headers=headers, json=body, timeout=20)
print("status:", r.status_code)
print(r.text[:400])