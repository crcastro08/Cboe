import sys, json, requests

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

r = requests.get(URL, headers=HEADERS, timeout=60)
print("HTTP:", r.status_code, "| bytes:", len(r.content))
print("x-deny-reason / server:", r.headers.get("x-deny-reason"), r.headers.get("server"))
if r.status_code != 200:
    print(r.text[:300]); sys.exit(1)

j = r.json()
print("timestamp:", j.get("timestamp"))
d = j["data"]
print("spot:", d.get("current_price"), "| claves:", list(d.keys())[:8])

opts = d["options"]
print("contratos:", len(opts))
print("ejemplo:", json.dumps(opts[0]))

spxw = sum(1 for o in opts if o["option"].startswith("SPXW"))
con_oi = sum(1 for o in opts if (o.get("open_interest") or 0) > 0)
print("SPXW:", spxw, "| SPX:", len(opts) - spxw, "| con OI>0:", con_oi)
