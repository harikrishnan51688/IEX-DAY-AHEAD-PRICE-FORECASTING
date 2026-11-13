import json, time, calendar, requests, pandas as pd

BASE = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "RSC": "1",                      # ask Next.js for the flight payload
    "Accept": "text/x-component",
}

def fetch_range(frm, to):
    params = {"interval": "ONE_FOURTH_HOUR", "dp": "SELECT_RANGE",
              "showGraph": "false", "toDate": to, "fromDate": frm,
              "_rsc": "a46q9"}
    r = requests.get(BASE, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    txt = r.text
    key = '"reportData":'
    i = txt.find(key)
    if i == -1:
        raise ValueError("reportData not found (got HTML or blocked?)")
    obj, _ = json.JSONDecoder().raw_decode(txt[i + len(key):])
    return obj["records"]

rows, failed = [], []
for m in range(1, 13):
    last = calendar.monthrange(2025, m)[1]
    frm, to = f"01-{m:02d}-2025", f"{last:02d}-{m:02d}-2025"
    for attempt in range(3):
        try:
            recs = fetch_range(frm, to)
            rows += recs
            print(m, len(recs), "rows")
            break
        except Exception as e:
            print("retry", m, e)
            time.sleep(3 * (attempt + 1))
    else:
        failed.append(m)
    time.sleep(1.5)

df = pd.DataFrame(rows).drop_duplicates(["date", "period"])
for c in ["purchase_bid", "sell_bid", "mcv", "final_scheduled_volume", "mcp"]:
    df[c] = pd.to_numeric(df[c])
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df = df.sort_values(["date", "period"])
df.to_csv("iex_dam_2025_15min.csv", index=False)
print(len(df), "rows; failed months:", failed)