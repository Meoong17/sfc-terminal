import json, time, datetime, re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

CACHE = "/home/ubuntu/sfc/.etf_cache.json"
URL = "https://farside.co.uk/btc/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
COLUMNS = ["IBIT","FBTC","BITB","ARKB","BTCO","EZBC","BRRR","HODL","BTCW","MSBT","GBTC","BTC"]

def parse_num(s):
    s = s.strip().replace(",", "").replace("$", "").replace("+", "")
    if not s or s in ("-", "–"):
        return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    elif s.startswith("-"):
        neg = True
        s = s[1:]
    try:
        v = float(s)
    except Exception:
        return 0.0
    return -v if neg else v

def parse_date(s):
    # "13 Jul 2026"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s.strip())
    if not m:
        return None
    d, mon, y = m.group(1), m.group(2), m.group(3)
    month = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}[mon[:3]]
    return f"{y}-{month:02d}-{int(d):02d}"

def fetch():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled","--no-sandbox"])
        ctx = b.new_context(user_agent=UA, viewport={"width":1366,"height":900}, locale="en-US")
        page = ctx.new_page()
        page.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        st = time.time()
        while time.time()-st < 40 and "Just a moment" in page.title():
            page.wait_for_timeout(1000)
        if "Just a moment" in page.title():
            raise RuntimeError("Cloudflare challenge not resolved")
        for _ in range(40):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(300)
        page.wait_for_timeout(1500)
        html = page.content()
        b.close()
    return html

def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    t = tables[0]
    flows = []
    cumulative_usd = None
    for tr in t.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th","td"])]
        if len(cells) < 14:
            continue
        date_s = cells[0]
        date = parse_date(date_s)
        if date:
            vals = [parse_num(c) for c in cells[1:13]]
            total = parse_num(cells[13])
            etfs = dict(zip(COLUMNS, vals))
            flows.append({"date": date, "total_btc": None,
                          "total_usd": round(total * 1_000_000), "etfs": etfs})
        elif date_s.strip().lower() == "total":
            # cumulative net inflow in USD millions -> last cell
            cumulative_usd = round(parse_num(cells[13]) * 1_000_000)
    return flows, cumulative_usd

def main():
    html = fetch()
    flows, cumulative_usd = parse(html)
    print(f"parsed {len(flows)} flow rows; cumulative_usd=${cumulative_usd}")

    with open(CACHE) as f:
        cache = json.load(f)
    old = {x["date"]: x for x in cache["flows"]}
    merged = {**old, **{x["date"]: x for x in flows}}
    cache["flows"] = [merged[d] for d in sorted(merged.keys())]

    if cumulative_usd is not None:
        cache["cumulative_usd"] = cumulative_usd
    cache["cumulative_btc"] = None
    cache["last_update"] = datetime.datetime.now().isoformat(timespec="seconds")
    cache["cached_at"] = time.time()

    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=2)

    print("updated flows:", len(cache["flows"]))
    for d in ["2026-07-30","2026-07-31"]:
        print(d, cache["flows"][next(i for i,x in enumerate(cache["flows"]) if x["date"]==d)])

if __name__ == "__main__":
    main()
