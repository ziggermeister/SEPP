# params_ingest.py
# Build a dated, hashed parameter pack (mu/sigma/rho/yields) for your SEPP runs.
# Sources: Tiingo (primary) → Nasdaq Data Link (backup) → yfinance (last resort),
#          FRED for risk-free (optional), issuer fact sheets for SEC-30D yield (best),
#          vendor/yfinance TTM dividend yield fallback.
#
# Usage:
#   pip install pandas numpy scikit-learn yfinance requests python-dateutil
#   export TIINGO_TOKEN=...
#   export NDL_TOKEN=...          # (Nasdaq Data Link, optional)
#   python params_ingest.py --start 2015-01-01 --universe SGOV,VGIT,BND,VWOB,SCHD,CDC,VIG,GLD,VTI,IEFA,VWO,QQQ,CHAT,IBIT,DGIN
#
# Output:
#   params/params_<YYYY-MM-DD>_<stable_hash>.json

import os, json, re, time, hashlib, argparse, datetime as dt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.covariance import LedoitWolf

UNIVERSE_DEFAULT = ["SGOV","VGIT","BND","VWOB","SCHD","CDC","VIG","GLD","VTI",
                    "IEFA","VWO","QQQ","CHAT","IBIT","DGIN"]

ISSUER_HINTS = {
    # ticker -> ("issuer", "factsheet url pattern")
    "VTI":   ("Vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"),
    "VGIT":  ("Vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"),
    "BND":   ("Vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"),
    "VWO":   ("Vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"),
    "IEFA":  ("iShares",  "https://www.ishares.com/us/products/239665/ishares-core-msci-eafe-etf"),
    "QQQ":   ("Invesco",  "https://www.invesco.com/us/financial-products/etfs/product-detail?productId=QQQ"),
    "SGOV":  ("iShares",  "https://www.ishares.com/us/products/309470/ishares-0-3-month-treasury-bond-etf"),
    "SCHD":  ("Schwab",   "https://www.schwabassetmanagement.com/products/schd"),
    "CDC":   ("Victory",  "https://victorysharesliterature.vcm.com/products/victoryshares-us-eq-inc-enhanced-volatility-wtd-etf"),
    "VIG":   ("Vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"),
    "VWOB":  ("Vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"),
    "GLD":   ("SPDR",     "https://www.ssga.com/us/en/individual/etfs/funds/spdr-gold-shares-gld"),
    # others may need vendor fallback
}

NON_INCOME = {"GLD","IBIT","DGIN","CHAT"}  # force yield = 0.0

def stable_hash(obj) -> str:
    b = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:16]

def monthly_log_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    px_m = adj_close.resample("M").last()
    return np.log(px_m).diff().dropna(how="all")

def _tiingo_prices(tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
    token = os.getenv("TIINGO_TOKEN")
    if not token: return None
    # Tiingo supports a /tiingo/daily/<ticker>/prices endpoint (per ticker).
    # We’ll batch via yfinance style for simplicity: do sequential pulls (fast enough for ~15).
    frames = []
    base = "https://api.tiingo.com/tiingo/daily/{t}/prices"
    for t in tickers:
        try:
            r = requests.get(base.format(t=t),
                             params={"startDate": start, "endDate": end, "token": token},
                             timeout=20)
            r.raise_for_status()
            j = r.json()
            if not j: continue
            df = pd.DataFrame(j)
            # Tiingo “adjClose” is in “adjClose” or “adjClose” via “adjClose” key; fallbacks:
            price = None
            for k in ("adjClose","adjClose","adjClose"):  # keep simple
                if k in df.columns:
                    price = df[k]; break
            if price is None and "adjClose" not in df.columns and "adjClose" in df:
                price = df["adjClose"]
            if price is None and "adjClose" not in df.columns:
                # compute from adjHigh/adjLow/adjOpen/adjClose if present
                if "adjClose" in df: price = df["adjClose"]
            if price is None and "close" in df.columns:
                price = df["close"]
            if price is None:
                continue
            df = pd.DataFrame({"Adj Close": price.values}, index=pd.to_datetime(df["date"]))
            df.rename(columns={"Adj Close": t}, inplace=True)
            frames.append(df[[t]])
        except Exception:
            return None
    if not frames:
        return None
    out = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    return out

def _nasdaqdl_prices(tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
    # Placeholder: implement if you use Nasdaq Data Link (Quandl). Return DataFrame like _tiingo_prices.
    return None

def _yfinance_prices(tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        y = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, group_by='ticker')
        if isinstance(y, pd.DataFrame) and "Adj Close" in y.columns:
            # multi-index form
            ac = y["Adj Close"]
        else:
            # unified columns
            if "Adj Close" in y.columns:
                ac = y["Adj Close"].to_frame()
            else:
                return None
        # normalize to wide tickers
        if isinstance(ac.columns, pd.MultiIndex):
            cols = [c[0] for c in ac.columns]
            ac.columns = cols
        ac = ac.reindex(columns=tickers).dropna(how="all").sort_index()
        return ac
    except Exception:
        return None

def fetch_prices(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, str]:
    for fn, label in [(_tiingo_prices,"Tiingo"), (_nasdaqdl_prices,"NasdaqDataLink"), (_yfinance_prices,"yfinance")]:
        df = fn(tickers, start, end)
        if df is not None and df.shape[0] >= 60:  # ≥ 60 daily rows (we resample monthly next)
            return df, label
    raise RuntimeError("Failed to fetch adjusted prices from all providers")

def fetch_ttm_div_yield_vendor(ticker: str) -> Optional[float]:
    # yfinance: Dividends over last 12 months / last price.
    try:
        info = yf.Ticker(ticker)
        hist = info.history(period="1y", actions=True)
        if "Dividends" in hist and hist["Dividends"].sum() > 0:
            last = info.history(period="5d")["Close"].iloc[-1]
            y = float(hist["Dividends"].sum() / last)
            return float(np.clip(y, 0.0, 0.2))
    except Exception:
        return None
    return None

def fetch_issuer_sec30d_yield(ticker: str) -> Optional[float]:
    # Very lightweight HTML scrape with a couple of regex patterns
    # NOTE: issuer pages change; treat as best-effort. Fallback to vendor if parse fails.
    try:
        if ticker not in ISSUER_HINTS: return None
        issuer, url_pat = ISSUER_HINTS[ticker]
        url = url_pat.format(ticker=ticker)
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        html = r.text
        # Try common patterns: "SEC 30-Day Yield: 4.18%" / "30 Day SEC Yield (Subsidized) 4.18%"
        m = re.search(r"(SEC\s*30[-\s]?Day.*?)(\d{1,2}\.\d{1,2})\s*%", html, flags=re.I|re.S)
        if not m:
            m = re.search(r"30[-\s]?Day\s*SEC\s*Yield.*?(\d{1,2}\.\d{1,2})\s*%", html, flags=re.I|re.S)
        if m:
            val = float(m.group(len(m.groups())))
            return val/100.0
    except Exception:
        return None
    return None

def get_income_yield(ticker: str) -> float:
    if ticker in NON_INCOME:
        return 0.0
    y = fetch_issuer_sec30d_yield(ticker)
    if y is not None and 0.0 <= y <= 0.15:
        return y
    y2 = fetch_ttm_div_yield_vendor(ticker)
    if y2 is not None:
        return float(np.clip(y2, 0.0, 0.15))
    return 0.0

def build_pack(tickers: List[str], start: str, end: str) -> Dict:
    px, price_source = fetch_prices(tickers, start, end)
    # monthly total-return proxy: auto_adjusted prices from source include dividends reinvested (yfinance: auto_adjust=True)
    r = monthly_log_returns(px[tickers]).dropna(how="all").fillna(0.0)

    # sanity: enough months
    months = r.shape[0]
    if months < 60:
        raise RuntimeError(f"Not enough monthly data ({months}). Need ≥60.")

    # realized mean / cov with Ledoit–Wolf shrinkage on demeaned monthly returns
    mu_m = r.mean().values
    X = r.values - mu_m
    lw = LedoitWolf().fit(X)
    cov_m = lw.covariance_
    # annualize
    mu_yr  = mu_m * 12.0
    cov_yr = cov_m * 12.0
    sig_yr = np.sqrt(np.diag(cov_yr))

    # correlations from monthly cov
    std_m = np.sqrt(np.diag(lw.covariance_))
    rho = lw.covariance_ / np.outer(std_m, std_m)
    rho = np.clip(rho, -1, 1)

    # yields
    yields = {t: get_income_yield(t) for t in tickers}

    pack = {
        "as_of": dt.date.today().isoformat(),
        "universe": tickers,
        "sources": {
            "prices": price_source,
            "yields": "Issuer SEC-30D (best-effort scrape) → yfinance TTM fallback",
            "rf": "FRED/optional"
        },
        "notes": {
            "returns": "monthly log from adjusted prices; annualized by *12 (geometric mean)",
            "covariance": "Ledoit–Wolf shrinkage on monthly returns; annualized *12",
            "correlations": "derived from monthly covariance",
            "yields": "SEC-30D where available; otherwise TTM dividend yield proxy"
        },
        "mu": dict(zip(tickers, mu_yr.round(10).tolist())),
        "sigma": dict(zip(tickers, sig_yr.round(10).tolist())),
        "rho": pd.DataFrame(rho, index=tickers, columns=tickers).round(10).to_dict(),
        "yield_rate": {k: float(v) for k, v in yields.items()},
        "monthly_obs": int(months),
    }
    pack["stable_hash"] = stable_hash(pack)
    return pack

def qa_pack(pack: Dict, prev_pack: Optional[Dict]=None) -> List[str]:
    issues = []
    tickers = pack["universe"]
    # finite checks
    arr_mu  = np.array([pack["mu"][t] for t in tickers], float)
    arr_sig = np.array([pack["sigma"][t] for t in tickers], float)
    rho = pd.DataFrame(pack["rho"]).loc[tickers, tickers].values
    yld = np.array([pack["yield_rate"][t] for t in tickers], float)

    if not np.isfinite(arr_mu).all():  issues.append("Non-finite mu found")
    if not np.isfinite(arr_sig).all(): issues.append("Non-finite sigma found")
    if not np.isfinite(rho).all():     issues.append("Non-finite rho entries")
    if not np.isfinite(yld).all():     issues.append("Non-finite yields")

    # bounds
    if (arr_sig < 0).any(): issues.append("Negative sigma")
    if not np.allclose(np.diag(rho), 1.0, atol=1e-6): issues.append("rho diag not 1")
    if (np.abs(rho) > 1.0 + 1e-8).any(): issues.append("rho outside [-1,1]")
    if (yld < 0).any() or (yld > 0.15).any(): issues.append("yield out of [0,15%]")

    # symmetry
    if not np.allclose(rho, rho.T, atol=1e-8): issues.append("rho not symmetric")

    # drift check vs previous pack
    if prev_pack:
        pmu  = np.array([prev_pack["mu"][t] for t in tickers], float)
        psig = np.array([prev_pack["sigma"][t] for t in tickers], float)
        prho = pd.DataFrame(prev_pack["rho"]).loc[tickers, tickers].values
        dmu  = np.abs(arr_mu - pmu)
        dsig = np.abs(arr_sig - psig) / np.maximum(psig, 1e-8)
        drho = np.median(np.abs(rho - prho))
        if (dmu > 0.02).any():  issues.append("Δmu > 2% on at least one asset")
        if (dsig > 0.05).any(): issues.append("Δsigma > 5% on at least one asset")
        if drho > 0.15:         issues.append(f"median |Δrho| > 0.15 (={drho:.3f})")
    return issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end",   default=dt.date.today().isoformat())
    ap.add_argument("--universe", default=",".join(UNIVERSE_DEFAULT))
    ap.add_argument("--prev", default=None, help="Path to previous params_*.json for drift QA")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    pack = build_pack(tickers, args.start, args.end)

    prev = None
    if args.prev and os.path.exists(args.prev):
        with open(args.prev) as f: prev = json.load(f)
    issues = qa_pack(pack, prev)
    print("QA:", "OK" if not issues else "WARN -> " + "; ".join(issues))

    os.makedirs("params", exist_ok=True)
    out = f"params/params_{pack['as_of']}_{pack['stable_hash']}.json"
    with open(out, "w") as f:
        json.dump(pack, f, indent=2)
    print("Wrote", out)

if __name__ == "__main__":
    main()
