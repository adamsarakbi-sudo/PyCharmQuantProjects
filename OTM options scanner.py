import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time
from functools import lru_cache

# ================= CONFIG ================= #
SLEEP_BETWEEN_TICKERS = 0.25
RISK_FREE_RATE = 0.045

MIN_DTE = 20
MAX_DTE = 60

MIN_VOLUME = 50
MIN_OI = 200
MAX_SPREAD_PCT = 0.25
MAX_DOLLAR_SPREAD = 0.15

MIN_WIDTH = 2.0
MIN_PROB_ITM = 0.30
MIN_TREND_STRENGTH = 0.02
MAX_IV_RANK = 0.45

SMA_FAST = 20
SMA_SLOW = 50

MAX_PORTFOLIO_VEGA = 5.0
MAX_PORTFOLIO_GAMMA = 0.5
MAX_PORTFOLIO_DELTA = 2.0

# ================= TICKERS ================= #
sp500_tickers = sp500_tickers = [
"MMM","ABT","ABBV","ACN","ATVI","ADM","ADBE","AAP","AES","AFL",
"A","APD","AKAM","ALK","ALB","ARE","ALXN","ALGN","ALLE","LNT",
"ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP",
"AIG","AMT","AWK","AMP","ABC","AME","AMGN","APH","ADI","ANSS",
"ANTM","AON","APA","AAPL","AMAT","APTV","ADM","AJG","AIZ","T",
"ATO","ADSK","AZO","AVB","AVY","BKR","BLL","BAC","BAX","BDX",
"BRK.B","BBY","BIO","TECH","BIIB","BLK","BK","BWA","BXP","BSX",
"BHGE","BR","BF.B","CHRW","COG","CDNS","CZR","CPB","COF","CAH",
"CBOE","KMX","CCL","CAT","CBOE","CBRE","CDW","CE","CNC","CNP",
"CTL","CF","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS",
"CSCO","C","CFG","CTXS","CLX","CME","CMS","KO","CTSH","CL","CMCSA",
"CMA","CAG","CXO","COP","ED","STZ","COO","CPRT","GLW","CTVA","COST",
"CCI","CSX","CME","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN",
"DXC","FANG","F","FAST","FRT","FDX","FITB","FRC","FE","FIS","FISV",
"FLT","FLIR","FLS","FLR","FTNT","FTV","FBHS","FOXA","FOX","BEN",
"FCX","GPS","GRMN","IT","GD","GE","GILD","GL","HOLX","HBI","HOG",
"HIG","HAS","HCA","PEAK","HP","HSY","HES","HLT","HFC","HOLX","HD",
"HON","HRL","HST","HPQ","HUM","HBAN","IEX","IDXX","INFO","ITW",
"ILMN","INCY","IR","INTC","ICE","IFF","IP","IPG","IPGP","IQV","IRM",
"JKHY","JNJ","JCI","JPM","JNPR","KSU","K","KEY","KEYS","KMB","KIM",
"KMI","KMX","KO","CTAS","KR","LRCX","LAMR","LEG","LLY","LNC","LIN",
"LMT","L","LOW","LKQ","LH","LYB","MTB","MRO","MPC","MKTX","MAR","MMC",
"MLM","MAS","MA","MKC","MXIM","MCD","MCK","MDT","MRK","MET","MGM",
"MCHP","MU","MSFT","MS","MAA","MHK","TAP","MCO","MSCI","MDLZ","MON",
"NWL","NOC","NCLH","NTRS","NOC","NLOK","NDAQ","NOV","NTAP","NFLX",
"NMC","NEE","NLSN","NKE","NI","NSC","NTR","NOC","NUE","NVDA","ORLY",
"OXY","OMC","OKE","ORCL","OTIS","PCAR","PKG","PH","PAYX","PAYC","PYPL",
"PNR","PBCT","PEP","PKI","PRGO","PFE","PM","PSX","PNW","PXD","PNC",
"PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PSA","PHM","QRVO",
"PWR","QCOM","DGX","RL","RJF","RTN","O","REG","REGN","RF","RSG",
"RMD","RHI","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB",
"STX","SEE","SRE","NOW","SHW","SPG","SWKS","SLG","SBNY","SEDG","STT",
"STE","SYK","SIVB","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR",
"TRV","TMO","TIF","TWTR","TJX","TSCO","TSN","UDR","UHS","ULTA","USB",
"UAA","UNP","UAL","UNH","UPS","URI","UHS","VLO","VTR","VZ","VRSK","VRSN",
"VCTR","VFC","VTRS","V","VNO","VMC","WRB","WAB","WMT","WBA","DIS",
"WM","WAT","WY","WHR","WMB","WEC","WELL","WST","WSM","WTW","XEL","XOM",
"XLNX","XYL","YUM","ZBH","ZION","ZTS"
]

# ================= CACHING ================= #
@lru_cache(maxsize=512)
def get_history(ticker, period="6mo"):
    return yf.Ticker(ticker).history(period=period)

@lru_cache(maxsize=256)
def get_chain(ticker, exp):
    return yf.Ticker(ticker).option_chain(exp)

# ================= UTILS ================= #
def forecast_vol(close):
    returns = np.log(close / close.shift(1))
    return returns.ewm(span=30).std().iloc[-1] * np.sqrt(252)

def iv_rank_from_realized(close, current_iv):
    returns = np.log(close / close.shift(1))
    hist_iv = returns.rolling(20).std() * np.sqrt(252)
    iv_min, iv_max = hist_iv.min(), hist_iv.max()
    if iv_max == iv_min:
        return np.nan
    return (current_iv - iv_min) / (iv_max - iv_min)

def vol_acceleration(close):
    returns = close.pct_change()
    return (returns.rolling(5).std() - returns.rolling(20).std()).iloc[-1]

# ================= BLACK-SCHOLES ================= #
def bs_metrics(S, K, T, r, sigma, opt_type):
    if sigma <= 0 or T <= 0:
        return [np.nan]*7

    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf = norm.pdf(d1)

    if opt_type == "call":
        delta = norm.cdf(d1)
        prob_itm = norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        prob_itm = norm.cdf(-d2)

    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T) / 100
    theta = -(S * pdf * sigma) / (2 * np.sqrt(T)) / 365

    return delta, gamma, vega, theta, prob_itm

# ================= TREND FILTER ================= #
def trend_regime(close):
    sma_fast = close.rolling(SMA_FAST).mean()
    sma_slow = close.rolling(SMA_SLOW).mean()
    spot = close.iloc[-1]

    bullish = (
        sma_fast.iloc[-1] > sma_slow.iloc[-1] and
        sma_fast.diff().iloc[-3:].mean() > 0 and
        spot > sma_fast.iloc[-1]
    )
    bearish = (
        sma_fast.iloc[-1] < sma_slow.iloc[-1] and
        sma_fast.diff().iloc[-3:].mean() < 0 and
        spot < sma_fast.iloc[-1]
    )

    strength = abs(spot - sma_fast.iloc[-1]) / sma_fast.iloc[-1]
    return bullish, bearish, strength

# ================= SCANNER ================= #
results = []
seen_tickers = set()

portfolio_delta = 0
portfolio_vega = 0
portfolio_gamma = 0

today = datetime.now().date()

for ticker in sp500_tickers:
    try:
        if ticker in seen_tickers:
            continue

        hist = get_history(ticker)
        if hist.empty or len(hist) < 60:
            continue

        close = hist["Close"]
        spot = close.iloc[-1]

        bullish, bearish, trend_strength = trend_regime(close)
        if not (bullish or bearish):
            continue
        if trend_strength < MIN_TREND_STRENGTH:
            continue

        if vol_acceleration(close) <= 0:
            continue

        stock = yf.Ticker(ticker)

        # ===== Earnings Filter =====
        cal = stock.calendar
        earnings_date = None

        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                earnings_date = cal.loc["Earnings Date"][0].date()
        elif isinstance(cal, dict):
            if "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if isinstance(ed, (list, tuple)):
                    earnings_date = pd.to_datetime(ed[0]).date()
                else:
                    earnings_date = pd.to_datetime(ed).date()

        if earnings_date and abs((earnings_date - today).days) <= 7:
            continue

        expirations = stock.options
        if not expirations:
            continue

        valid_exps = [
            e for e in expirations
            if MIN_DTE <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= MAX_DTE
        ]

        best_trade = None

        for exp in valid_exps[:2]:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            T = dte / 365

            chain = get_chain(ticker, exp)
            opt_df = chain.calls if bullish else chain.puts
            opt_type = "call" if bullish else "put"

            opt_df = opt_df.copy()
            opt_df["mid"] = (opt_df["bid"] + opt_df["ask"]) / 2
            opt_df["spread_pct"] = (opt_df["ask"] - opt_df["bid"]) / opt_df["mid"]
            opt_df["dollar_spread"] = opt_df["ask"] - opt_df["bid"]

            opt_df = opt_df.dropna(subset=["mid", "impliedVolatility"])
            opt_df = opt_df[
                ((opt_df["volume"] >= MIN_VOLUME) | (opt_df["openInterest"] >= MIN_OI)) &
                (opt_df["spread_pct"] <= MAX_SPREAD_PCT) &
                (opt_df["dollar_spread"] <= MAX_DOLLAR_SPREAD)
            ]

            if len(opt_df) < 2:
                continue

            # ===== Delta Calculation =====
            deltas = []
            greeks = []

            for _, row in opt_df.iterrows():
                d, g, v, t, p = bs_metrics(
                    spot, row["strike"], T, RISK_FREE_RATE,
                    row["impliedVolatility"], opt_type
                )
                deltas.append(d)
                greeks.append((g, v, t, p))

            opt_df["delta"] = deltas
            opt_df["gamma"], opt_df["vega"], opt_df["theta"], opt_df["prob_itm"] = zip(*greeks)

            delta_buy_target = 0.30 if dte < 30 else 0.20
            delta_sell_target = 0.15 if dte < 30 else 0.08
            if opt_type == "put":
                delta_buy_target *= -1
                delta_sell_target *= -1

            buy = opt_df.iloc[(opt_df["delta"] - delta_buy_target).abs().argsort()[:1]]
            sell = opt_df.iloc[(opt_df["delta"] - delta_sell_target).abs().argsort()[:1]]

            K_buy = buy["strike"].values[0]
            K_sell = sell["strike"].values[0]

            # ===== Structural Filters =====
            if opt_type == "call" and K_buy >= K_sell:
                continue
            if opt_type == "put" and K_buy <= K_sell:
                continue

            width = abs(K_sell - K_buy)
            if width < MIN_WIDTH:
                continue

            spread_cost = buy["mid"].values[0] - sell["mid"].values[0]
            max_profit = width - spread_cost
            if max_profit <= 0:
                continue

            # ===== Probability & EM Filters =====
            if buy["prob_itm"].values[0] < MIN_PROB_ITM:
                continue

            expected_move = spot * buy["impliedVolatility"].values[0] * np.sqrt(T)
            if abs(K_buy - spot) > 0.7 * expected_move:
                continue

            iv_rank = iv_rank_from_realized(close, buy["impliedVolatility"].values[0])
            if iv_rank is not None and iv_rank > MAX_IV_RANK:
                continue

            hedge_delta = buy["delta"].values[0] - sell["delta"].values[0]
            gamma_total = buy["gamma"].values[0] - sell["gamma"].values[0]
            vega_total = buy["vega"].values[0] - sell["vega"].values[0]

            if abs(portfolio_delta + hedge_delta) > MAX_PORTFOLIO_DELTA:
                continue
            if portfolio_vega + vega_total > MAX_PORTFOLIO_VEGA:
                continue
            if portfolio_gamma + gamma_total > MAX_PORTFOLIO_GAMMA:
                continue

            rr = max_profit / spread_cost

            trade = {
                "Ticker": ticker,
                "Type": f"{opt_type.upper()} DEBIT",
                "BuyStrike": K_buy,
                "SellStrike": K_sell,
                "Expiry": exp,
                "DTE": dte,
                "Spot": round(spot, 2),
                "SpreadCost": round(spread_cost, 2),
                "MaxProfit": round(max_profit, 2),
                "RR": round(rr, 2),
                "ProbITM": round(buy["prob_itm"].values[0], 2),
                "Delta": round(hedge_delta, 3),
                "Gamma": round(gamma_total, 3),
                "Vega": round(vega_total, 3),
                "Theta": round(buy["theta"].values[0] - sell["theta"].values[0], 3),
                "TrendStrength": round(trend_strength, 3)
            }

            if best_trade is None or rr > best_trade["RR"]:
                best_trade = trade

        if best_trade:
            results.append(best_trade)
            seen_tickers.add(ticker)

        time.sleep(SLEEP_BETWEEN_TICKERS)

    except Exception as e:
        print(f"{ticker} skipped: {e}")

# ================= OUTPUT ================= #
df = pd.DataFrame(results)

if df.empty:
    print("No high-quality debit spreads found today.")
else:
    df = df.sort_values("RR", ascending=False)
    df.to_csv("high_quality_debit_spreads.csv", index=False)
    print(df.to_string(index=False))
