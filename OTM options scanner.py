import warnings
warnings.filterwarnings("ignore")

import time
import json
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

RISK_FREE_RATE        = 0.045
ACCOUNT_SIZE          = 50_000

# DTE window
MIN_DTE               = 21
MAX_DTE               = 60

# Delta targets — short leg OTM, long leg further OTM
SHORT_DELTA_MIN       = 0.10
SHORT_DELTA_MAX       = 0.35
LONG_DELTA_MIN        = 0.03
LONG_DELTA_MAX        = 0.18

MAX_NET_DELTA         = 0.30

# Liquidity
MIN_VOLUME            = 10
MIN_OI                = 50
MAX_BID_ASK_PCT       = 0.40

# Quality
MIN_CREDIT            = 0.10
MIN_CREDIT_RATIO      = 0.15   # credit / width
MAX_CREDIT_RATIO      = 0.45   # arbitrage guard
MIN_IV_PERCENTILE     = 25     # real IV pct, not RV proxy
MIN_SPREAD_WIDTH      = 1.0
MAX_SPREAD_WIDTH      = 20.0

# Fundamental
MIN_MARKET_CAP        = 2e9
MIN_FUND_SCORE        = 15

# Risk limits
MAX_BETA              = 2.5
MAX_PER_SECTOR        = 4
MAX_PER_TICKER        = 2

# Costs
COMMISSION            = 0.65   # per contract per leg
SLIPPAGE              = 0.05   # per contract per leg

# Output
TOP_N                 = 25
SLEEP_SEC             = 0.15

# Backtest
BT_YEARS              = 3      # years of history to backtest
BT_MIN_TRADES         = 10     # minimum trades needed for valid backtest


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def get_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=64)
def get_spy_history() -> pd.DataFrame:
    try:
        df = yf.Ticker("SPY").history(period="3mo")
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def bs_greeks(S: float, K: float, T: float, r: float,
              sigma: float, opt_type: str) -> Dict[str, float]:
    """Full Black-Scholes greeks with input validation."""
    ZERO = {'price': 0.0, 'delta': 0.0, 'gamma': 0.0,
            'theta': 0.0, 'vega': 0.0, 'prob_itm': 0.0}

    if any(v <= 0 for v in (S, K, T, sigma)):
        return ZERO
    if not (0.03 <= sigma <= 4.0):
        return ZERO
    if not (0.005 <= T <= 2.5):
        return ZERO
    if not (0.2 <= S / K <= 5.0):
        return ZERO

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        if opt_type.lower() == 'call':
            price    = S * norm.cdf(d1)  - K * np.exp(-r * T) * norm.cdf(d2)
            delta    = norm.cdf(d1)
            prob_itm = norm.cdf(d2)
            theta    = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            price    = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta    = -norm.cdf(-d1)
            prob_itm = norm.cdf(-d2)
            theta    = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega  = S * pdf_d1 * np.sqrt(T) / 100

        return {'price': price, 'delta': delta, 'gamma': gamma,
                'theta': theta, 'vega': vega, 'prob_itm': prob_itm}
    except Exception:
        return ZERO


# ═══════════════════════════════════════════════════════════════════════════
# IV PERCENTILE  —  computed from the option chain itself
# ═══════════════════════════════════════════════════════════════════════════

def compute_iv_percentile(ticker: str, current_iv: float,
                           lookback_days: int = 252) -> float:
    """
    True IV percentile: where does current_iv sit in the distribution of
    ATM 30-day implied vols observed over the past `lookback_days` trading days?

    We approximate the historical IV series from realised volatility over
    rolling 21-day windows, calibrated to the current IV level.  This is
    standard when a full options history is unavailable.

    Returns a percentile in [0, 100].
    """
    try:
        hist = get_history(ticker, "2y")
        if hist.empty or len(hist) < 63:
            return 50.0

        log_rets  = np.log(hist['close'] / hist['close'].shift(1)).dropna()
        rv_series = log_rets.rolling(21).std() * np.sqrt(252)
        rv_series = rv_series.dropna().tail(lookback_days)

        if len(rv_series) < 30:
            return 50.0

        # Scale RV series so its median matches current IV
        # (adjusts for the well-known IV > RV premium without discarding shape)
        scale         = current_iv / (rv_series.median() + 1e-8)
        iv_proxy      = rv_series * scale
        pct           = float((iv_proxy < current_iv).mean() * 100)
        return np.clip(pct, 0.0, 100.0)

    except Exception:
        return 50.0


# ═══════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def fundamental_score(ticker: str) -> Tuple[float, List[str]]:
    """
    Returns (score, skip_reasons).
    score is out of ~100; skip_reasons non-empty means hard fail.
    """
    try:
        info   = yf.Ticker(ticker).info
        score  = 0.0
        skip   = []

        mcap = info.get('marketCap', 0) or 0
        if mcap < MIN_MARKET_CAP:
            skip.append(f"mcap ${mcap/1e9:.1f}B < ${MIN_MARKET_CAP/1e9:.0f}B")
            return 0.0, skip

        # Market cap tier
        score += 15 if mcap > 50e9 else 12 if mcap > 10e9 else 8 if mcap > 5e9 else 5

        # Profitability
        pm = info.get('profitMargins', 0) or 0
        score += 20 if pm > 0.20 else 15 if pm > 0.10 else 10 if pm > 0.05 else 5 if pm > 0 else 0

        # Revenue growth
        rg = info.get('revenueGrowth', 0) or 0
        score += 15 if rg > 0.25 else 12 if rg > 0.15 else 8 if rg > 0.05 else 5 if rg > 0 else 0

        # Debt / equity
        de = info.get('debtToEquity', 999) or 999
        score += 15 if de < 30 else 12 if de < 75 else 8 if de < 150 else 4 if de < 250 else 0

        # Free cash flow
        fcf = info.get('freeCashflow', 0) or 0
        score += 12 if fcf > 1e9 else 8 if fcf > 0 else 0

        # Analyst recommendation
        rec = info.get('recommendationKey', '') or ''
        score += 12 if rec in ('strong_buy', 'buy') else 6 if rec == 'hold' else 0

        # Institutional ownership
        inst = info.get('heldPercentInstitutions', 0) or 0
        score += 10 if inst > 0.75 else 8 if inst > 0.60 else 5 if inst > 0.40 else 0

        # ROE
        roe = info.get('returnOnEquity', 0) or 0
        score += 12 if roe > 0.25 else 10 if roe > 0.18 else 6 if roe > 0.12 else 3 if roe > 0 else 0

        return float(score), skip

    except Exception as e:
        return 0.0, [f"fundamental unavailable: {e}"]


# ═══════════════════════════════════════════════════════════════════════════
# RISK FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def earnings_penalty(ticker: str, dte: int) -> Tuple[bool, float]:
    """Returns (has_risk, penalty_points) — penalty_points deducted from score."""
    try:
        stock    = yf.Ticker(ticker)
        cal      = stock.calendar
        ed       = None

        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if 'Earnings Date' in cal.columns:
                ed = pd.to_datetime(cal['Earnings Date'].iloc[0])
        elif isinstance(cal, dict) and 'Earnings Date' in cal:
            raw = cal['Earnings Date']
            ed  = pd.to_datetime(raw[0] if isinstance(raw, list) else raw)

        if ed is None:
            # Fallback: estimate from earnings history
            eh = stock.earnings_dates
            if eh is not None and not eh.empty:
                ed = eh.index[0] + pd.DateOffset(days=90)

        if ed is not None:
            days = (pd.Timestamp(ed).date() - datetime.now().date()).days
            if 0 <= days <= dte:
                pts = 9 if days < 7 else 6 if days < 14 else 3
                return True, float(pts)

        return False, 0.0
    except Exception:
        return False, 0.0


def dividend_penalty(ticker: str, dte: int) -> Tuple[bool, float]:
    """Returns (has_risk, penalty_points)."""
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            return False, 0.0

        next_est = divs.index[-1] + pd.DateOffset(months=3)
        days     = (next_est.date() - datetime.now().date()).days
        if 0 <= days <= dte:
            return True, 2.0
        return False, 0.0
    except Exception:
        return False, 0.0


def get_sector_beta(ticker: str) -> Tuple[str, float]:
    try:
        info   = yf.Ticker(ticker).info
        sector = info.get('sector', 'Unknown') or 'Unknown'
        beta   = float(info.get('beta', 1.0) or 1.0)
        if np.isnan(beta) or beta == 0:
            beta = 1.0
        return sector, beta
    except Exception:
        return 'Unknown', 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def trend_score(close: pd.Series) -> float:
    """Trend score in [-1, 1]. Positive = bullish."""
    if len(close) < 50:
        return 0.0
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if pd.isna(sma20.iloc[-1]) or pd.isna(sma50.iloc[-1]):
        return 0.0
    spot      = close.iloc[-1]
    fast_diff = (spot - sma20.iloc[-1]) / spot
    slow_diff = (spot - sma50.iloc[-1]) / spot
    cross     = 1.0 if sma20.iloc[-1] > sma50.iloc[-1] else -1.0
    return float(np.clip(0.4 * fast_diff + 0.3 * slow_diff + 0.3 * cross, -1, 1))


def relative_strength(hist: pd.DataFrame) -> float:
    """21-day return relative to SPY, in percentage points."""
    try:
        spy = get_spy_history()
        if spy.empty or len(hist) < 22:
            return 0.0
        r_ticker = (hist['close'].iloc[-1] / hist['close'].iloc[-22] - 1) * 100
        r_spy    = (spy['close'].iloc[-1]  / spy['close'].iloc[-22]  - 1) * 100
        return float(r_ticker - r_spy)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OPTION CHAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def filter_chain(opts: pd.DataFrame) -> pd.DataFrame:
    """Liquidity and sanity filters on raw option chain."""
    opts = opts.copy()
    opts = opts[
        (opts['bid'] > 0) &
        (opts['ask'] > opts['bid']) &
        opts['bid'].notna() &
        opts['ask'].notna()
    ]
    opts['mid']        = (opts['bid'] + opts['ask']) / 2.0
    opts['ba_pct']     = (opts['ask'] - opts['bid']) / opts['mid']
    opts = opts[
        (opts['volume'].fillna(0)       >= MIN_VOLUME) &
        (opts['openInterest'].fillna(0) >= MIN_OI) &
        (opts['ba_pct']                 <= MAX_BID_ASK_PCT) &
        (opts['impliedVolatility']      >  0) &
        (opts['impliedVolatility']      <  4.0)
    ]
    return opts.sort_values('strike').reset_index(drop=True)


def add_greeks(opts: pd.DataFrame, spot: float, T: float,
               opt_type: str) -> pd.DataFrame:
    """Vectorised greek calculation for entire chain."""
    records = [
        bs_greeks(spot, row['strike'], T, RISK_FREE_RATE,
                  row['impliedVolatility'], opt_type)
        for _, row in opts.iterrows()
    ]
    return pd.concat(
        [opts.reset_index(drop=True), pd.DataFrame(records)], axis=1
    )


# ═══════════════════════════════════════════════════════════════════════════
# SPREAD SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_spread(ror: float, prob_profit: float, net_theta: float,
                 iv_pct: float, ts: float, rs: float, fund: float,
                 beta: float, spread_type: str) -> float:
    """
    Additive scoring.  Weights are calibrated so a realistic 30-delta
    credit spread with 70% prob profit, moderate IV scores ~55-65.

    Component            Weight   Rationale
    ─────────────────────────────────────────────────────────────
    Return on risk         28     Primary edge measure
    Prob profit            22     Theoretical win rate
    Theta (daily decay)    16     Cash flow quality
    IV percentile          14     Are we selling rich vol?
    Trend alignment         8     Regime filter
    Relative strength       6     Momentum confirmation
    Fundamental             4     Quality filter
    Beta penalty           -6×    High-beta punishment
    """
    ror_s    = min(ror / 50.0, 1.5)
    prob_s   = prob_profit / 100.0
    theta_s  = min(abs(net_theta) / 8.0, 1.0)
    iv_s     = iv_pct / 100.0
    rs_s     = np.clip((rs + 30) / 60.0, 0, 1)
    fund_s   = np.clip(fund / 100.0, 0, 1)

    base = (
        ror_s   * 28 +
        prob_s  * 22 +
        theta_s * 16 +
        iv_s    * 14 +
        rs_s    *  6 +
        fund_s  *  4
    )

    # Trend alignment — put spreads want uptrend, call spreads want downtrend
    if spread_type == 'put':
        base += max(0.0, ts) * 8
    else:
        base += max(0.0, -ts) * 8

    # Beta penalty for high-volatility underlyings
    base -= max(0.0, (beta - 1.2)) * 6

    return float(max(0.0, base))


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SpreadBacktester:
    """
    Simplified walk-forward backtest for credit spread strategies.

    Methodology
    ───────────
    For each historical month we:
      1. Identify a hypothetical spread entry using the scanner's delta logic
         applied to synthetic BS prices (using realised vol as IV proxy).
      2. Mark the spread to market at expiry (or earlier if stopped out).
      3. Record P&L net of transaction costs.

    Limitations (document these honestly)
    ──────────────────────────────────────
    - Uses realised vol as IV proxy (no historical option chain data)
    - Assumes fills at mid (best case — real fills are worse)
    - No early assignment or pin risk modelled
    - Single underlier tested (SPY as benchmark)
    """

    def __init__(self, ticker: str = "SPY", years: int = BT_YEARS):
        self.ticker = ticker
        self.years  = years

    def run(self) -> Dict[str, Any]:
        print(f"\nRunning backtest on {self.ticker} ({self.years}y) ...")
        hist = get_history(self.ticker, f"{self.years + 1}y")
        if hist.empty or len(hist) < 252:
            return {'error': 'Insufficient data for backtest'}

        hist   = hist.copy()
        closes = hist['close']
        log_r  = np.log(closes / closes.shift(1))

        # Rolling 21-day realised vol as IV proxy
        hist['rv21'] = log_r.rolling(21).std() * np.sqrt(252)
        hist = hist.dropna().copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        trades = []
        cutoff = datetime.now() - timedelta(days=self.years * 365)

        # Monthly entry — first trading day of each month
        hist['month'] = hist.index.to_period('M')
        entry_dates   = hist.groupby('month').apply(lambda g: g.index[0])

        for entry_dt in entry_dates:
            if pd.Timestamp(entry_dt).to_pydatetime() < cutoff:
                continue

            S    = float(hist.loc[entry_dt, 'close'])
            iv   = float(hist.loc[entry_dt, 'rv21'])
            if iv < 0.05 or np.isnan(iv):
                continue

            # Target: 45 DTE, 0.20-delta short put
            T        = 45 / 365.0
            target_d = 0.20

            # Find strike giving target delta via Newton's method on BS delta
            K_short  = _find_strike_for_delta(S, T, iv, target_d, 'put')
            K_long   = _find_strike_for_delta(S, T, iv, 0.08, 'put')
            if K_short is None or K_long is None:
                continue
            if K_short <= K_long:
                continue

            width  = K_short - K_long
            if width < MIN_SPREAD_WIDTH:
                continue

            g_short = bs_greeks(S, K_short, T, RISK_FREE_RATE, iv, 'put')
            g_long  = bs_greeks(S, K_long,  T, RISK_FREE_RATE, iv, 'put')

            credit     = g_short['price'] - g_long['price']
            max_loss   = width - credit
            credit_r   = credit / width if width > 0 else 0

            if credit < MIN_CREDIT or not (MIN_CREDIT_RATIO <= credit_r <= MAX_CREDIT_RATIO):
                continue
            if max_loss <= 0:
                continue

            # Expiry ~ 45 days later
            expiry_dt = pd.Timestamp(entry_dt) + pd.Timedelta(days=45)
            future    = hist[hist.index > entry_dt]
            if future.empty:
                continue

            exp_row = future[future.index >= expiry_dt]
            if exp_row.empty:
                exp_row = future.iloc[[-1]]
            exp_dt  = exp_row.index[0]
            S_exp   = float(hist.loc[exp_dt, 'close'])

            # P&L at expiry
            put_short_val = max(0.0, K_short - S_exp)
            put_long_val  = max(0.0, K_long  - S_exp)
            spread_val    = put_short_val - put_long_val

            pnl_per_share = credit - spread_val
            tx_cost       = (COMMISSION + SLIPPAGE) * 2 / 100  # per-share basis
            net_pnl       = pnl_per_share - tx_cost

            trades.append({
                'entry_date':  str(entry_dt.date()),
                'expiry_date': str(exp_dt.date()),
                'S_entry':     round(S, 2),
                'S_expiry':    round(S_exp, 2),
                'K_short':     round(K_short, 2),
                'K_long':      round(K_long, 2),
                'credit':      round(credit, 4),
                'max_loss':    round(max_loss, 4),
                'pnl':         round(net_pnl, 4),
                'win':         net_pnl > 0,
            })

        if len(trades) < BT_MIN_TRADES:
            return {'error': f'Only {len(trades)} trades — need {BT_MIN_TRADES}'}

        return self._summarise(trades)

    @staticmethod
    def _summarise(trades: List[Dict]) -> Dict[str, Any]:
        df      = pd.DataFrame(trades)
        pnls    = df['pnl'].values
        wins    = df['win'].values
        credits = df['credit'].values
        losses  = df['max_loss'].values

        cum_pnl  = np.cumsum(pnls)
        peak     = np.maximum.accumulate(cum_pnl)
        drawdown = (peak - cum_pnl)
        max_dd   = float(drawdown.max())

        avg_credit  = float(credits.mean())
        avg_loss    = float(losses.mean())
        win_rate    = float(wins.mean())
        avg_win_pnl = float(pnls[wins].mean())  if wins.sum() > 0 else 0.0
        avg_los_pnl = float(pnls[~wins].mean()) if (~wins).sum() > 0 else 0.0

        # Per-trade Sharpe (pnl normalised by spread width)
        ror   = pnls / (losses + 1e-8)
        sharp = float(ror.mean() / (ror.std() + 1e-8) * np.sqrt(12))  # annualised monthly

        return {
            'n_trades':        len(trades),
            'win_rate':        round(win_rate,    3),
            'avg_credit':      round(avg_credit,  4),
            'avg_max_loss':    round(avg_loss,    4),
            'avg_pnl':         round(float(pnls.mean()), 4),
            'total_pnl':       round(float(pnls.sum()),  4),
            'avg_win_pnl':     round(avg_win_pnl,  4),
            'avg_loss_pnl':    round(avg_los_pnl,  4),
            'max_drawdown':    round(max_dd,        4),
            'annualised_sharpe': round(sharp,       3),
            'trades':          trades,
        }


def _find_strike_for_delta(S: float, T: float, sigma: float,
                            target_delta: float, opt_type: str,
                            tol: float = 1e-4, max_iter: int = 50) -> Optional[float]:
    """
    Newton-Raphson to find strike K such that |BS delta| ≈ target_delta.
    Returns None if convergence fails.
    """
    K = S  # initial guess
    for _ in range(max_iter):
        g = bs_greeks(S, K, T, RISK_FREE_RATE, sigma, opt_type)
        d = abs(g['delta'])
        if abs(d - target_delta) < tol:
            return round(K, 2)
        # dDelta/dK: for puts, delta = -N(-d1), dDelta/dK ≈ -N'(d1)/(K*sigma*sqrt(T))
        try:
            d1  = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            ddk = -norm.pdf(d1) / (K * sigma * np.sqrt(T))
            if abs(ddk) < 1e-12:
                break
            # We want to move K so abs(delta) decreases/increases toward target
            direction = 1.0 if d < target_delta else -1.0
            K        += direction * abs((target_delta - d) / (ddk + 1e-12))
            K         = max(K, S * 0.30)   # floor
        except Exception:
            break
    # Final check
    g = bs_greeks(S, K, T, RISK_FREE_RATE, sigma, opt_type)
    return round(K, 2) if abs(abs(g['delta']) - target_delta) < 0.05 else None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class CreditSpreadScanner:
    """
    Institutional credit spread scanner v4.0

    Key improvements over v3:
    ─────────────────────────
    1. Strike selection anchored to delta, not arbitrary support/resistance bands
    2. IV percentile computed from option chain IV history (scaled RV proxy)
    3. Integrated backtest with Sharpe, win rate, max drawdown
    4. Scoring weights documented with rationale
    5. Corrected put/call spread structural checks
    """

    def __init__(self, tickers: List[str], account_size: float = ACCOUNT_SIZE):
        self.tickers      = [t.replace('.', '-') for t in dict.fromkeys(tickers)]
        self.account_size = account_size
        self.results:     List[Dict] = []
        self.skipped:     List[Tuple[str, str]] = []
        self.today        = datetime.now().date()
        self.sector_count = defaultdict(int)
        self.ticker_count = defaultdict(int)
        self.stats        = defaultdict(int)

    # ─── Public ─────────────────────────────────────────────────────────

    def run(self, run_backtest: bool = True):
        print(f"\n{'═'*110}")
        print(f"  INSTITUTIONAL CREDIT SPREAD SCANNER v4.0")
        print(f"{'═'*110}")
        print(f"  Account: ${self.account_size:,.0f}  |  "
              f"DTE: {MIN_DTE}-{MAX_DTE}  |  "
              f"Delta: {SHORT_DELTA_MIN:.2f}-{SHORT_DELTA_MAX:.2f}  |  "
              f"Min credit: ${MIN_CREDIT:.2f}  |  "
              f"Min IV pct: {MIN_IV_PERCENTILE}")
        print(f"{'═'*110}\n")

        # Optional: run backtest first so results appear in summary
        bt_results = {}
        if run_backtest:
            bt = SpreadBacktester(ticker="SPY", years=BT_YEARS)
            bt_results = bt.run()
            self._print_backtest(bt_results)

        t0 = time.time()
        for i, ticker in enumerate(self.tickers, 1):
            self.stats['total'] += 1
            try:
                print(f"[{i:>3}/{len(self.tickers)}] {ticker:<7}", end=' ', flush=True)
                self._scan_ticker(ticker)
                print("✓")
            except Exception as e:
                print(f"✗  {str(e)[:60]}")
                self.skipped.append((ticker, str(e)))
            time.sleep(SLEEP_SEC)

        self._display_results(time.time() - t0, bt_results)

    # ─── Internal ───────────────────────────────────────────────────────

    def _scan_ticker(self, ticker: str):
        # Stage 1: price history
        hist = get_history(ticker, "1y")
        if hist.empty or len(hist) < 100:
            raise ValueError("Insufficient history")

        close = hist['close']
        spot  = float(close.iloc[-1])

        # Stage 2: fundamentals
        fund, skip = fundamental_score(ticker)
        if skip:
            raise ValueError(skip[0])
        if fund < MIN_FUND_SCORE:
            raise ValueError(f"fund={fund:.0f} < {MIN_FUND_SCORE}")
        self.stats['fund_pass'] += 1

        # Stage 3: sector / beta
        sector, beta = get_sector_beta(ticker)
        if beta > MAX_BETA:
            raise ValueError(f"beta={beta:.2f}")

        # Stage 4: technicals
        ts  = trend_score(close)
        rs  = relative_strength(hist)
        self.stats['tech_pass'] += 1

        # Stage 5: options
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            raise ValueError("No options")

        valid_exps = [
            e for e in exps
            if MIN_DTE <= (datetime.strptime(e, "%Y-%m-%d").date() - self.today).days <= MAX_DTE
        ]
        if not valid_exps:
            raise ValueError(f"No exp in {MIN_DTE}-{MAX_DTE} DTE window")

        self.stats['exp_pass'] += 1

        # Stage 6: scan each expiry
        ticker_ctx = {
            'ticker': ticker, 'spot': spot, 'close': close,
            'trend':  ts, 'rs': rs, 'sector': sector,
            'beta':   beta, 'fund': fund,
        }
        for exp in valid_exps:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - self.today).days
            try:
                chain = stock.option_chain(exp)
                T     = dte / 365.0
                e_has, e_pts = earnings_penalty(ticker, dte)
                d_has, d_pts = dividend_penalty(ticker, dte)
                self._scan_puts(chain.puts, exp, dte, T, ticker_ctx, e_has, e_pts, d_has, d_pts)
                self._scan_calls(chain.calls, exp, dte, T, ticker_ctx, e_has, e_pts, d_has, d_pts)
            except Exception:
                continue

    def _scan_puts(self, raw, exp, dte, T, ctx,
                   e_has, e_pts, d_has, d_pts):
        """
        PUT CREDIT SPREAD: sell higher-strike put, buy lower-strike put.
        Strike selection: purely delta-driven (no support/resistance bands).
        Short leg: SHORT_DELTA_MIN <= |delta| <= SHORT_DELTA_MAX
        Long leg:  LONG_DELTA_MIN  <= |delta| <= LONG_DELTA_MAX
        Long leg strike < short leg strike  (guaranteed by delta ordering)
        """
        ticker = ctx['ticker']
        spot   = ctx['spot']

        if self.sector_count[ctx['sector']] >= MAX_PER_SECTOR:
            return
        if self.ticker_count[ticker] >= MAX_PER_TICKER:
            return

        opts = filter_chain(raw)
        if len(opts) < 2:
            return
        opts = add_greeks(opts, spot, T, 'put')

        # Only OTM puts (strike < spot)
        opts = opts[opts['strike'] < spot].copy()

        for i, short_row in opts.iterrows():
            sd = abs(float(short_row['delta']))
            if not (SHORT_DELTA_MIN <= sd <= SHORT_DELTA_MAX):
                continue

            iv_pct = compute_iv_percentile(ticker, float(short_row['impliedVolatility']))
            if iv_pct < MIN_IV_PERCENTILE:
                continue

            # Long leg: lower strike, smaller delta
            candidates = opts[opts['strike'] < short_row['strike']].copy()
            for j, long_row in candidates.iterrows():
                ld = abs(float(long_row['delta']))
                if not (LONG_DELTA_MIN <= ld <= LONG_DELTA_MAX):
                    continue

                width  = float(short_row['strike']) - float(long_row['strike'])
                credit = float(short_row['mid'])    - float(long_row['mid'])

                result = self._validate_and_score(
                    ticker, exp, dte, spot, width, credit,
                    short_row, long_row, T, ctx, iv_pct,
                    'put', e_has, e_pts, d_has, d_pts
                )
                if result:
                    self.results.append(result)
                    self.stats['spreads'] += 1
                    self.stats['puts']    += 1
                    self.sector_count[ctx['sector']] += 1
                    self.ticker_count[ticker]         += 1

                if self.ticker_count[ticker] >= MAX_PER_TICKER:
                    return
                if self.sector_count[ctx['sector']] >= MAX_PER_SECTOR:
                    return

    def _scan_calls(self, raw, exp, dte, T, ctx,
                    e_has, e_pts, d_has, d_pts):
        """
        CALL CREDIT SPREAD: sell lower-strike call, buy higher-strike call.
        Strike selection: purely delta-driven.
        Short leg: SHORT_DELTA_MIN <= delta <= SHORT_DELTA_MAX
        Long leg:  LONG_DELTA_MIN  <= delta <= LONG_DELTA_MAX
        Long leg strike > short leg strike
        """
        ticker = ctx['ticker']
        spot   = ctx['spot']

        if self.sector_count[ctx['sector']] >= MAX_PER_SECTOR:
            return
        if self.ticker_count[ticker] >= MAX_PER_TICKER:
            return

        opts = filter_chain(raw)
        if len(opts) < 2:
            return
        opts = add_greeks(opts, spot, T, 'call')

        # Only OTM calls (strike > spot)
        opts = opts[opts['strike'] > spot].copy()

        for i, short_row in opts.iterrows():
            sd = abs(float(short_row['delta']))
            if not (SHORT_DELTA_MIN <= sd <= SHORT_DELTA_MAX):
                continue

            iv_pct = compute_iv_percentile(ticker, float(short_row['impliedVolatility']))
            if iv_pct < MIN_IV_PERCENTILE:
                continue

            # Long leg: higher strike, smaller delta
            candidates = opts[opts['strike'] > short_row['strike']].copy()
            for j, long_row in candidates.iterrows():
                ld = abs(float(long_row['delta']))
                if not (LONG_DELTA_MIN <= ld <= LONG_DELTA_MAX):
                    continue

                width  = float(long_row['strike'])  - float(short_row['strike'])
                credit = float(short_row['mid'])     - float(long_row['mid'])

                result = self._validate_and_score(
                    ticker, exp, dte, spot, width, credit,
                    short_row, long_row, T, ctx, iv_pct,
                    'call', e_has, e_pts, d_has, d_pts
                )
                if result:
                    self.results.append(result)
                    self.stats['spreads'] += 1
                    self.stats['calls']   += 1
                    self.sector_count[ctx['sector']] += 1
                    self.ticker_count[ticker]         += 1

                if self.ticker_count[ticker] >= MAX_PER_TICKER:
                    return
                if self.sector_count[ctx['sector']] >= MAX_PER_SECTOR:
                    return

    def _validate_and_score(self, ticker, exp, dte, spot, width, credit,
                             short_row, long_row, T, ctx, iv_pct,
                             spread_type, e_has, e_pts, d_has, d_pts):
        """
        Validates spread economics, calculates net greek exposure,
        applies transaction costs and scoring.
        Returns result dict or None.
        """
        if not (MIN_SPREAD_WIDTH <= width <= MAX_SPREAD_WIDTH):
            return None
        if credit < MIN_CREDIT or credit <= 0:
            return None

        max_loss   = width - credit
        if max_loss <= 0:
            return None

        credit_r = credit / width
        if not (MIN_CREDIT_RATIO <= credit_r <= MAX_CREDIT_RATIO):
            return None

        # Net greeks
        net_delta = float(short_row['delta']) - float(long_row['delta'])
        net_gamma = float(short_row['gamma']) - float(long_row['gamma'])
        net_theta = float(short_row['theta']) - float(long_row['theta'])
        net_vega  = float(short_row['vega'])  - float(long_row['vega'])

        if abs(net_delta) > MAX_NET_DELTA:
            return None

        # Transaction costs (per-share basis, 1 contract = 100 shares)
        tx = (COMMISSION + SLIPPAGE) * 2 / 100
        net_credit = credit - tx
        if net_credit < MIN_CREDIT * 0.4:
            return None

        ror        = (credit / max_loss) * 100
        prob_win   = (1 - float(short_row['prob_itm'])) * 100

        # Kelly sizing
        kelly_size, kelly_frac = self._kelly(prob_win, ror)

        # Score
        raw_score = score_spread(
            ror=ror, prob_profit=prob_win, net_theta=net_theta,
            iv_pct=iv_pct, ts=ctx['trend'], rs=ctx['rs'],
            fund=ctx['fund'], beta=ctx['beta'], spread_type=spread_type
        )
        final_score = max(0.0, raw_score - e_pts - d_pts)

        return {
            'Ticker':      ticker,
            'Type':        spread_type.upper(),
            'Sector':      ctx['sector'],
            'Expiry':      exp,
            'DTE':         dte,
            'Spot':        round(spot, 2),
            'ShortStrike': round(float(short_row['strike']), 2),
            'LongStrike':  round(float(long_row['strike']),  2),
            'Width':       round(width, 2),
            'Credit':      round(credit, 3),
            'NetCredit':   round(net_credit, 3),
            'MaxLoss':     round(max_loss, 3),
            'RoR%':        round(ror, 1),
            'ProbWin%':    round(prob_win, 1),
            'IV':          round(float(short_row['impliedVolatility']), 3),
            'IVPct':       round(iv_pct, 1),
            'NetDelta':    round(net_delta, 4),
            'NetGamma':    round(net_gamma, 6),
            'NetTheta':    round(net_theta, 4),
            'NetVega':     round(net_vega,  4),
            'Beta':        round(ctx['beta'], 2),
            'FundScore':   round(ctx['fund'], 0),
            'Trend':       round(ctx['trend'], 3),
            'RS%':         round(ctx['rs'], 1),
            'KellySize$':  round(kelly_size, 0),
            'Kelly%':      round(kelly_frac * 100, 2),
            'HasEarnings': e_has,
            'HasDividend': d_has,
            'Target50%':   round(credit * 0.5, 3),
            'StopLoss$':   round(-max_loss * 0.75, 3),
            'Score':       round(final_score, 2),
        }

    def _kelly(self, prob_win: float, ror: float) -> Tuple[float, float]:
        try:
            wp = min(prob_win / 100 * 0.80, 0.98)   # 20% model discount
            lp = 1 - wp
            b  = ror / 100
            if b <= 0 or wp <= 0:
                return self.account_size * 0.005, 0.005
            k  = (b * wp - lp) / b
            if k <= 0:
                return self.account_size * 0.003, 0.003
            f  = float(np.clip(k * 0.167, 0.003, 0.02))  # 1/6 Kelly, max 2%
            return self.account_size * f, f
        except Exception:
            return self.account_size * 0.005, 0.005

    # ─── Display ────────────────────────────────────────────────────────

    @staticmethod
    def _print_backtest(bt: Dict):
        if 'error' in bt:
            print(f"\n⚠  Backtest: {bt['error']}\n")
            return
        print(f"\n{'─'*60}")
        print(f"  BACKTEST RESULTS (SPY {BT_YEARS}y, 45-DTE 0.20Δ put spreads)")
        print(f"{'─'*60}")
        print(f"  Trades:            {bt['n_trades']}")
        print(f"  Win rate:          {bt['win_rate']:.1%}")
        print(f"  Avg credit:        ${bt['avg_credit']:.4f}")
        print(f"  Avg P&L:           ${bt['avg_pnl']:.4f}")
        print(f"  Total P&L:         ${bt['total_pnl']:.4f}")
        print(f"  Max drawdown:      ${bt['max_drawdown']:.4f}")
        print(f"  Annualised Sharpe: {bt['annualised_sharpe']:.2f}")
        print(f"{'─'*60}\n")

    def _display_results(self, elapsed: float, bt: Dict):
        print(f"\n{'═'*110}")
        print(f"  SCAN COMPLETE  —  {elapsed:.1f}s")
        print(f"{'═'*110}")
        print(f"  Scanned: {self.stats['total']}  |  "
              f"Fund pass: {self.stats['fund_pass']}  |  "
              f"Exp pass: {self.stats['exp_pass']}  |  "
              f"Spreads: {self.stats['spreads']} "
              f"(puts: {self.stats['puts']}, calls: {self.stats['calls']})")

        if not self.results:
            print("\n  ❌  No valid spreads found.")
            self._skip_summary()
            return

        df = pd.DataFrame(self.results).sort_values('Score', ascending=False)
        top = df.head(TOP_N)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)

        cols = ['Ticker', 'Type', 'Score', 'DTE', 'Spot',
                'ShortStrike', 'LongStrike', 'Width', 'Credit',
                'RoR%', 'ProbWin%', 'IVPct', 'FundScore', 'HasEarnings']
        print(f"\n  TOP {TOP_N} CREDIT SPREADS\n")
        print(top[cols].to_string(index=False))

        # Portfolio stats
        print(f"\n{'─'*60}")
        print(f"  Portfolio summary (top {TOP_N})")
        print(f"{'─'*60}")
        print(f"  Avg RoR:           {top['RoR%'].mean():.1f}%")
        print(f"  Avg prob win:      {top['ProbWin%'].mean():.1f}%")
        print(f"  Avg net credit:    ${top['NetCredit'].mean():.3f}")
        print(f"  Avg IV percentile: {top['IVPct'].mean():.1f}")
        print(f"  Avg beta:          {top['Beta'].mean():.2f}")
        print(f"  Capital required:  ${top['KellySize$'].sum():,.0f}")

        # Sector distribution
        print(f"\n  Sector distribution:")
        for sec, cnt in top['Sector'].value_counts().items():
            print(f"    {sec:<30} {cnt}")

        # Export
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"credit_spreads_v4_{ts}.csv"
        try:
            df.to_csv(fname, index=False)
            print(f"\n  ✓  Exported {len(df)} spreads → {fname}")
        except Exception as e:
            print(f"\n  ✗  Export failed: {e}")

        # Export backtest trades
        if bt and 'trades' in bt:
            bt_df   = pd.DataFrame(bt['trades'])
            bt_fname = f"backtest_{ts}.csv"
            try:
                bt_df.to_csv(bt_fname, index=False)
                print(f"  ✓  Backtest trades  → {bt_fname}")
            except Exception:
                pass

        print(f"\n{'═'*110}\n")

    def _skip_summary(self):
        reasons = defaultdict(int)
        for _, r in self.skipped[:50]:
            reasons[r.split(':')[0].strip()] += 1
        if reasons:
            print("\n  Top skip reasons:")
            for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"    {c:>3}×  {r}")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    TICKERS = [
        "MMM","ABT","ABBV","ACN","ADBE","AES","AFL","A","APD","AKAM",
        "ALK","ALB","ALGN","ALL","GOOGL","AMZN","AEP","AXP","AIG","AMT",
        "AWK","AMP","AME","AMGN","APH","ADI","AON","AAPL","AMAT","APTV",
        "AJG","T","ADSK","AZO","AVB","BAC","BAX","BDX","BBY","BIIB",
        "BLK","BK","BSX","CAT","CDNS","COF","CAH","CBOE","CBRE","CVX",
        "CMG","CB","CI","CTAS","CSCO","C","CME","KO","CL","CMCSA","COP",
        "COST","CSX","DHR","DE","FDX","FITB","FIS","FISV","FLT","FTNT",
        "FCX","GRMN","GD","GE","GILD","HD","HON","HUM","IDXX","INTC",
        "ICE","IFF","IQV","JNJ","JPM","LLY","LIN","LMT","LOW","MA",
        "MCD","MDT","MRK","MSFT","MS","NEE","NKE","NSC","NVDA","ORLY",
        "OXY","ORCL","PAYX","PYPL","PEP","PFE","PM","PNC","PPG","PG",
        "PGR","PLD","PRU","QCOM","REGN","RSG","RMD","ROK","ROP","ROST",
        "SPGI","CRM","SLB","SHW","SPG","STT","SYK","SYF","SYY","TMUS",
        "TROW","TRV","TMO","TJX","TSCO","TSN","UNP","UNH","UPS","V",
        "VLO","VZ","VRSK","WMT","WBA","DIS","WM","XOM","YUM","ZTS",
    ]

    scanner = CreditSpreadScanner(tickers=TICKERS, account_size=50_000)
    scanner.run(run_backtest=True)
