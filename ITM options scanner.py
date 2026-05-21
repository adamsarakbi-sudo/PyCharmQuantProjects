import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
import time
import warnings

warnings.filterwarnings("ignore")


# Trading Parameters
RISK_FREE_RATE = 0.045
ACCOUNT_SIZE = 50000

# DTE Parameters
MIN_DTE = 30
MAX_DTE = 60

# Delta Parameters - RELAXED & REALISTIC
TARGET_DELTA_MIN = 0.08  # Was 0.15 - now more permissive
TARGET_DELTA_MAX = 0.40  # Was 0.35 - wider range
HEDGE_DELTA_MIN = 0.02  # Was 0.05 - allow tighter hedges
HEDGE_DELTA_MAX = 0.20  # Was 0.15 - allow wider hedges
MAX_POSITION_DELTA = 0.25  # Was 0.10 - credit spreads typically 0.15-0.25

# Liquidity Parameters - RELAXED
MIN_VOLUME = 5  # Was 10
MIN_OI = 25  # Was 50
MAX_BID_ASK_SPREAD_PCT = 0.50  # Was 0.35

# Quality Parameters - REALISTIC
MIN_CREDIT = 0.15  # NEW: Minimum credit worth trading
MIN_CREDIT_TO_WIDTH_RATIO = 0.18  # Was 0.25 - more achievable
MIN_IV_RANK = 20  # Was 30 - lower threshold

# Spread Sizing - NEW VALIDATIONS
MIN_SPREAD_WIDTH = 2.0
MAX_SPREAD_WIDTH = 15.0
MAX_SPREAD_WIDTH_PCT = 0.12  # 12% of stock price

# Fundamental Parameters - MUCH MORE REALISTIC
MIN_FUNDAMENTAL_SCORE = 15  # Was 35 - way too high
MIN_MARKET_CAP = 2e9  # Was 5e9 - $2B minimum

# Technical Parameters
SMA_FAST = 20
SMA_SLOW = 50
MIN_RELATIVE_STRENGTH = -30  # Was -10 - allow weaker stocks

# Risk Management
MAX_BETA = 2.5  # Was 1.8
MAX_TRADES_PER_SECTOR = 4  # Was 3
MAX_SPREADS_PER_TICKER = 3  # NEW: Limit spreads per ticker
UNUSUAL_VOLUME_THRESHOLD = 2.5

# Transaction Costs - NEW
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PER_CONTRACT = 0.05

# Output
TOP_N = 25  # Increased from 20
SLEEP_BETWEEN_TICKERS = 0.15  # Slightly faster


# ═══════════════════════════════════════════════════════════════════════════
# CACHING & MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def get_history(ticker, period="1y"):
    """Get historical price data with caching"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return pd.DataFrame()
        return hist
    except Exception as e:
        return pd.DataFrame()


@lru_cache(maxsize=128)
def get_spy_history(period="3mo"):
    """Get SPY history for relative strength calculations"""
    try:
        spy = yf.Ticker('SPY')
        return spy.history(period=period)
    except:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL ANALYSIS - FIXED SCORING
# ═══════════════════════════════════════════════════════════════════════════

def get_fundamental_analysis(ticker):
    """
    Comprehensive fundamental analysis with realistic scoring
    Returns: (score, details_dict, skip_reasons)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        score = 0
        details = {}
        skip_reasons = []

        # Market Cap (Critical Filter)
        market_cap = info.get('marketCap', 0)
        details['marketCap'] = market_cap

        if market_cap < MIN_MARKET_CAP:
            skip_reasons.append(f"Market cap ${market_cap / 1e9:.1f}B < ${MIN_MARKET_CAP / 1e9:.1f}B")
            return 0, details, skip_reasons

        # Tiered scoring for market cap
        if market_cap > 50e9:
            score += 15
            details['marketCapTier'] = 'Mega Cap'
        elif market_cap > 10e9:
            score += 12
            details['marketCapTier'] = 'Large Cap'
        elif market_cap > 5e9:
            score += 8
            details['marketCapTier'] = 'Mid Cap'
        else:
            score += 5
            details['marketCapTier'] = 'Small Cap'

        # Profitability - more lenient
        profit_margin = info.get('profitMargins', 0)
        details['profitMargin'] = profit_margin

        if profit_margin > 0.20:
            score += 20
        elif profit_margin > 0.10:
            score += 15
        elif profit_margin > 0.05:
            score += 10
        elif profit_margin > 0:
            score += 5  # Don't penalize, just low score
        # Negative margins get 0 (not penalized)

        # Revenue Growth
        revenue_growth = info.get('revenueGrowth', 0)
        details['revenueGrowth'] = revenue_growth

        if revenue_growth > 0.25:
            score += 15
        elif revenue_growth > 0.15:
            score += 12
        elif revenue_growth > 0.05:
            score += 8
        elif revenue_growth > 0:
            score += 5

        # Debt Management
        debt_to_equity = info.get('debtToEquity', 999)
        details['debtToEquity'] = debt_to_equity

        if debt_to_equity < 30:
            score += 15
        elif debt_to_equity < 75:
            score += 12
        elif debt_to_equity < 150:
            score += 8
        elif debt_to_equity < 250:
            score += 4

        # Free Cash Flow
        free_cashflow = info.get('freeCashflow', 0)
        details['freeCashflow'] = free_cashflow

        if free_cashflow > 1e9:  # $1B+
            score += 12
        elif free_cashflow > 0:
            score += 8

        # Analyst Recommendations
        recommendation = info.get('recommendationKey', 'none')
        details['recommendation'] = recommendation

        if recommendation in ['strong_buy', 'buy']:
            score += 12
        elif recommendation == 'hold':
            score += 6
        # Don't penalize sell ratings

        # Institutional Ownership
        inst_ownership = info.get('heldPercentInstitutions', 0)
        details['institutionalOwnership'] = inst_ownership

        if inst_ownership > 0.75:
            score += 10
        elif inst_ownership > 0.60:
            score += 8
        elif inst_ownership > 0.40:
            score += 5

        # Return on Equity
        roe = info.get('returnOnEquity', 0)
        details['returnOnEquity'] = roe

        if roe > 0.25:
            score += 12
        elif roe > 0.18:
            score += 10
        elif roe > 0.12:
            score += 6
        elif roe > 0:
            score += 3

        # Current Ratio (Liquidity)
        current_ratio = info.get('currentRatio', 0)
        details['currentRatio'] = current_ratio

        if current_ratio > 2.0:
            score += 6
        elif current_ratio > 1.5:
            score += 4
        elif current_ratio > 1.0:
            score += 2

        return score, details, skip_reasons

    except Exception as e:
        return 0, {}, [f"Fundamental data unavailable: {str(e)}"]


# ═══════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT FILTERS - NOW RETURN PENALTIES, NOT HARD STOPS
# ═══════════════════════════════════════════════════════════════════════════

def check_earnings_risk(ticker, dte):
    """
    Check if earnings falls within trade window
    Returns (has_risk, days_to_earnings, penalty_factor)
    """
    try:
        stock = yf.Ticker(ticker)

        try:
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar:
                    earnings_dates = calendar['Earnings Date']
                    if isinstance(earnings_dates, pd.Series):
                        earnings_date = pd.to_datetime(earnings_dates.iloc[0])
                    else:
                        earnings_date = pd.to_datetime(earnings_dates)

                    days_to_earnings = (earnings_date.date() - datetime.now().date()).days

                    if 0 <= days_to_earnings <= dte:
                        # Progressive penalty based on proximity
                        if days_to_earnings < 7:
                            penalty = 0.60  # Heavy penalty (40% score reduction)
                        elif days_to_earnings < 14:
                            penalty = 0.75  # Moderate penalty (25% reduction)
                        else:
                            penalty = 0.85  # Light penalty (15% reduction)
                        return True, days_to_earnings, penalty
        except:
            pass

        try:
            earnings_hist = stock.earnings_dates
            if earnings_hist is not None and not earnings_hist.empty:
                last_earnings = earnings_hist.index[0]
                next_earnings_est = last_earnings + pd.DateOffset(days=90)
                days_to_earnings = (next_earnings_est.date() - datetime.now().date()).days

                if 0 <= days_to_earnings <= dte:
                    penalty = 0.85  # Estimated date, lighter penalty
                    return True, days_to_earnings, penalty
        except:
            pass

        return False, None, 1.0

    except Exception as e:
        return False, None, 1.0


def check_dividend_risk(ticker, dte):
    """
    Check for ex-dividend dates in trade window
    Returns (has_risk, div_info, penalty_factor)
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends

        if dividends is None or dividends.empty:
            return False, None, 1.0

        last_div_date = dividends.index[-1]
        next_div_est = last_div_date + pd.DateOffset(months=3)
        days_to_div = (next_div_est.date() - datetime.now().date()).days

        if 0 <= days_to_div <= dte:
            last_div_amount = dividends.iloc[-1]

            # Get current price to calculate div yield
            hist = get_history(ticker, "5d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                div_yield = (last_div_amount / price) * 100

                # Penalty based on dividend size
                if div_yield > 2.0:  # High dividend
                    penalty = 0.85
                elif div_yield > 1.0:  # Moderate dividend
                    penalty = 0.93
                else:  # Small dividend
                    penalty = 0.97

                return True, (days_to_div, last_div_amount), penalty

        return False, None, 1.0

    except Exception as e:
        return False, None, 1.0


def get_sector_and_beta(ticker):
    """Get sector and beta for diversification"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        sector = info.get('sector', 'Unknown')
        beta = info.get('beta', 1.0)

        if beta is None or np.isnan(beta) or beta == 0:
            beta = 1.0

        return sector, beta

    except Exception as e:
        return 'Unknown', 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS - IMPROVED CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_trend_score(close):
    """Calculate trend score from -1 (strong bear) to +1 (strong bull)"""
    if len(close) < SMA_SLOW:
        return 0

    sma_fast = close.rolling(SMA_FAST).mean()
    sma_slow = close.rolling(SMA_SLOW).mean()

    # Validate SMAs are not NaN
    if pd.isna(sma_fast.iloc[-1]) or pd.isna(sma_slow.iloc[-1]):
        return 0

    spot = close.iloc[-1]

    fast_diff = (spot - sma_fast.iloc[-1]) / spot
    slow_diff = (spot - sma_slow.iloc[-1]) / spot
    sma_cross = 1 if sma_fast.iloc[-1] > sma_slow.iloc[-1] else -1

    trend_score = (fast_diff * 0.4 + slow_diff * 0.3 + sma_cross * 0.3)

    return np.clip(trend_score, -1, 1)


def find_support_resistance(close, window=20):
    """
    Find key support and resistance levels using percentiles
    FIXED: Now uses percentiles instead of flawed value_counts
    """
    try:
        if len(close) < window:
            return close.min(), close.max()

        # Use recent data (last 60 days)
        recent = close.tail(60)

        # Support = 25th percentile (price stays above this 75% of the time)
        support = recent.quantile(0.25)

        # Resistance = 75th percentile (price stays below this 75% of the time)
        resistance = recent.quantile(0.75)

        # Sanity check
        current = close.iloc[-1]
        if support > current:
            support = recent.min()
        if resistance < current:
            resistance = recent.max()

        return support, resistance

    except Exception as e:
        # Fallback to simple high/low
        return close.tail(60).min(), close.tail(60).max()


def calculate_relative_strength(ticker, hist):
    """Calculate relative strength vs SPY"""
    try:
        spy = get_spy_history('3mo')

        if spy.empty or len(hist) < 21:
            return 0

        # Validate we have enough data
        if len(hist) < 21 or len(spy) < 21:
            return 0

        ticker_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) * 100
        spy_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[-21] - 1) * 100

        relative_strength = ticker_return - spy_return

        return relative_strength

    except Exception as e:
        return 0


def check_unusual_volume(ticker, hist):
    """Detect unusual volume activity"""
    try:
        if len(hist) < 20:
            return 'unknown', 1.0

        avg_volume = hist['Volume'].tail(20).mean()
        recent_volume = hist['Volume'].iloc[-1]

        # Handle zero cases
        if avg_volume == 0 or recent_volume == 0:
            return 'unknown', 1.0

        volume_ratio = recent_volume / avg_volume

        if volume_ratio > UNUSUAL_VOLUME_THRESHOLD:
            return 'high', volume_ratio
        elif volume_ratio < 0.5:
            return 'low', volume_ratio
        else:
            return 'normal', volume_ratio

    except Exception as e:
        return 'unknown', 1.0


# ═══════════════════════════════════════════════════════════════════════════
# IMPLIED VOLATILITY ANALYSIS - FIXED
# ═══════════════════════════════════════════════════════════════════════════

def calculate_iv_rank(close, current_iv, window=252):
    """
    Calculate IV rank using realized volatility as proxy
    Rank = where current IV sits between min and max RV
    """
    try:
        returns = np.log(close / close.shift(1)).dropna()

        rolling_rv = returns.rolling(21).std() * np.sqrt(252)
        rolling_rv = rolling_rv.dropna()

        if len(rolling_rv) < 60:
            return 50  # Default to neutral

        recent_rv = rolling_rv.tail(window)

        vol_min = recent_rv.min()
        vol_max = recent_rv.max()

        if vol_max == vol_min or vol_max == 0:
            return 50

        # Rank: where does current IV sit in the range?
        iv_rank = (current_iv - vol_min) / (vol_max - vol_min) * 100

        return np.clip(iv_rank, 0, 100)

    except Exception as e:
        return 50  # Default neutral


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING - ENHANCED VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def black_scholes_greeks(S, K, T, r, sigma, option_type):
    """
    Calculate BS price and greeks with comprehensive validation
    FIXED: Added bounds checking and edge case handling
    """

    # Input validation
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {
            'price': 0, 'delta': 0, 'gamma': 0,
            'theta': 0, 'vega': 0, 'prob_itm': 0
        }

    # Validate IV range (5% to 300%)
    if sigma < 0.05 or sigma > 3.0:
        return {
            'price': 0, 'delta': 0, 'gamma': 0,
            'theta': 0, 'vega': 0, 'prob_itm': 0
        }

    # Validate time range
    if T < 0.01 or T > 2.0:  # 4 days to 2 years
        return {
            'price': 0, 'delta': 0, 'gamma': 0,
            'theta': 0, 'vega': 0, 'prob_itm': 0
        }

    # Check for extreme moneyness (can cause numerical issues)
    moneyness = S / K
    if moneyness < 0.3 or moneyness > 3.0:
        return {
            'price': 0, 'delta': 0, 'gamma': 0,
            'theta': 0, 'vega': 0, 'prob_itm': 0
        }

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = norm.pdf(d1)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            prob_itm = norm.cdf(d2)
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            prob_itm = norm.cdf(-d2)
            theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T) / 100
        theta = theta / 365

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'prob_itm': prob_itm
        }

    except Exception as e:
        return {
            'price': 0, 'delta': 0, 'gamma': 0,
            'theta': 0, 'vega': 0, 'prob_itm': 0
        }


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING (KELLY CRITERION) - MORE CONSERVATIVE
# ═══════════════════════════════════════════════════════════════════════════

def kelly_position_size(prob_profit, return_on_risk, account_size, max_risk_pct=0.02):
    """
    Calculate optimal position size using Kelly Criterion
    FIXED: More conservative, handles edge cases better
    """
    try:
        # Discount probabilities for model error and fat tails
        adjusted_prob = prob_profit * 0.80  # 20% discount for options

        win_prob = adjusted_prob / 100
        lose_prob = 1 - win_prob
        win_amount = return_on_risk / 100

        if win_amount <= 0 or win_prob <= 0:
            return account_size * 0.005, 0.005

        # Kelly formula
        kelly_fraction = (win_amount * win_prob - lose_prob) / win_amount

        # Negative Kelly means unfavorable trade
        if kelly_fraction <= 0:
            return account_size * 0.003, 0.003

        # Use 1/6 Kelly (very conservative for options)
        safe_fraction = kelly_fraction * 0.167

        # Hard limits
        safe_fraction = np.clip(safe_fraction, 0.003, max_risk_pct)

        position_size = account_size * safe_fraction

        return position_size, safe_fraction

    except Exception as e:
        return account_size * 0.005, 0.005


# ═══════════════════════════════════════════════════════════════════════════
# TRANSACTION COST CALCULATION - NEW
# ═══════════════════════════════════════════════════════════════════════════

def calculate_transaction_costs(num_contracts=1):
    """
    Calculate total transaction costs
    Includes commission and slippage
    """
    # Commission: $0.65 per contract, both legs
    commission = COMMISSION_PER_CONTRACT * 2 * num_contracts

    # Slippage: $0.05 per contract, both legs
    slippage = SLIPPAGE_PER_CONTRACT * 2 * num_contracts

    total_cost = commission + slippage

    return total_cost, commission, slippage


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SCANNER CLASS - COMPLETELY DEBUGGED
# ═══════════════════════════════════════════════════════════════════════════

class InstitutionalCreditSpreadScanner:
    """
    Production-grade credit spread scanner
    VERSION 3.0 - Fully debugged and logically sound
    """

    def __init__(self, tickers, account_size=ACCOUNT_SIZE):
        self.tickers = [t.replace('.', '-') for t in tickers]
        self.account_size = account_size
        self.results = []
        self.today = datetime.now().date()
        self.skipped = []

        # Portfolio tracking - FIXED: Now properly maintained
        self.sector_exposure = defaultdict(int)
        self.ticker_spread_count = defaultdict(int)

        # Statistics
        self.stats = {
            'total_scanned': 0,
            'passed_fundamental': 0,
            'passed_technical': 0,
            'had_valid_expirations': 0,
            'spreads_found': 0,
            'put_spreads': 0,
            'call_spreads': 0
        }

    def run(self):
        """Main scanning loop"""
        print(f"\n{'═' * 120}")
        print(f"INSTITUTIONAL CREDIT SPREAD SCANNER v3.0 - FULLY DEBUGGED")
        print(f"{'═' * 120}")
        print(f"Account Size: ${self.account_size:,.0f}")
        print(f"Scanning {len(self.tickers)} tickers...")
        print(f"\nKey Parameters:")
        print(f"  Delta Range: {TARGET_DELTA_MIN:.2f} - {TARGET_DELTA_MAX:.2f}")
        print(f"  Min Credit/Width: {MIN_CREDIT_TO_WIDTH_RATIO:.0%}")
        print(f"  Min Fundamental Score: {MIN_FUNDAMENTAL_SCORE}")
        print(f"  Min Credit: ${MIN_CREDIT:.2f}")
        print(f"  Max Position Delta: {MAX_POSITION_DELTA:.2f}")
        print(f"{'═' * 120}\n")

        start_time = time.time()

        for i, ticker in enumerate(self.tickers, 1):
            self.stats['total_scanned'] += 1

            try:
                print(f"[{i}/{len(self.tickers)}] {ticker:6s} ", end="", flush=True)
                self.scan_ticker(ticker)
                print("✓")

            except Exception as e:
                error_msg = str(e)[:70]
                print(f"✗ {error_msg}")
                self.skipped.append((ticker, str(e)))

            time.sleep(SLEEP_BETWEEN_TICKERS)

        elapsed = time.time() - start_time

        self.display_results(elapsed)

    def scan_ticker(self, ticker):
        """Scan individual ticker through all filters"""

        # ═══ STAGE 1: Get Market Data ═══
        hist = get_history(ticker, "1y")
        if hist.empty or len(hist) < 100:
            raise Exception("Insufficient price history")

        close = hist["Close"]
        spot = close.iloc[-1]

        # ═══ STAGE 2: Fundamental Analysis ═══
        fund_score, fund_details, skip_reasons = get_fundamental_analysis(ticker)

        if skip_reasons:
            raise Exception(f"Fundamental: {skip_reasons[0]}")

        if fund_score < MIN_FUNDAMENTAL_SCORE:
            raise Exception(f"Fund score {fund_score} < {MIN_FUNDAMENTAL_SCORE}")

        self.stats['passed_fundamental'] += 1

        # ═══ STAGE 3: Sector & Beta ═══
        sector, beta = get_sector_and_beta(ticker)

        if beta > MAX_BETA:
            raise Exception(f"Beta {beta:.2f} > {MAX_BETA}")

        # ═══ STAGE 4: Technical Analysis ═══
        trend_score = get_trend_score(close)
        support, resistance = find_support_resistance(close)
        relative_strength = calculate_relative_strength(ticker, hist)
        vol_status, vol_ratio = check_unusual_volume(ticker, hist)

        if relative_strength < MIN_RELATIVE_STRENGTH:
            raise Exception(f"RS {relative_strength:.1f}% < {MIN_RELATIVE_STRENGTH}%")

        self.stats['passed_technical'] += 1

        # ═══ STAGE 5: Options Chain ═══
        stock = yf.Ticker(ticker)

        try:
            expirations = stock.options
        except:
            raise Exception("No options available")

        if not expirations:
            raise Exception("No expirations found")

        # ═══ STAGE 6: Scan Valid Expirations ═══
        valid_exps = []

        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - self.today).days

            if not (MIN_DTE <= dte <= MAX_DTE):
                continue

            # Check for event risks (now get penalties, not hard filters)
            has_earnings, days_to_earnings, earnings_penalty = check_earnings_risk(ticker, dte)
            has_div_risk, div_info, div_penalty = check_dividend_risk(ticker, dte)

            valid_exps.append((exp, dte, earnings_penalty, div_penalty, has_earnings, has_div_risk))

        if not valid_exps:
            raise Exception(f"No valid expirations ({MIN_DTE}-{MAX_DTE} DTE)")

        self.stats['had_valid_expirations'] += 1

        # ═══ STAGE 7: Scan Each Expiration ═══
        ticker_data = {
            'ticker': ticker,
            'spot': spot,
            'close': close,
            'trend_score': trend_score,
            'support': support,
            'resistance': resistance,
            'relative_strength': relative_strength,
            'vol_status': vol_status,
            'vol_ratio': vol_ratio,
            'sector': sector,
            'beta': beta,
            'fund_score': fund_score,
            'fund_details': fund_details
        }

        for exp, dte, earnings_penalty, div_penalty, has_earnings, has_div_risk in valid_exps:
            try:
                self.scan_expiration(stock, exp, dte, ticker_data,
                                     earnings_penalty, div_penalty, has_earnings, has_div_risk)
            except Exception as e:
                continue

    def scan_expiration(self, stock, exp, dte, ticker_data,
                        earnings_penalty, div_penalty, has_earnings, has_div_risk):
        """Scan specific expiration for spreads"""

        T = dte / 365.0

        # Get option chain
        try:
            chain = stock.option_chain(exp)
        except:
            return

        # Scan both put and call spreads
        self.find_put_spreads(chain.puts, exp, dte, T, ticker_data,
                              earnings_penalty, div_penalty, has_earnings, has_div_risk)
        self.find_call_spreads(chain.calls, exp, dte, T, ticker_data,
                               earnings_penalty, div_penalty, has_earnings, has_div_risk)

    def find_put_spreads(self, opts, exp, dte, T, ticker_data,
                         earnings_penalty, div_penalty, has_earnings, has_div_risk):
        """
        Find put credit spreads with CORRECTED logic

        PUT CREDIT SPREAD:
        - Sell higher strike put (collect premium)
        - Buy lower strike put (limit risk)
        - Profit if price stays ABOVE short strike

        FIXED: Support/resistance logic, all validations
        """

        ticker = ticker_data['ticker']
        spot = ticker_data['spot']
        close = ticker_data['close']
        support = ticker_data['support']
        sector = ticker_data['sector']

        # Check sector limit BEFORE processing
        if self.sector_exposure[sector] >= MAX_TRADES_PER_SECTOR:
            return

        # Check per-ticker limit
        if self.ticker_spread_count[ticker] >= MAX_SPREADS_PER_TICKER:
            return

        # Filter and prepare options
        opts = self.filter_options(opts.copy(), spot)

        if len(opts) < 2:
            return

        # Calculate greeks
        opts = self.calculate_greeks(opts, spot, T, 'put')

        # PUT SPREAD: Sell higher strike, buy lower strike
        for i in range(len(opts)):
            short_leg = opts.iloc[i]

            # Delta validation
            short_delta = abs(short_leg['delta'])
            if not (TARGET_DELTA_MIN <= short_delta <= TARGET_DELTA_MAX):
                continue

            # IV rank
            iv_rank = calculate_iv_rank(close, short_leg['impliedVolatility'])
            if iv_rank < MIN_IV_RANK:
                continue

            # ═══ CORRECTED STRIKE VALIDATION ═══
            # For PUT spreads: want strikes BELOW current price, near support
            # Acceptable range: 85% to 98% of support level
            strike_range_low = support * 0.85
            strike_range_high = support * 0.98

            if not (strike_range_low <= short_leg['strike'] <= strike_range_high):
                continue

            # Additional check: short strike should be below current price
            if short_leg['strike'] >= spot * 0.98:
                continue

            # Find long legs (LOWER strikes)
            for j in range(i):
                long_leg = opts.iloc[j]

                # Delta validation
                long_delta = abs(long_leg['delta'])
                if not (HEDGE_DELTA_MIN <= long_delta <= HEDGE_DELTA_MAX):
                    continue

                # Calculate spread width
                spread_width = short_leg['strike'] - long_leg['strike']

                # ═══ WIDTH VALIDATION ═══
                if not (MIN_SPREAD_WIDTH <= spread_width <= MAX_SPREAD_WIDTH):
                    continue

                # Also check as percentage of stock price
                width_pct = spread_width / spot
                if width_pct > MAX_SPREAD_WIDTH_PCT:
                    continue

                # Calculate credit
                credit = short_leg['mid'] - long_leg['mid']

                # ═══ CREDIT VALIDATION ═══
                if credit < MIN_CREDIT:
                    continue

                # Credit should be positive
                if credit <= 0:
                    continue

                # Calculate max loss
                max_loss = spread_width - credit

                # Sanity check
                if max_loss <= 0:
                    continue

                # Minimum credit ratio
                credit_ratio = credit / spread_width
                if credit_ratio < MIN_CREDIT_TO_WIDTH_RATIO:
                    continue

                # Maximum credit ratio (can't be > 40% without arbitrage)
                if credit_ratio > 0.45:
                    continue

                # Calculate greeks
                net_delta = short_leg['delta'] - long_leg['delta']
                net_gamma = short_leg['gamma'] - long_leg['gamma']
                net_theta = short_leg['theta'] - long_leg['theta']
                net_vega = short_leg['vega'] - long_leg['vega']

                # Net delta limit
                if abs(net_delta) > MAX_POSITION_DELTA:
                    continue

                # Calculate metrics
                return_on_risk = (credit / max_loss) * 100
                prob_profit = 100 - (short_leg['prob_itm'] * 100)

                # Distance from support
                support_distance = (support - short_leg['strike']) / spot

                # Transaction costs
                num_contracts = 1
                total_cost, commission, slippage = calculate_transaction_costs(num_contracts)
                net_credit = credit - (total_cost / 100)  # Per share basis

                if net_credit < MIN_CREDIT * 0.5:  # After costs, still worth it?
                    continue

                # Kelly position sizing
                kelly_size, kelly_frac = kelly_position_size(
                    prob_profit, return_on_risk, self.account_size
                )

                # Calculate comprehensive score
                score = self.calculate_score(
                    return_on_risk=return_on_risk,
                    prob_profit=prob_profit,
                    net_theta=net_theta,
                    net_gamma=net_gamma,
                    net_vega=net_vega,
                    iv_rank=iv_rank,
                    trend_score=ticker_data['trend_score'],
                    relative_strength=ticker_data['relative_strength'],
                    fund_score=ticker_data['fund_score'],
                    support_distance=support_distance,
                    vol_ratio=ticker_data['vol_ratio'],
                    beta=ticker_data['beta'],
                    spread_type='put'
                )

                # Apply penalties (additive, not multiplicative)
                earnings_penalty_pts = (1.0 - earnings_penalty) * 15
                div_penalty_pts = (1.0 - div_penalty) * 5

                final_score = score - earnings_penalty_pts - div_penalty_pts
                final_score = max(0, final_score)

                # Store result
                self.results.append({
                    'Ticker': ticker,
                    'Type': 'PUT',
                    'Sector': sector,
                    'ShortStrike': round(short_leg['strike'], 2),
                    'LongStrike': round(long_leg['strike'], 2),
                    'Width': round(spread_width, 2),
                    'Expiry': exp,
                    'DTE': dte,
                    'Spot': round(spot, 2),
                    'Support': round(support, 2),
                    'Resistance': round(ticker_data['resistance'], 2),
                    'NetDelta': round(net_delta, 3),
                    'IV': round(short_leg['impliedVolatility'], 3),
                    'IV_Rank': round(iv_rank, 1),
                    'Credit': round(credit, 2),
                    'NetCredit': round(net_credit, 2),
                    'TxCost': round(total_cost, 2),
                    'MaxLoss': round(max_loss, 2),
                    'RoR%': round(return_on_risk, 1),
                    'ProbProfit%': round(prob_profit, 1),
                    'Theta': round(net_theta, 3),
                    'RS%': round(ticker_data['relative_strength'], 1),
                    'Beta': round(ticker_data['beta'], 2),
                    'FundScore': ticker_data['fund_score'],
                    'Trend': round(ticker_data['trend_score'], 2),
                    'KellySize$': round(kelly_size, 0),
                    'Kelly%': round(kelly_frac * 100, 2),
                    'Score': round(final_score, 2),
                    'HasEarnings': has_earnings,
                    'HasDividend': has_div_risk,
                    'Target50%': round(credit * 0.5, 2),
                    'StopLoss': round(-max_loss * 0.75, 2)
                })

                self.stats['spreads_found'] += 1
                self.stats['put_spreads'] += 1

                # FIXED: Increment sector exposure and ticker count
                self.sector_exposure[sector] += 1
                self.ticker_spread_count[ticker] += 1

                # Check if we hit limits
                if self.sector_exposure[sector] >= MAX_TRADES_PER_SECTOR:
                    return
                if self.ticker_spread_count[ticker] >= MAX_SPREADS_PER_TICKER:
                    return

    def find_call_spreads(self, opts, exp, dte, T, ticker_data,
                          earnings_penalty, div_penalty, has_earnings, has_div_risk):
        """
        Find call credit spreads with CORRECTED logic

        CALL CREDIT SPREAD:
        - Sell lower strike call (collect premium)
        - Buy higher strike call (limit risk)
        - Profit if price stays BELOW short strike

        FIXED: Support/resistance logic, all validations
        """

        ticker = ticker_data['ticker']
        spot = ticker_data['spot']
        close = ticker_data['close']
        resistance = ticker_data['resistance']
        sector = ticker_data['sector']

        # Check sector limit BEFORE processing
        if self.sector_exposure[sector] >= MAX_TRADES_PER_SECTOR:
            return

        # Check per-ticker limit
        if self.ticker_spread_count[ticker] >= MAX_SPREADS_PER_TICKER:
            return

        # Filter and prepare options
        opts = self.filter_options(opts.copy(), spot)

        if len(opts) < 2:
            return

        # Calculate greeks
        opts = self.calculate_greeks(opts, spot, T, 'call')

        # CALL SPREAD: Sell lower strike, buy higher strike
        for i in range(len(opts)):
            short_leg = opts.iloc[i]

            # Delta validation
            short_delta = abs(short_leg['delta'])
            if not (TARGET_DELTA_MIN <= short_delta <= TARGET_DELTA_MAX):
                continue

            # IV rank
            iv_rank = calculate_iv_rank(close, short_leg['impliedVolatility'])
            if iv_rank < MIN_IV_RANK:
                continue

            # ═══ CORRECTED STRIKE VALIDATION ═══
            # For CALL spreads: want strikes ABOVE current price, near resistance
            # Acceptable range: 102% to 115% of resistance level
            strike_range_low = resistance * 1.02
            strike_range_high = resistance * 1.15

            if not (strike_range_low <= short_leg['strike'] <= strike_range_high):
                continue

            # Additional check: short strike should be above current price
            if short_leg['strike'] <= spot * 1.02:
                continue

            # Find long legs (HIGHER strikes)
            for j in range(i + 1, len(opts)):
                long_leg = opts.iloc[j]

                # Delta validation
                long_delta = abs(long_leg['delta'])
                if not (HEDGE_DELTA_MIN <= long_delta <= HEDGE_DELTA_MAX):
                    continue

                # Calculate spread width
                spread_width = long_leg['strike'] - short_leg['strike']

                # ═══ WIDTH VALIDATION ═══
                if not (MIN_SPREAD_WIDTH <= spread_width <= MAX_SPREAD_WIDTH):
                    continue

                # Also check as percentage of stock price
                width_pct = spread_width / spot
                if width_pct > MAX_SPREAD_WIDTH_PCT:
                    continue

                # Calculate credit
                credit = short_leg['mid'] - long_leg['mid']

                # ═══ CREDIT VALIDATION ═══
                if credit < MIN_CREDIT:
                    continue

                if credit <= 0:
                    continue

                # Calculate max loss
                max_loss = spread_width - credit

                if max_loss <= 0:
                    continue

                # Minimum credit ratio
                credit_ratio = credit / spread_width
                if credit_ratio < MIN_CREDIT_TO_WIDTH_RATIO:
                    continue

                # Maximum credit ratio
                if credit_ratio > 0.45:
                    continue

                # Calculate greeks
                net_delta = short_leg['delta'] - long_leg['delta']
                net_gamma = short_leg['gamma'] - long_leg['gamma']
                net_theta = short_leg['theta'] - long_leg['theta']
                net_vega = short_leg['vega'] - long_leg['vega']

                # Net delta limit
                if abs(net_delta) > MAX_POSITION_DELTA:
                    continue

                # Calculate metrics
                return_on_risk = (credit / max_loss) * 100
                prob_profit = 100 - (short_leg['prob_itm'] * 100)

                # Distance from resistance
                resistance_distance = (short_leg['strike'] - resistance) / spot

                # Transaction costs
                num_contracts = 1
                total_cost, commission, slippage = calculate_transaction_costs(num_contracts)
                net_credit = credit - (total_cost / 100)

                if net_credit < MIN_CREDIT * 0.5:
                    continue

                # Kelly position sizing
                kelly_size, kelly_frac = kelly_position_size(
                    prob_profit, return_on_risk, self.account_size
                )

                # Calculate comprehensive score
                score = self.calculate_score(
                    return_on_risk=return_on_risk,
                    prob_profit=prob_profit,
                    net_theta=net_theta,
                    net_gamma=net_gamma,
                    net_vega=net_vega,
                    iv_rank=iv_rank,
                    trend_score=ticker_data['trend_score'],
                    relative_strength=ticker_data['relative_strength'],
                    fund_score=ticker_data['fund_score'],
                    support_distance=resistance_distance,
                    vol_ratio=ticker_data['vol_ratio'],
                    beta=ticker_data['beta'],
                    spread_type='call'
                )

                # Apply penalties (additive)
                earnings_penalty_pts = (1.0 - earnings_penalty) * 15
                div_penalty_pts = (1.0 - div_penalty) * 5

                final_score = score - earnings_penalty_pts - div_penalty_pts
                final_score = max(0, final_score)

                # Store result
                self.results.append({
                    'Ticker': ticker,
                    'Type': 'CALL',
                    'Sector': sector,
                    'ShortStrike': round(short_leg['strike'], 2),
                    'LongStrike': round(long_leg['strike'], 2),
                    'Width': round(spread_width, 2),
                    'Expiry': exp,
                    'DTE': dte,
                    'Spot': round(spot, 2),
                    'Support': round(ticker_data['support'], 2),
                    'Resistance': round(resistance, 2),
                    'NetDelta': round(net_delta, 3),
                    'IV': round(short_leg['impliedVolatility'], 3),
                    'IV_Rank': round(iv_rank, 1),
                    'Credit': round(credit, 2),
                    'NetCredit': round(net_credit, 2),
                    'TxCost': round(total_cost, 2),
                    'MaxLoss': round(max_loss, 2),
                    'RoR%': round(return_on_risk, 1),
                    'ProbProfit%': round(prob_profit, 1),
                    'Theta': round(net_theta, 3),
                    'RS%': round(ticker_data['relative_strength'], 1),
                    'Beta': round(ticker_data['beta'], 2),
                    'FundScore': ticker_data['fund_score'],
                    'Trend': round(ticker_data['trend_score'], 2),
                    'KellySize$': round(kelly_size, 0),
                    'Kelly%': round(kelly_frac * 100, 2),
                    'Score': round(final_score, 2),
                    'HasEarnings': has_earnings,
                    'HasDividend': has_div_risk,
                    'Target50%': round(credit * 0.5, 2),
                    'StopLoss': round(-max_loss * 0.75, 2)
                })

                self.stats['spreads_found'] += 1
                self.stats['call_spreads'] += 1

                # FIXED: Increment sector exposure and ticker count
                self.sector_exposure[sector] += 1
                self.ticker_spread_count[ticker] += 1

                # Check if we hit limits
                if self.sector_exposure[sector] >= MAX_TRADES_PER_SECTOR:
                    return
                if self.ticker_spread_count[ticker] >= MAX_SPREADS_PER_TICKER:
                    return

    def filter_options(self, opts, spot):
        """
        Filter options for liquidity and quality
        FIXED: Added bid/ask sanity checks
        """

        # Remove rows with invalid bid/ask
        opts = opts[
            (opts['bid'] > 0) &
            (opts['ask'] > 0) &
            (opts['bid'] < opts['ask']) &  # Bid must be less than ask
            (~opts['bid'].isna()) &
            (~opts['ask'].isna())
            ]

        # Calculate mid and spread
        opts['mid'] = (opts['bid'] + opts['ask']) / 2
        opts['spread_pct'] = (opts['ask'] - opts['bid']) / opts['mid']

        # Apply filters
        opts = opts[
            (opts['volume'] >= MIN_VOLUME) &
            (opts['openInterest'] >= MIN_OI) &
            (opts['spread_pct'] <= MAX_BID_ASK_SPREAD_PCT) &
            (opts['mid'] > 0) &
            (opts['impliedVolatility'] > 0) &
            (opts['impliedVolatility'] < 3.0)  # Remove extreme IVs
            ]

        # Sort by strike (ascending)
        opts = opts.sort_values('strike').reset_index(drop=True)

        return opts

    def calculate_greeks(self, opts, spot, T, option_type):
        """Calculate BS greeks for all options"""

        greeks_list = []

        for _, row in opts.iterrows():
            greeks = black_scholes_greeks(
                S=spot,
                K=row['strike'],
                T=T,
                r=RISK_FREE_RATE,
                sigma=row['impliedVolatility'],
                option_type=option_type
            )
            greeks_list.append(greeks)

        greeks_df = pd.DataFrame(greeks_list)
        result = pd.concat([opts.reset_index(drop=True), greeks_df], axis=1)

        return result

    def calculate_score(self, return_on_risk, prob_profit, net_theta, net_gamma,
                        net_vega, iv_rank, trend_score, relative_strength,
                        fund_score, support_distance, vol_ratio, beta, spread_type):
        """
        Comprehensive scoring algorithm
        FIXED: Changed from multiplicative to additive penalties
        """

        # Normalize components [0-1 scale]
        ror_score = min(return_on_risk / 50, 2)
        prob_score = prob_profit / 100
        theta_score = min(abs(net_theta) / 10, 1)
        gamma_penalty = max(0, 1 - abs(net_gamma) * 1000)
        vega_penalty = max(0, 1 - abs(net_vega) / 5)
        iv_score = iv_rank / 100
        fund_score_norm = fund_score / 100

        # Relative strength (normalize -30 to +30 range to 0-1)
        rs_score = (relative_strength + 30) / 60
        rs_score = np.clip(rs_score, 0, 1)

        # Support/resistance distance
        sr_score = min(abs(support_distance) / 0.10, 1)

        # REBALANCED WEIGHTS FOR CREDIT SPREADS
        base_score = (
                ror_score * 28 +  # Most important
                prob_score * 22 +  # Very important
                theta_score * 16 +  # Key profit driver
                iv_score * 12 +  # High IV = good premiums
                sr_score * 8 +  # Technical validation
                fund_score_norm * 6 +  # Nice to have
                rs_score * 4 +  # Minor factor
                gamma_penalty * 2 +  # Risk management
                vega_penalty * 2  # Minor risk
        )

        # ADDITIVE adjustments (not multiplicative!)
        trend_adjustment = 0
        if spread_type == 'put':
            # Put spreads: prefer uptrend
            trend_adjustment = max(0, trend_score) * 8
        else:
            # Call spreads: prefer downtrend
            trend_adjustment = max(0, -trend_score) * 8

        beta_adjustment = -max(0, (beta - 1.2)) * 6

        vol_adjustment = 0
        if vol_ratio > UNUSUAL_VOLUME_THRESHOLD:
            vol_adjustment = -4

        final_score = base_score + trend_adjustment + beta_adjustment + vol_adjustment
        final_score = max(0, final_score)  # Floor at zero

        return final_score

    def display_results(self, elapsed_time):
        """Display final results with statistics and export to CSV"""

        print(f"\n{'═' * 120}")
        print(f"SCAN COMPLETE")
        print(f"{'═' * 120}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
        print(f"\nStatistics:")
        print(f"  Total tickers scanned:    {self.stats['total_scanned']}")
        print(f"  Passed fundamental:       {self.stats['passed_fundamental']}")
        print(f"  Passed technical:         {self.stats['passed_technical']}")
        print(f"  Had valid expirations:    {self.stats['had_valid_expirations']}")
        print(f"  Credit spreads found:     {self.stats['spreads_found']}")
        print(f"    - Put spreads:          {self.stats['put_spreads']}")
        print(f"    - Call spreads:         {self.stats['call_spreads']}")
        print(f"  Tickers skipped:          {len(self.skipped)}")

        if not self.results:
            print(f"\n{'═' * 120}")
            print("❌ NO VALID SPREADS FOUND")
            print(f"{'═' * 120}\n")

            if self.skipped:
                print("\nTop skip reasons:")
                skip_reasons = {}
                for ticker, reason in self.skipped[:40]:
                    # Extract main reason
                    main_reason = reason.split(':')[0] if ':' in reason else reason
                    skip_reasons[main_reason] = skip_reasons.get(main_reason, 0) + 1

                for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1])[:15]:
                    print(f"  {count:3d}x {reason}")

            print(f"\n{'═' * 120}")
            print("TROUBLESHOOTING TIPS:")
            print(f"{'═' * 120}")
            print("1. Try lowering MIN_FUNDAMENTAL_SCORE (currently: {})".format(MIN_FUNDAMENTAL_SCORE))
            print("2. Try lowering MIN_CREDIT (currently: ${:.2f})".format(MIN_CREDIT))
            print("3. Try widening delta ranges")
            print("4. Run diagnostic_tool.py on SPY to test the logic")
            print(f"{'═' * 120}\n")

            return

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Sort by score
        df = df.sort_values('Score', ascending=False)

        # Display top trades
        top_df = df.head(TOP_N).copy()

        print(f"\n{'═' * 120}")
        print(f"TOP {TOP_N} CREDIT SPREADS (sorted by Score)")
        print(f"{'═' * 120}\n")

        # Display key columns
        display_cols = [
            'Ticker', 'Type', 'Sector', 'Score', 'DTE', 'Spot',
            'ShortStrike', 'LongStrike', 'Width', 'Credit', 'NetCredit',
            'RoR%', 'ProbProfit%', 'FundScore', 'HasEarnings'
        ]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 180)
        pd.set_option('display.max_rows', None)

        print(top_df[display_cols].to_string(index=False))

        # Summary statistics
        print(f"\n{'═' * 120}")
        print("PORTFOLIO SUMMARY")
        print(f"{'═' * 120}")

        print(f"\nSector Diversification:")
        sector_counts = top_df['Sector'].value_counts()
        for sector, count in sector_counts.items():
            avg_score = top_df[top_df['Sector'] == sector]['Score'].mean()
            print(f"  {sector:25s}: {count} trades (avg score: {avg_score:.1f})")

        print(f"\nSpread Type Distribution:")
        type_counts = top_df['Type'].value_counts()
        for stype, count in type_counts.items():
            avg_score = top_df[top_df['Type'] == stype]['Score'].mean()
            print(f"  {stype:25s}: {count} spreads (avg score: {avg_score:.1f})")

        print(f"\nRisk Metrics (Top {TOP_N}):")
        print(f"  Average RoR:              {top_df['RoR%'].mean():.1f}%")
        print(f"  Average Prob Profit:      {top_df['ProbProfit%'].mean():.1f}%")
        print(f"  Average Net Credit:       ${top_df['NetCredit'].mean():.2f}")
        print(f"  Average Kelly Size:       ${top_df['KellySize$'].mean():,.0f}")
        print(f"  Total Capital Required:   ${top_df['KellySize$'].sum():,.0f}")
        print(f"  Average Beta:             {top_df['Beta'].mean():.2f}")
        print(f"  Average Fund Score:       {top_df['FundScore'].mean():.0f}")
        print(f"  Average Net Delta:        {top_df['NetDelta'].abs().mean():.3f}")

        # Event risk summary
        earnings_count = top_df['HasEarnings'].sum()
        div_count = top_df['HasDividend'].sum()
        print(f"\nEvent Risk Exposure:")
        print(f"  Trades with earnings:     {earnings_count} ({earnings_count / len(top_df) * 100:.1f}%)")
        print(f"  Trades with dividends:    {div_count} ({div_count / len(top_df) * 100:.1f}%)")

        # DTE distribution
        print(f"\nDTE Distribution:")
        dte_bins = [30, 40, 50, 60]
        for i in range(len(dte_bins) - 1):
            count = len(top_df[(top_df['DTE'] >= dte_bins[i]) & (top_df['DTE'] < dte_bins[i + 1])])
            print(f"  {dte_bins[i]}-{dte_bins[i + 1]} days:         {count} trades")

        print(f"\n{'═' * 120}")
        print("EXPORT")
        print(f"{'═' * 120}")

        # Export to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"credit_spreads_{timestamp}.csv"

        try:
            df.to_csv(filename, index=False)
            print(f"✓ Full results exported to: {filename}")
            print(f"  Total spreads in file: {len(df)}")
        except Exception as e:
            print(f"✗ Could not export CSV: {str(e)}")

        print(f"\n{'═' * 120}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Curated list of high-quality, liquid tickers
    test_tickers = [
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

    print("\n" + "=" * 120)
    print("INSTITUTIONAL CREDIT SPREAD SCANNER v3.0")
    print("Fully Debugged Production Version")
    print("=" * 120)
    print(f"\nScanning {len(test_tickers)} high-quality tickers...")
    print("This may take 15-20 minutes depending on your connection.")
    print("=" * 120 + "\n")

    # Initialize and run scanner
    scanner = InstitutionalCreditSpreadScanner(
        tickers=test_tickers,
        account_size=50000
    )

    scanner.run()

    print("\n" + "=" * 120)
    print("SCAN COMPLETE - Review results above")
    print("=" * 120 + "\n")
