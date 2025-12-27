import argparse
import json
import math
import multiprocessing
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

START_CAPITAL = 1000.0
BARS_PER_DAY = 24.0
TRADING_DAYS_PER_YEAR = 252.0
BARS_PER_YEAR = int(BARS_PER_DAY * TRADING_DAYS_PER_YEAR)
MIN_ATR = 1e-5
MIN_STOP_PCT = 1e-5
MAX_UNITS_HARD_CAP = 1e9
MAX_TRADES_ALLOWED = int(5e6)
MC_TRIALS = 300
MC_NOISE_BASE = 0.0008
PARALLEL_JOBS = max(1, multiprocessing.cpu_count() - 1)
DEFAULT_FOLDS = 5
ITER_PER_FOLD = 400

def to_native(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return


def params_to_native(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: to_native(v) for k, v in d.items()}


def safe_json(obj):
    if isinstance(obj, dict):
        return {safe_json(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

class IndicatorCache:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cache = {}
        self.open = df['open'].values.astype(float)
        self.high = df['high'].values.astype(float)
        self.low = df['low'].values.astype(float)
        self.close = df['close'].values.astype(float)
        self.n = len(df)

    def ema(self, period: int):
        key = ('ema', period)
        if key in self.cache:
            return self.cache[key]
        series = pd.Series(self.close).ewm(span=period, adjust=False).mean().values
        self.cache[key] = series
        return series

    def atr(self, period: int = 14):
        key = ('atr', period)
        if key in self.cache:
            return self.cache[key]
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        a = tr.ewm(alpha=1/period, adjust=False).mean()
        arr = a.bfill().fillna(MIN_ATR).clip(lower=MIN_ATR).values.astype(float)
        self.cache[key] = arr
        return arr

    def wilder_rsi(self, period: int = 14):
        key = ('rsi', period)
        if key in self.cache:
            return self.cache[key]
        close = pd.Series(self.close)
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/period, adjust=False).mean().replace(0, MIN_ATR)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        arr = rsi.fillna(50).values.astype(float)
        self.cache[key] = arr
        return arr

    def adx(self, period: int = 14):
        key = ('adx', period)
        if key in self.cache:
            return self.cache[key]
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.close)
        up = high.diff()
        down = -low.diff()
        plus_dm = ((up > down) & (up > 0)) * up
        minus_dm = ((down > up) & (down > 0)) * down
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_ = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, MIN_ATR)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_ + MIN_ATR))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_ + MIN_ATR))
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + MIN_ATR)).fillna(0)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        arr = adx.fillna(0).values.astype(float)
        self.cache[key] = arr
        return arr

def detect_regime_from_cache(icache: IndicatorCache, lookback: int = 252,
                             atr_period: int = 14, adx_period: int = 14) -> np.ndarray:
    adx = icache.adx(adx_period)
    at = icache.atr(atr_period)
    n = len(adx)

    at_series = pd.Series(at)
    vol_q = at_series.rolling(lookback, min_periods=1).quantile(0.5).fillna(at_series).values

    regime = np.zeros(n, dtype=int)
    high_vol_mask = at > vol_q * 1.25
    low_vol_mask = at < vol_q * 0.75
    trending_mask = adx > 25
    range_mask = adx <= 25

    regime[np.logical_and(range_mask, low_vol_mask)] = 0
    regime[np.logical_and(range_mask, high_vol_mask)] = 2
    regime[np.logical_and(trending_mask, high_vol_mask)] = 1
    regime[np.logical_and(trending_mask, ~high_vol_mask)] = 1

    return regime

def calc_spread_slippage(curr_price: float, curr_atr: float, units: float, equity: float,
                         base_spread: float, base_slippage: float,
                         impact_coef: float = 0.12, vol_coef: float = 0.7,
                         rand_scale: float = 0.5, mc_noise: float = 0.0) -> Tuple[float, float]:

    rel_vol = curr_atr / max(curr_price, 1e-9)
    notional = units * curr_price

    spread = base_spread + vol_coef * rel_vol
    spread *= (1 + mc_noise * (np.random.rand() - 0.5))
    spread = max(0.0, spread)

    impact = impact_coef * (notional / (equity + 1e-9))
    vol_term = vol_coef * rel_vol
    slippage = base_slippage + impact + vol_term
    slippage *= (1 + rand_scale * (np.random.rand() - 0.5) +
                 mc_noise * (np.random.rand() - 0.5))
    slippage = max(0.0, slippage)

    return spread, slippage

def simulate_strategy_fast(
    icache: IndicatorCache,
    params: Dict[str, Any],
    start_capital: float = START_CAPITAL,
    bars_per_year: int = BARS_PER_YEAR,
    mc_noise: float = 0.0,
    return_trades: bool = False
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[pd.DataFrame]]:

    close = icache.close
    high = icache.high
    low = icache.low
    openp = icache.open
    n = icache.n

    EMA_FAST = int(params.get('EMA_FAST', 20))
    EMA_SLOW = int(params.get('EMA_SLOW', 100))
    RSI_PERIOD = int(params.get('RSI_PERIOD', 14))
    RSI_LO = float(params.get('RSI_LO', 30))
    RSI_HI = float(params.get('RSI_HI', 70))
    ATR_PERIOD = int(params.get('ATR_PERIOD', 14))
    ATR_SL = float(params.get('ATR_SL', 3.0))
    ATR_TP = float(params.get('ATR_TP', 2.0))
    RISK_PER_TRADE = float(params.get('RISK_PER_TRADE', 0.01))
    LEV_MAX = float(params.get('LEV_MAX', 2.0))
    ADX_PERIOD = int(params.get('ADX_PERIOD', 14))
    TIME_STOP_MAX = int(params.get('TIME_STOP_MAX', 100))
    BASE_SPREAD = float(params.get('SPREAD', 0.0))
    BASE_SLIPPAGE = float(params.get('SLIPPAGE', 0.0))
    PYRAMID = int(params.get('PYRAMID', 0))
    MAX_PYRAMID = int(params.get('MAX_PYRAMID', 2))
    COOLDOWN = int(params.get('COOLDOWN', 1))

    ATR_SL = max(ATR_SL, 0.5)
    ATR_TP = max(ATR_TP, 0.5)
    RISK_PER_TRADE = max(RISK_PER_TRADE, 1e-6)
    LEV_MAX = max(LEV_MAX, 0.1)

    ema_f = icache.ema(EMA_FAST)
    ema_s = icache.ema(EMA_SLOW)
    rsi = icache.wilder_rsi(RSI_PERIOD)
    atr_series = icache.atr(ATR_PERIOD)
    adx_ = icache.adx(ADX_PERIOD)
    regime_series = detect_regime_from_cache(icache, lookback=252,
                                             atr_period=ATR_PERIOD,
                                             adx_period=ADX_PERIOD)

    equity = float(start_capital)
    equity_curve = [equity]
    positions: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    cooldown = 0
    start_idx = max(EMA_SLOW, ATR_PERIOD, RSI_PERIOD, ADX_PERIOD) + 1
    total_trades = 0
    base_atr_sl = ATR_SL
    base_atr_tp = ATR_TP

    for i in range(start_idx, n):
        if total_trades > MAX_TRADES_ALLOWED:
            break

        j_open = float(openp[i])
        j_high = float(high[i])
        j_low = float(low[i])
        j_close = float(close[i])

        if mc_noise > 0:
            noise_scale = mc_noise * (atr_series[i] if not np.isnan(atr_series[i]) else MIN_ATR)
            j_open *= (1.0 + np.random.uniform(-noise_scale, noise_scale) / max(j_open, 1e-9))
            j_high *= (1.0 + np.random.uniform(-noise_scale, noise_scale) / max(j_high, 1e-9))
            j_low *= (1.0 + np.random.uniform(-noise_scale, noise_scale) / max(j_low, 1e-9))
            j_close *= (1.0 + np.random.uniform(-noise_scale, noise_scale) / max(j_close, 1e-9))

            j_high = max(j_high, j_open, j_close)
            j_low = min(j_low, j_open, j_close)

        curr_price = float(j_close)
        curr_high = float(j_high)
        curr_low = float(j_low)
        curr_open = float(j_open)
        curr_atr = float(max(atr_series[i], MIN_ATR))
        curr_adx = float(adx_[i]) if not np.isnan(adx_[i]) else 0.0
        curr_regime = int(regime_series[i]) if i < len(regime_series) else 0

        if curr_regime == 1:
            mult_sl = min(3.0, 1.0 + base_atr_sl * 0.1)
            mult_tp = min(4.0, 1.0 + base_atr_tp * 0.2)
        elif curr_regime == 2:
            mult_sl = max(0.8, 1.0 + base_atr_sl * 0.05)
            mult_tp = max(0.8, 1.0 + base_atr_tp * 0.05)
        else:
            mult_sl = 1.0
            mult_tp = 1.0

        adaptive_sl = base_atr_sl * mult_sl
        adaptive_tp = base_atr_tp * mult_tp

        new_positions: List[Dict[str, Any]] = []
        for pos in positions:
            direction = int(pos['direction'])
            units = float(pos['units'])
            entry = float(pos['entry'])
            sl = float(pos['sl'])
            tp = float(pos['tp'])
            opened_at = int(pos['opened_at'])
            peak_unreal = float(pos.get('peak_unreal', -np.inf))

            spread_exit, slippage_exit = calc_spread_slippage(
                curr_price, curr_atr, units, equity,
                BASE_SPREAD, BASE_SLIPPAGE, mc_noise=mc_noise
            )

            hit = None
            if curr_low <= sl and curr_high >= tp:
                hit = 'stop' if abs(curr_open - sl) <= abs(curr_open - tp) else 'target'
            elif curr_low <= sl:
                hit = 'stop'
            elif curr_high >= tp:
                hit = 'target'

            if hit == 'stop':
                exit_price = sl - slippage_exit if direction == 1 else sl + slippage_exit
                pnl = direction * units * (exit_price - entry)
                equity += pnl
                trades.append({
                    **pos, 'exit': exit_price, 'exit_index': i,
                    'pnl': float(pnl), 'exit_reason': 'stop'
                })
                total_trades += 1
                continue

            if hit == 'target':
                exit_price = tp - slippage_exit if direction == 1 else tp + slippage_exit
                pnl = direction * units * (exit_price - entry)
                equity += pnl
                trades.append({
                    **pos, 'exit': exit_price, 'exit_index': i,
                    'pnl': float(pnl), 'exit_reason': 'tp'
                })
                total_trades += 1
                continue

            if (i - opened_at) >= TIME_STOP_MAX:
                exit_price = curr_price - slippage_exit if direction == 1 else curr_price + slippage_exit
                pnl = direction * units * (exit_price - entry)
                equity += pnl
                trades.append({
                    **pos, 'exit': exit_price, 'exit_index': i,
                    'pnl': float(pnl), 'exit_reason': 'time_stop'
                })
                total_trades += 1
                continue

            run = (curr_price - entry) * direction
            one_r = curr_atr * adaptive_tp
            if run >= one_r:
                if direction == 1:
                    sl = max(sl, entry)
                else:
                    sl = min(sl, entry)
                trailing_mult = 0.75
                if direction == 1:
                    sl = max(sl, curr_price - curr_atr * adaptive_sl * trailing_mult)
                else:
                    sl = min(sl, curr_price + curr_atr * adaptive_sl * trailing_mult)
                pos['sl'] = float(sl)

            pos['peak_unreal'] = float(max(peak_unreal, direction * (curr_price - entry) * units))
            new_positions.append(pos)

        positions = new_positions

        trend_up = ema_f[i - 1] > ema_s[i - 1]
        trend_dn = ema_f[i - 1] < ema_s[i - 1]
        strong_trend = curr_adx > 20

        local_atr_median = float(np.median(atr_series[max(0, i - 50):i + 1]))
        vol_factor = min(2.0, max(0.5, curr_atr / (local_atr_median + 1e-12)))

        rsi_val = rsi[i - 1]
        adj_rsi_lo = RSI_LO * (1.0 if vol_factor < 1.2 else 0.9)
        adj_rsi_hi = RSI_HI * (1.0 if vol_factor < 1.2 else 1.05)

        long_moment = trend_up and strong_trend and (rsi_val < adj_rsi_lo)
        short_moment = trend_dn and strong_trend and (rsi_val > adj_rsi_hi)
        long_revert = trend_up and (rsi_val < adj_rsi_lo)
        short_revert = trend_dn and (rsi_val > adj_rsi_hi)

        signal = 0
        if long_moment:
            signal = 1
        elif short_moment:
            signal = -1
        else:
            if long_revert:
                signal = 1
            elif short_revert:
                signal = -1

        if cooldown > 0:
            cooldown -= 1
            signal = 0

        if signal != 0:
            stop_distance = max(curr_atr * adaptive_sl, curr_price * MIN_STOP_PCT)
            risk_amount = equity * RISK_PER_TRADE
            naive_units = risk_amount / (stop_distance + 1e-12)

            spread_entry, slippage_entry = calc_spread_slippage(
                curr_price, curr_atr, naive_units, equity,
                BASE_SPREAD, BASE_SLIPPAGE, mc_noise=mc_noise
            )
            half_spread = spread_entry / 2.0

            if signal == 1:
                entry_price = curr_price + half_spread + slippage_entry
                sl_price = entry_price - stop_distance
                tp_price = entry_price + curr_atr * adaptive_tp
            else:
                entry_price = curr_price - half_spread - slippage_entry
                sl_price = entry_price + stop_distance
                tp_price = entry_price - curr_atr * adaptive_tp

            if abs(entry_price - sl_price) < max(curr_price * MIN_STOP_PCT, MIN_ATR * 1e-3):
                adjust = max(curr_price * (MIN_STOP_PCT * 10), MIN_ATR * 1e-2)
                if signal == 1:
                    sl_price = entry_price - adjust
                else:
                    sl_price = entry_price + adjust

            max_units_by_lev = (equity * LEV_MAX) / max(curr_price, 1e-12)
            absolute_units_cap = min(max_units_by_lev, MAX_UNITS_HARD_CAP)
            units = min(naive_units, absolute_units_cap)
            units = max(units, 0.0)
            if units < 1e-9:
                units = 0.0

            curr_pyramids = sum(1 for p in positions if int(p['direction']) == signal)
            allow_pyramid = (PYRAMID and (curr_pyramids < MAX_PYRAMID))

            if curr_pyramids == 0 or allow_pyramid:
                if units > 0:
                    pos = {
                        'direction': int(signal),
                        'units': float(units),
                        'entry': float(entry_price),
                        'sl': float(sl_price),
                        'tp': float(tp_price),
                        'opened_at': int(i),
                        'peak_unreal': float(-np.inf)
                    }
                    positions.append(pos)
                    cooldown = COOLDOWN

        equity_curve.append(float(equity))

    equity_arr = np.array(equity_curve, dtype=float)
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['pnl'] = trades_df['pnl'].astype(float)

    returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-12)
    if len(returns) > 1:
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret / (std_ret + 1e-12)) * math.sqrt(bars_per_year)
        downside = returns[returns < 0]
        sortino = (
            mean_ret / (downside.std() + 1e-12) * math.sqrt(bars_per_year)
            if len(downside) > 0 else None
        )
    else:
        sharpe, sortino = 0.0, None

    equity_series = pd.Series(equity_arr)
    peak = equity_series.cummax()
    drawdown_pct = float(((peak - equity_series) / (peak + 1e-12)).max()) if len(equity_series) > 0 else 0.0
    drawdown_abs = float((peak - equity_series).max()) if len(equity_series) > 0 else 0.0

    metrics = {
        'start_capital': float(start_capital),
        'end_capital': float(equity),
        'total_pnl': float(equity - start_capital),
        'n_trades': int(len(trades_df)),
        'sharpe': float(sharpe),
        'sortino': float(sortino) if sortino is not None else None,
        'max_drawdown_pct': float(drawdown_pct),
        'max_drawdown_abs': float(drawdown_abs)
    }

    if return_trades:
        return metrics, equity_series, trades_df, {
            'positions_end': positions,
            'regime_last': int(curr_regime)
        }
    return metrics, equity_series, trades_df

def simple_metrics_agg(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    cols = set()
    for m in metrics_list:
        cols.update(m.keys())
    out = {}
    for c in cols:
        vals = []
        for m in metrics_list:
            v = m.get(c, None)
            if v is None:
                continue
            try:
                if np.isfinite(v):
                    vals.append(float(v))
            except Exception:
                pass
        out[c] = float(np.mean(vals)) if len(vals) > 0 else None
    return out


class RobustBayesianOptimizer:
    def __init__(self, df: pd.DataFrame, space, folds: int = 5, random_state: int = 42,
                 mc_for_top: int = 0, mc_parallel: bool = True):
        self.df = df.reset_index(drop=True)
        self.space = space
        self.random_state = int(random_state)
        self.folds = int(max(2, folds))
        self.eval_cache: Dict[Tuple, Dict[str, Any]] = {}
        self.icache_global = IndicatorCache(self.df)
        self.mc_for_top = int(mc_for_top)
        self.mc_parallel = bool(mc_parallel)

    def _params_key(self, params: Dict[str, Any]):
        return tuple(sorted((k, to_native(v)) for k, v in params.items()))

    def _make_folds(self):
        total = len(self.df)
        if total < 50:
            raise ValueError("Not enough data for K-fold.")
        test_len = max(10, int(total / (self.folds + 1)))
        fold_start = 0
        folds = []
        for fold in range(self.folds):
            train_end = fold_start + int(test_len * (0.6 / 0.4))
            train_end = min(train_end, total - test_len - 1)
            if train_end <= 100:
                train_end = int(total * 0.6)
            train_df = self.df.iloc[:train_end].reset_index(drop=True)
            test_start = train_end
            test_end = min(total, train_end + test_len)
            test_df = self.df.iloc[test_start:test_end].reset_index(drop=True)
            folds.append((train_df, test_df))
            fold_start += test_len
            if test_end >= total:
                break
        if len(folds) == 0:
            train_end = int(total * 0.8)
            folds = [(self.df.iloc[:train_end].reset_index(drop=True),
                      self.df.iloc[train_end:].reset_index(drop=True))]
        return folds

    def evaluate_single(self, df_segment: pd.DataFrame, params: Dict[str, Any], mc_noise: float = 0.0) -> Dict[str, Any]:
        key = (self._params_key(params), len(df_segment))
        if key in self.eval_cache:
            return deepcopy(self.eval_cache[key])

        icache = IndicatorCache(df_segment)
        try:
            metrics, _, _ = simulate_strategy_fast(icache, params, mc_noise=mc_noise)
        except Exception as e:
            metrics = {
                'sharpe': float(-999.0),
                'sortino': float(-999.0),
                'max_drawdown_pct': 1.0,
                'total_pnl': float(-1e9),
                'end_capital': 0.0,
                'n_trades': 0
            }
        if 'returns_var' not in metrics:
            metrics['returns_var'] = metrics.get('returns_var', None)

        self.eval_cache[key] = deepcopy(metrics)
        return deepcopy(metrics)

    def evaluate_kfold(self, params: Dict[str, Any]) -> Dict[str, Any]:
        folds = self._make_folds()
        metrics_test_list = []
        metrics_train_list = []
        worst_drawdown = 0.0
        any_zero_trades = False
        any_catastrophic = False

        for (train_df, test_df) in folds:
            m_train = self.evaluate_single(train_df, params, mc_noise=0.0)
            metrics_train_list.append(m_train)
            m_test = self.evaluate_single(test_df, params, mc_noise=0.0)
            metrics_test_list.append(m_test)

            dd = float(m_test.get('max_drawdown_pct', 1.0) or 1.0)
            worst_drawdown = max(worst_drawdown, dd)
            if int(m_test.get('n_trades', 0)) < 2:
                any_zero_trades = True
                endcap = float(m_test.get('end_capital', 0.0) or 0.0)
            if dd > 0.9 or endcap < 0.1:
                any_catastrophic = True
        agg_test = simple_metrics_agg(metrics_test_list)
        agg_train = simple_metrics_agg(metrics_train_list)
        agg_test['worst_drawdown'] = worst_drawdown
        agg_test['any_zero_trades'] = any_zero_trades
        agg_test['any_catastrophic'] = any_catastrophic
        agg_test['_folds_test'] = metrics_test_list
        agg_test['_folds_train'] = metrics_train_list
        return agg_test

    def robust_objective_from_agg(self, agg: Dict[str, Any]) -> float:
        sharpe = float(agg.get('sharpe', 0.0) or 0.0)
        sortino = float(agg.get('sortino', sharpe) or sharpe)
        dd = float(agg.get('max_drawdown_pct', 1.0) or 1.0)
        pnl = float(agg.get('total_pnl', 0.0) or 0.0)
        ntrades = float(agg.get('n_trades', 0) or 0)
        if agg.get('any_catastrophic', False):
            return 1e6
        if agg.get('any_zero_trades', False) or ntrades < 3:
            return 9e5
        reward_sharpe = -1.0 * sharpe
        dd_penalty = 4.0 * dd
        trade_penalty = 0.0004 * max(0.0, ntrades - 500.0)
        stability_penalty = abs(sharpe - sortino)
        pnl_reward = -0.00015 * pnl
        worst_drawdown = float(agg.get('worst_drawdown', 0.0) or 0.0)
        worst_penalty = 10.0 * max(0.0, worst_drawdown - 0.5)

        obj = (
            reward_sharpe +
            dd_penalty +
            trade_penalty +
            stability_penalty +
            pnl_reward +
            worst_penalty
        )
        if not np.isfinite(obj):
            obj = 1e6

        return float(obj)

    def skopt_objective(self, x_list):
        param_names = [d.name for d in self.space]
        params_raw = dict(zip(param_names, x_list))
        params = {
            'EMA_FAST': int(params_raw['EMA_FAST']),
            'EMA_SLOW': int(params_raw['EMA_SLOW']),
            'RSI_PERIOD': int(params_raw['RSI_PERIOD']),
            'RSI_LO': float(params_raw['RSI_LO']),
            'RSI_HI': float(params_raw['RSI_HI']),
            'ATR_PERIOD': int(params_raw['ATR_PERIOD']),
            'ATR_SL': float(params_raw['ATR_SL']),
            'ATR_TP': float(params_raw['ATR_TP']),
            'RISK_PER_TRADE': float(params_raw['RISK_PER_TRADE']),
            'LEV_MAX': float(params_raw['LEV_MAX']),
            'ADX_PERIOD': int(params_raw['ADX_PERIOD']),
            'TIME_STOP_MAX': int(params_raw['TIME_STOP_MAX']),
            'SPREAD': float(params_raw['SPREAD']),
            'SLIPPAGE': float(params_raw['SLIPPAGE']),
            'PYRAMID': int(params_raw.get('PYRAMID', 0)),
            'MAX_PYRAMID': int(params_raw.get('MAX_PYRAMID', 2)),
            'COOLDOWN': int(params_raw.get('COOLDOWN', 1))
        }

        if int(params['EMA_FAST']) >= int(params['EMA_SLOW']):
            return 9e5

        agg_test = self.evaluate_kfold(params)
        obj = self.robust_objective_from_agg(agg_test)
        return float(obj)

    def run(self, n_calls: int = 500, n_initial_points: int = 50):
        print(f"Running Robust Bayesian optimization (gp_minimize) n_calls={n_calls}, folds={self.folds}")
        res = gp_minimize(
            func=self.skopt_objective,
            dimensions=self.space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=self.random_state,
            verbose=True
        )

        param_names = [d.name for d in self.space]
        best_vals = res.x
        best_params_raw = dict(zip(param_names, best_vals))
        best_params = {
            'EMA_FAST': int(best_params_raw['EMA_FAST']),
            'EMA_SLOW': int(best_params_raw['EMA_SLOW']),
            'RSI_PERIOD': int(best_params_raw['RSI_PERIOD']),
            'RSI_LO': float(best_params_raw['RSI_LO']),
            'RSI_HI': float(best_params_raw['RSI_HI']),
            'ATR_PERIOD': int(best_params_raw['ATR_PERIOD']),
            'ATR_SL': float(best_params_raw['ATR_SL']),
            'ATR_TP': float(best_params_raw['ATR_TP']),
            'RISK_PER_TRADE': float(best_params_raw['RISK_PER_TRADE']),
            'LEV_MAX': float(best_params_raw['LEV_MAX']),
            'ADX_PERIOD': int(best_params_raw['ADX_PERIOD']),
            'TIME_STOP_MAX': int(best_params_raw['TIME_STOP_MAX']),
            'SPREAD': float(best_params_raw['SPREAD']),
            'SLIPPAGE': float(best_params_raw['SLIPPAGE']),
            'PYRAMID': int(best_params_raw.get('PYRAMID', 0)),
            'MAX_PYRAMID': int(best_params_raw.get('MAX_PYRAMID', 2)),
            'COOLDOWN': int(best_params_raw.get('COOLDOWN', 1))
        }

        final_agg = self.evaluate_kfold(best_params)
        out = {
            'result': res,
            'best_params': best_params,
            'best_agg_test_metrics': final_agg,
            'eval_cache': self.eval_cache
        }

        if self.mc_for_top and self.mc_for_top > 0:
            print(f"Running Monte Carlo sanity for top candidate (trials={self.mc_for_top}) ...")
            mc_report = monte_carlo(self.df, best_params, trials=self.mc_for_top, parallel=self.mc_parallel)
            out['mc_top'] = mc_report

        return out

def _mc_worker(args):
    df_test, params, mc_noise = args
    icache = IndicatorCache(df_test)
    return simulate_strategy_fast(icache, params, mc_noise=mc_noise)[0]


def monte_carlo(df_test: pd.DataFrame, params: Dict[str, Any],
                trials: int = MC_TRIALS, parallel: bool = True):
    inputs = []
    for _ in range(trials):
        mc_noise = MC_NOISE_BASE * np.random.uniform(0.6, 1.6)
        inputs.append((df_test, params, mc_noise))

    if parallel and JOBLIB_AVAILABLE:
        results = Parallel(n_jobs=PARALLEL_JOBS)(
            delayed(_mc_worker)(inp) for inp in inputs
        )
    else:
        results = [_mc_worker(inp) for inp in inputs]

    dfm = pd.DataFrame(results)
    summary = {}

    for col in ['total_pnl', 'end_capital', 'sharpe', 'max_drawdown_pct']:
        if col in dfm.columns:
            vals = dfm[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                summary[col] = None
            else:
                summary[col] = {
                    'mean': float(vals.mean()),
                    'median': float(vals.median()),
                    'std': float(vals.std()),
                    'p05': float(np.percentile(vals, 5)),
                    'p25': float(np.percentile(vals, 25)),
                    'p75': float(np.percentile(vals, 75)),
                    'p95': float(np.percentile(vals, 95))
                }
        else:
            summary[col] = None

    return {'summary': summary, 'raw': dfm}

def parameter_sensitivity(base_params: Dict[str, Any], df_test: pd.DataFrame,
                          frac: float = 0.25, steps: int = 7):
    icache = IndicatorCache(df_test)
    sens = {}

    for k, v in base_params.items():
        if k in ['PYRAMID', 'MAX_PYRAMID', 'COOLDOWN']:
            continue

        if isinstance(v, int):
            rng = max(1, int(abs(v) * frac))
            values = sorted(set(max(1, v - rng) + i for i in range(steps)))
        elif isinstance(v, float):
            rng = max(abs(v) * frac, 0.01)
            values = [
                float(max(0.0, v - rng + 2 * rng * i / (steps - 1)))
                for i in range(steps)
            ]
        else:
            continue

        recs = []
        for val in values:
            p = deepcopy(base_params)
            p[k] = int(val) if isinstance(v, int) else float(val)
            metrics, _, _ = simulate_strategy_fast(icache, p, mc_noise=0.0)
            sortino = metrics.get('sortino') or metrics['sharpe']
            scr = (
                    -metrics['sharpe'] +
                    4 * metrics['max_drawdown_pct'] +
                    0.0004 * max(0, metrics['n_trades'] - 500) +
                    abs(metrics['sharpe'] - sortino) -
                    0.00015 * metrics['total_pnl']
            )


            recs.append((to_native(val), scr))

        sens[k] = recs

    return sens

def run_full_pipeline(file_path: str, folds: int = DEFAULT_FOLDS,
                      n_calls: int = 500, train_frac: float = 0.8,
                      random_state: int = 42, mc_trials: int = 300,
                      parallel_mc: bool = True):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy().reset_index(drop=True)
    total = len(df)
    if total < 200:
        raise ValueError("Not enough data for rolling walk-forward. Need more bars.")
    space = [
        Integer(10, 60, name='EMA_FAST'),
        Integer(30, 200, name='EMA_SLOW'),
        Integer(6, 25, name='RSI_PERIOD'),
        Real(5.0, 30.0, name='RSI_LO'),
        Real(50.0, 80.0, name='RSI_HI'),
        Integer(5, 40, name='ATR_PERIOD'),
        Real(0.8, 8.0, name='ATR_SL'),
        Real(0.8, 12.0, name='ATR_TP'),
        Real(0.001, 0.03, name='RISK_PER_TRADE'),
        Real(1, 10, name='LEV_MAX'),
        Integer(6, 30, name='ADX_PERIOD'),
        Integer(10, 200, name='TIME_STOP_MAX'),
        Real(0.01, 0.8, name='SPREAD'),
        Real(0.005, 0.01, name='SLIPPAGE'),
        Integer(0, 1, name='PYRAMID'),
        Integer(0, 4, name='MAX_PYRAMID'),
        Integer(0, 3, name='COOLDOWN'),
    ]

    results_folds = []
    test_len = max(10, int(total / (folds + 1)))
    fold_start = 0

    for fold in range(folds):
        train_end = fold_start + int(test_len * (train_frac / (1 - train_frac)))
        train_end = min(train_end, total - test_len - 1)
        if train_end <= 100:
            train_end = int(total * train_frac)

        train_df = df.iloc[:train_end].reset_index(drop=True)
        test_start = train_end
        test_end = min(total, train_end + test_len)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        print(f"[Fold {fold}] train {0}:{train_end} test {test_start}:{test_end} "
              f"(len train={len(train_df)}, test={len(test_df)})")

        bo = RobustBayesianOptimizer(train_df, space, random_state=random_state + fold)
        bo_res = bo.run(n_calls=n_calls, n_initial_points=max(10, int(n_calls * 0.1)))

        best_params = bo_res['best_params']
        best_metrics = bo_res['best_agg_test_metrics']

        icache_test = IndicatorCache(test_df)
        metrics_test, eq_test, trades_test = simulate_strategy_fast(
            icache_test, best_params, mc_noise=0.0
        )

        results_folds.append({
            'fold': fold,
            'train_range': (0, train_end),
            'test_range': (test_start, test_end),
            'best_params': best_params,
            'best_metrics_train': best_metrics,
            'metrics_test': metrics_test,
            'eq_test': eq_test,
            'trades_test': trades_test,
            'eval_cache': bo_res['eval_cache']
        })

        fold_start += test_len

    aggregated = []
    for r in results_folds:
        aggregated.append((
            r['best_metrics_train'].get('sharpe', 0.0),
            r['best_params'],
            r
        ))

    aggregated_sorted = sorted(aggregated, key=lambda x: x[0], reverse=True)
    top_candidates = aggregated_sorted[:10]
    best_overall = aggregated_sorted[0] if aggregated_sorted else None

    if best_overall is None:
        raise RuntimeError("No candidate found")

    best_sharpe, best_params, best_meta = best_overall

    total_len = len(df)
    train_end = int(total_len * (1 - 0.2))
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_test = df.iloc[train_end:].reset_index(drop=True)

    icache_train = IndicatorCache(df_train)
    metrics_train, eq_train, trades_train = simulate_strategy_fast(
        icache_train, best_params, mc_noise=0.0
    )

    icache_test = IndicatorCache(df_test)
    metrics_test, eq_test, trades_test = simulate_strategy_fast(
        icache_test, best_params, mc_noise=0.0
    )

    print(f"Running Monte Carlo (trials={mc_trials}) on final OOS (parallel={parallel_mc and JOBLIB_AVAILABLE}) ...")
    mc_report = monte_carlo(df_test, best_params,
                            trials=mc_trials,
                            parallel=parallel_mc and JOBLIB_AVAILABLE)

    print("Computing local parameter sensitivity (small grid)...")
    sens = parameter_sensitivity(best_params, df_test, frac=0.25, steps=7)

    out = {
        'best_params': params_to_native(best_params),
        'final_train_metrics': metrics_train,
        'final_test_metrics': metrics_test,
        'mc_summary': mc_report['summary'],
        'sensitivity_sample': {
            k: sens[k][:5]
            for k in list(sens.keys())[:10]
        },
        'folds_summary': [{
            'fold': r['fold'],
            'best_params': params_to_native(r['best_params']),
            'train_sharpe': r['best_metrics_train'].get('sharpe')
        } for r in results_folds],
        'top_candidates': [{
            'sharpe': to_native(x[0]),
            'params': params_to_native(x[1])
        } for x in top_candidates]
    }

    return out
def main():
    parser = argparse.ArgumentParser(description="Faster Bayesian optimizer for Forex hybrid strategy")
    parser.add_argument(
        '--file', '-f',
        type=str,
        required=False,
        default="/Users/adamsarakbi/Desktop/FOREX DATA.csv",
        help='CSV path (timestamp,open,high,low,close,volume)'
    )
    parser.add_argument('--folds', type=int, default=DEFAULT_FOLDS)
    parser.add_argument('--n_calls', type=int, default=200, help='Number of gp_minimize calls per fold')
    parser.add_argument('--train_frac', type=float, default=0.6)
    parser.add_argument('--mc', type=int, default=MC_TRIALS)
    parser.add_argument('--parallel_mc', action='store_true', help='Parallelize Monte Carlo if joblib available')
    parser.add_argument('--random_state', type=int, default=42)

    args = parser.parse_args()

    out = run_full_pipeline(
        args.file,
        folds=args.folds,
        n_calls=args.n_calls,
        train_frac=args.train_frac,
        random_state=args.random_state,
        mc_trials=args.mc,
        parallel_mc=args.parallel_mc
    )

    print("\n=== FINAL OUTPUT SUMMARY ===")
    print(json.dumps(safe_json(out), indent=2))


if __name__ == "__main__":
    main()
