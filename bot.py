import os
import hmac
import hashlib
import time
import logging
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
    CallbackQueryHandler, MessageHandler, filters
)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com/fapi/v1"
USER_KEYS: dict[int, dict] = {}

# ─── Режим торговли (demo / real) ─────────────────────────────────────────────
REAL_FUTURES_API = "https://fapi.binance.com"
DEMO_FUTURES_API = "https://demo-fapi.binance.com"
USER_MODE: dict[int, str] = {}   # chat_id -> "real" | "demo"  (default: "real")

def get_futures_api(chat_id: int) -> str:
    return DEMO_FUTURES_API if USER_MODE.get(chat_id) == "demo" else REAL_FUTURES_API

# Подписки на алерты: chat_id -> set(symbols), плюс отдельный флаг подписки на скан рынка
ALERT_SUBSCRIPTIONS: dict[int, set] = {}
MARKET_SCAN_SUBSCRIBERS: set = set()
LAST_SIGNAL_SENT: dict[tuple, str] = {}  # (chat_id, symbol) -> last signal string
ALERT_SCORE_THRESHOLD = 5.0  # |score| >= 5 считается "сильным" сигналом для алерта

# ─── Автоторговля ─────────────────────────────────────────────────────────────
AUTOTRADE_ENABLED: dict[int, bool] = {}       # chat_id -> вкл/выкл
AUTOTRADE_RISK_PCT: dict[int, float] = {}     # chat_id -> % баланса на сделку
PENDING_TRADES: dict[str, dict] = {}          # trade_id -> данные сделки (до подтверждения)
OPEN_AUTOTRADES: dict[int, list] = {}         # chat_id -> список открытых авто-позиций
AUTOTRADE_SCORE_THRESHOLD = 5.0              # порог сигнала
DEFAULT_LEVERAGE = 3                          # плечо по умолчанию
MAX_CANDIDATES_LONG  = 8                      # сколько лонг-кандидатов отдаёт сканер в авто-анализ
MAX_CANDIDATES_SHORT = 8                      # сколько шорт-кандидатов отдаёт сканер
ATR_SL_MULT      = 1.5                        # базовый множитель ATR для стоп-лосса
ATR_SL_CAP_MULT  = 2.5                        # верхняя граница SL по структуре, в ATR
ATR_SL_LOOKBACK  = 20                         # глубина поиска swing-high/low для SL по структуре
ATR_TRAIL_MULT   = 1.0                        # дистанция трейлинг-стопа после TP1, в ATR
PENDING_TRADE_TTL_SEC = 1800                  # сколько живёт неподтверждённый авто-сигнал (30 мин)

# ─── Дневной лимит убытков ────────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT = 5.0                   # максимальный дневной убыток в % от баланса
AUTOTRADE_DAILY_LOSS: dict[int, float] = {}  # chat_id -> сумма убытков за сегодня ($)
AUTOTRADE_DAILY_DATE: dict[int, str]  = {}   # chat_id -> дата последнего сброса
AUTOTRADE_PAUSED:     dict[int, bool] = {}   # chat_id -> пауза из-за дневного лимита

# ─── BTC режим (кэш) ─────────────────────────────────────────────────────────
_BTC_TREND_CACHE: dict = {"trend": None, "updated_at": 0}  # обновляем раз в 30 мин

# ─── Funding Rate пороги (только экстремальные значения) ──────────────────────
FUNDING_EXTREME_HIGH =  0.10   # % — рынок перегрет лонгами → шорт-сигнал
FUNDING_EXTREME_LOW  = -0.05   # % — рынок перегрет шортами → лонг-сигнал

# ─── BTC режим (фильтр лонгов по альтам) ─────────────────────────────────────
BTC_REGIME_CACHE: dict = {}    # {"trend": str, "time": float}
BTC_CACHE_TTL = 300            # секунд (5 мин)

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    return symbol

def fmt_price(price: float) -> str:
    if price >= 1:
        return f"${price:,.2f}"
    elif price >= 0.0001:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"

def fmt_change(change: float) -> str:
    arrow = "🟢" if change >= 0 else "🔴"
    sign = "+" if change >= 0 else ""
    return f"{arrow} {sign}{change:.2f}%"

def signed_request(method, url, api_key, api_secret, params=None):
    params = dict(params) if params else {}
    params["timestamp"] = int(time.time() * 1000)
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": api_key}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10) if method == "GET" else requests.post(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Signed request error: {e}")
        return None

def get_btc_regime() -> str:
    """Возвращает текущий тренд BTC (кэшируется на 5 мин)."""
    now = time.time()
    if BTC_REGIME_CACHE.get("time", 0) + BTC_CACHE_TTL > now:
        return BTC_REGIME_CACHE.get("trend", "neutral")
    try:
        klines = get_klines("BTCUSDT", "4h", 200)
        if not klines:
            return "neutral"
        df = klines_to_df(klines)
        close = df["close"]
        ema20 = calc_ema(close, 20).iloc[-1]
        ema50 = calc_ema(close, 50).iloc[-1]
        ema200 = calc_ema(close, 200).iloc[-1]
        p = close.iloc[-1]
        if p > ema20 > ema50 > ema200:
            trend = "strong_up"
        elif p > ema50 > ema200:
            trend = "up"
        elif p < ema20 and p < ema50 and p < ema200:
            trend = "strong_down"
        elif p < ema50:
            trend = "down"
        else:
            trend = "neutral"
        BTC_REGIME_CACHE["trend"] = trend
        BTC_REGIME_CACHE["time"]  = now
        return trend
    except Exception as e:
        logger.warning(f"get_btc_regime error: {e}")
        return "neutral"


def get_funding_signal(symbol: str) -> tuple[float, str | None]:
    """
    Возвращает (funding_rate_pct, signal).
    signal = 'long' если funding экстремально отрицательный (перегрет шортами),
             'short' если экстремально положительный (перегрет лонгами),
             None если значение обычное.
    Работает ТОЛЬКО в экстремальных ситуациях.
    """
    try:
        data = get_funding_rate(symbol)
        if not data or "lastFundingRate" not in data:
            return 0.0, None
        fr = float(data["lastFundingRate"]) * 100
        if fr >= FUNDING_EXTREME_HIGH:
            return fr, "short"   # рынок перегрет лонгами → сигнал к шорту
        elif fr <= FUNDING_EXTREME_LOW:
            return fr, "long"    # рынок перегрет шортами → сигнал к лонгу
        return fr, None
    except Exception:
        return 0.0, None


def check_daily_loss_limit(chat_id: int, balance: float) -> tuple[bool, str]:
    """
    Проверяет не превышен ли дневной лимит убытков.
    Возвращает (торговля_разрешена, сообщение).
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Сброс счётчика в новый день
    if AUTOTRADE_DAILY_DATE.get(chat_id) != today:
        AUTOTRADE_DAILY_DATE[chat_id] = today
        AUTOTRADE_DAILY_LOSS[chat_id] = 0.0
        AUTOTRADE_PAUSED[chat_id] = False

    if AUTOTRADE_PAUSED.get(chat_id, False):
        loss = AUTOTRADE_DAILY_LOSS.get(chat_id, 0.0)
        limit = balance * DAILY_LOSS_LIMIT_PCT / 100
        return False, (
            f"🛑 Автоторговля приостановлена на сегодня\n"
            f"Дневной убыток: `${loss:.2f}` / лимит `${limit:.2f}` ({DAILY_LOSS_LIMIT_PCT}%)\n"
            f"Возобновится завтра автоматически."
        )
    return True, ""


def record_trade_result(chat_id: int, pnl_usd: float, balance: float):
    """
    Записывает результат сделки. Если убыток превысил дневной лимит — ставит паузу.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if AUTOTRADE_DAILY_DATE.get(chat_id) != today:
        AUTOTRADE_DAILY_DATE[chat_id] = today
        AUTOTRADE_DAILY_LOSS[chat_id] = 0.0

    if pnl_usd < 0:
        AUTOTRADE_DAILY_LOSS[chat_id] = AUTOTRADE_DAILY_LOSS.get(chat_id, 0.0) + abs(pnl_usd)

    limit = balance * DAILY_LOSS_LIMIT_PCT / 100
    if AUTOTRADE_DAILY_LOSS.get(chat_id, 0.0) >= limit:
        AUTOTRADE_PAUSED[chat_id] = True
        logger.warning(f"chat_id={chat_id}: дневной лимит убытков достигнут, автоторговля приостановлена")


def get_ticker_24h(symbol):
    try:
        r = requests.get(f"{BINANCE_API}/ticker/24hr", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def _binance_get_with_retry(url, params=None, timeout=10, max_retries=3):
    """GET-запрос к Binance с exponential backoff на 429/418.
    - 429 (rate limit) и 418 (IP-бан) — пауза 1→2→4с и ретрай, а не молчаливый None.
    - 5xx — тоже ретрай (кратковременные сбои нод).
    - остальные ошибки/таймауты — сразу None, как раньше.
    Без этого тяжёлый сканер мог получить 429 и тихо пропустить данные → битые
    свечи → ложные сигналы."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 418) or r.status_code >= 500:
                logger.warning(f"binance GET {url} status={r.status_code} — retry через {delay}s (попытка {attempt+1}/{max_retries})")
                if r.status_code == 418:
                    # IP-бан: Binance просит длинную паузу. Уважаем.
                    delay *= 4
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"binance GET {url} network: {e} — retry через {delay}s (попытка {attempt+1}/{max_retries})")
            time.sleep(delay)
            delay *= 2
            continue
        except Exception as e:
            logger.warning(f"binance GET {url}: {e}")
            return None
    return None


def get_klines(symbol, interval="1h", limit=200):
    return _binance_get_with_retry(
        f"{BINANCE_API}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )

def get_futures_ticker(symbol):
    try:
        r = requests.get(f"{BINANCE_FUTURES_API}/ticker/24hr", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def get_open_interest(symbol):
    try:
        r = requests.get(f"{BINANCE_FUTURES_API}/openInterest", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def get_funding_rate(symbol):
    try:
        r = requests.get(f"{BINANCE_FUTURES_API}/premiumIndex", params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def get_top_movers(limit=5):
    try:
        r = requests.get(f"{BINANCE_API}/ticker/24hr", timeout=15)
        r.raise_for_status()
        data = r.json()
        usdt_pairs = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 1_000_000]
        sorted_data = sorted(usdt_pairs, key=lambda x: float(x["priceChangePercent"]), reverse=True)
        return sorted_data[:limit], sorted_data[-limit:][::-1]
    except:
        return [], []

# ─── BTC Режим рынка ──────────────────────────────────────────────────────────

def get_btc_trend() -> str | None:
    """
    Возвращает тренд BTC на 4H: strong_up / up / neutral / down / strong_down.
    Кэшируется на 30 минут чтобы не спамить API при каждом скане.
    """
    now = time.time()
    if now - _BTC_TREND_CACHE["updated_at"] < 1800 and _BTC_TREND_CACHE["trend"]:
        return _BTC_TREND_CACHE["trend"]

    try:
        klines = get_klines("BTCUSDT", "4h", 210)
        if not klines or len(klines) < 210:
            return None
        df = klines_to_df(klines)
        close = df["close"]

        ema20  = calc_ema(close, 20).iloc[-1]
        ema50  = calc_ema(close, 50).iloc[-1]
        ema200 = calc_ema(close, 200).iloc[-1]
        p = close.iloc[-1]

        if p > ema20 > ema50 > ema200:
            trend = "strong_up"
        elif p > ema50 > ema200:
            trend = "up"
        elif p < ema20 and p < ema50 and p < ema200:
            trend = "strong_down"
        elif p < ema50:
            trend = "down"
        else:
            trend = "neutral"

        _BTC_TREND_CACHE["trend"] = trend
        _BTC_TREND_CACHE["updated_at"] = now
        logger.info(f"BTC 4H тренд обновлён: {trend}")
        return trend
    except Exception as e:
        logger.warning(f"get_btc_trend error: {e}")
        return None


def btc_regime_allows(direction: str) -> tuple[bool, str]:
    """
    Проверяет разрешает ли режим BTC открывать позицию в данном направлении.
    Возвращает (разрешено, причина).
    Блокирует: лонги при strong_down BTC, шорты при strong_up BTC.
    """
    trend = get_btc_trend()
    if trend is None:
        return True, "BTC тренд неизвестен — фильтр пропущен"

    trend_labels = {
        "strong_up":   "🚀 BTC сильный рост",
        "up":          "📈 BTC рост",
        "neutral":     "⚪ BTC боковик",
        "down":        "📉 BTC падение",
        "strong_down": "💥 BTC сильное падение",
    }
    label = trend_labels.get(trend, trend)

    if direction == "long" and trend == "strong_down":
        return False, f"❌ Лонг заблокирован: {label}"
    if direction == "short" and trend == "strong_up":
        return False, f"❌ Шорт заблокирован: {label}"
    return True, f"✅ BTC режим: {label}"


# ─── Частичное закрытие позиции ───────────────────────────────────────────────

def place_partial_tp_orders(symbol, direction, tp1_price, tp2_price, quantity,
                             price_precision, qty_precision, api_key, api_secret, chat_id=None):
    """
    Ставит два TP ордера:
    - TP1: 50% объёма (фиксируем половину прибыли)
    - TP2: оставшиеся 50% (ловим большое движение)
    """
    close_side = "SELL" if direction == "long" else "BUY"
    qty_half = max(round(quantity / 2, qty_precision), 0.001)
    qty_rest = max(round(quantity - qty_half, qty_precision), 0.001)

    tp1_params = {
        "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "quantity": qty_half,
        "stopPrice": round(tp1_price, price_precision),
        "positionSide": "BOTH", "reduceOnly": "true", "workingType": "MARK_PRICE",
    }
    tp2_params = {
        "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "quantity": qty_rest,
        "stopPrice": round(tp2_price, price_precision),
        "positionSide": "BOTH", "reduceOnly": "true", "workingType": "MARK_PRICE",
    }

    def _try(params, label):
        for attempt in range(1, 4):
            res = futures_signed_request("POST", "fapi/v1/order", api_key, api_secret, params, chat_id=chat_id)
            if res and "orderId" in res:
                return res
            logger.warning(f"{label} попытка {attempt}/3: {res}")
            time.sleep(1.5)
        return res

    tp1_res = _try(tp1_params, f"TP1 {symbol}")
    tp2_res = _try(tp2_params, f"TP2 {symbol}")
    return tp1_res, tp2_res

def klines_to_df(klines):
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calc_bollinger(series, period=20, std=2):
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_structural_sl(price, atr, direction, df, lookback=ATR_SL_LOOKBACK):
    """Стоп-лосс с учётом структуры рынка, а не голого ATR.

    Логика (для лонга):
      - базовый SL = price - ATR*ATR_SL_MULT (как раньше);
      - ищем ближайший swing-low за последние `lookback` баров — уровень,
        ниже которого логично признать сценарий сломанным;
      - если swing-low лежит ВНУТРИ ATR-диапазона (ближе к цене), берём его —
        стоп получается tighter, меньше выбивает шумом;
      - ограничиваем снизу: SL не дальше price - ATR*ATR_SL_CAP_MULT, чтобы
        слишком «глубокий» swing не раздувал риск сверх меры.
    Для шорта — зеркально (по swing-high).

    Возвращает (sl_price, r_distance) где r_distance = |price - sl| —
    фактический риск на инструмент, из него далее считаются TP1/TP2 (R:R 1.5 / 3.0).
    """
    base_risk = atr * ATR_SL_MULT
    max_risk = atr * ATR_SL_CAP_MULT

    window = df.tail(lookback)
    try:
        if direction == "long":
            swing = float(window["low"].min())
            sl_atr = price - base_risk
            sl_cap = price - max_risk
            # swing должен быть МЕЖДУ cap и ценой входа; берём ближайший к цене из (sl_atr, swing)
            if sl_cap < swing < price:
                sl = max(sl_atr, swing)  # tighter стоп → max() поднимает SL ближе к цене
            else:
                sl = sl_atr
        else:  # short
            swing = float(window["high"].max())
            sl_atr = price + base_risk
            sl_cap = price + max_risk
            if price < swing < sl_cap:
                sl = min(sl_atr, swing)
            else:
                sl = sl_atr
    except Exception:
        sl = price - base_risk if direction == "long" else price + base_risk

    risk = abs(price - sl)
    if risk <= 0:
        risk = base_risk
        sl = price - base_risk if direction == "long" else price + base_risk
    return sl, risk

def detect_candle_patterns(df):
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    body = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    total_range = h - l

    # Молот (Hammer)
    if lower_wick > 2 * body and upper_wick < body * 0.5 and total_range > 0:
        patterns.append("🔨 Молот (bullish)")

    # Перевёрнутый молот
    if upper_wick > 2 * body and lower_wick < body * 0.5 and c > o:
        patterns.append("🔨 Перевёрнутый молот (bullish)")

    # Доджи
    if body < total_range * 0.1 and total_range > 0:
        patterns.append("➕ Доджи (неопределённость)")

    # Бычье поглощение
    if prev["close"] < prev["open"] and c > o and c > prev["open"] and o < prev["close"]:
        patterns.append("🟢 Бычье поглощение (bullish)")

    # Медвежье поглощение
    if prev["close"] > prev["open"] and c < o and c < prev["open"] and o > prev["close"]:
        patterns.append("🔴 Медвежье поглощение (bearish)")

    # Утренняя звезда
    if (prev2["close"] < prev2["open"] and
        abs(prev["close"] - prev["open"]) < abs(prev2["close"] - prev2["open"]) * 0.3 and
        c > o and c > (prev2["open"] + prev2["close"]) / 2):
        patterns.append("🌟 Утренняя звезда (bullish)")

    # Вечерняя звезда
    if (prev2["close"] > prev2["open"] and
        abs(prev["close"] - prev["open"]) < abs(prev2["close"] - prev2["open"]) * 0.3 and
        c < o and c < (prev2["open"] + prev2["close"]) / 2):
        patterns.append("⭐ Вечерняя звезда (bearish)")

    # Shooting star
    if upper_wick > 2 * body and lower_wick < body * 0.3 and c < o:
        patterns.append("🌠 Падающая звезда (bearish)")

    return patterns if patterns else ["— Явных паттернов не обнаружено"]

def find_support_resistance(df, lookback=50):
    recent = df.tail(lookback)
    highs = recent["high"].nlargest(3).values
    lows = recent["low"].nsmallest(3).values
    return lows, highs

def check_divergence(df):
    rsi = calc_rsi(df["close"])
    price_last = df["close"].iloc[-1]
    price_prev = df["close"].iloc[-10]
    rsi_last = rsi.iloc[-1]
    rsi_prev = rsi.iloc[-10]

    if price_last > price_prev and rsi_last < rsi_prev:
        return "🔴 Медвежья дивергенция RSI (цена ↑, RSI ↓)"
    elif price_last < price_prev and rsi_last > rsi_prev:
        return "🟢 Бычья дивергенция RSI (цена ↓, RSI ↑)"
    return None

# ─── СТРАТЕГИИ ────────────────────────────────────────────────────────────────

def strategy_mean_reversion(df, r):
    """
    Mean Reversion (контртренд, отскок от уровня).
    Логика: RSI < 30 + цена у нижней BB + есть поддержка рядом → лонг.
            RSI > 70 + цена у верхней BB + есть сопротивление → шорт.
    Лучше работает в боковике (neutral trend).
    Винрейт ~55-70%, средний R ~0.5-1.5.
    """
    score = 0
    signals = []

    rsi = r["rsi"]
    bb_pos = r["bb_pos"]
    trend = r["trend"]
    close = df["close"].iloc[-1]
    supports = r["supports"]
    resistances = r["resistances"]

    # Условие боковика или умеренного тренда
    sideways = trend in ("neutral", "up", "down")

    # Лонг: перепроданность + нижняя BB + близко к поддержке
    if rsi < 30 and bb_pos == "below" and sideways:
        score += 3
        signals.append("RSI < 30 + цена у нижней BB ✅")
        if len(supports) > 0 and close <= supports[0] * 1.01:
            score += 2
            signals.append("Цена у поддержки ✅")

    elif rsi < 35 and bb_pos == "below":
        score += 1.5
        signals.append("RSI перепродан + нижняя BB ✅")

    # Шорт: перекупленность + верхняя BB + близко к сопротивлению
    if rsi > 70 and bb_pos == "above" and sideways:
        score -= 3
        signals.append("RSI > 70 + цена у верхней BB ❌")
        if len(resistances) > 0 and close >= resistances[0] * 0.99:
            score -= 2
            signals.append("Цена у сопротивления ❌")

    elif rsi > 65 and bb_pos == "above":
        score -= 1.5
        signals.append("RSI перекуплен + верхняя BB ❌")

    # Бычья дивергенция усиливает лонг
    if r["divergence"] and "Бычья" in r["divergence"] and score > 0:
        score += 1.5
        signals.append("Бычья дивергенция подтверждает ✅")

    # Медвежья дивергенция усиливает шорт
    if r["divergence"] and "Медвежья" in r["divergence"] and score < 0:
        score -= 1.5
        signals.append("Медвежья дивергенция подтверждает ❌")

    # Паттерны разворота
    for p in r["patterns"]:
        if score > 0 and "bullish" in p:
            score += 0.5
            signals.append(f"Разворотный паттерн: {p} ✅")
        elif score < 0 and "bearish" in p:
            score -= 0.5
            signals.append(f"Разворотный паттерн: {p} ❌")

    # Не торгуем против сильного тренда
    if trend == "strong_up" and score < 0:
        score *= 0.3
        signals.append("⚠️ Сильный аптренд — шорт опасен")
    if trend == "strong_down" and score > 0:
        score *= 0.3
        signals.append("⚠️ Сильный даунтренд — лонг опасен")

    return round(score, 2), signals


def strategy_trend_pullback(df, r):
    """
    Trend Following / Pullback в тренде.
    Логика: сильный тренд (EMA выстроены) + откат к EMA20/50 → вход по тренду.
    Золотой/мёртвый крест как дополнительный фильтр.
    Винрейт ~35-50%, R:R ~2-4.
    """
    score = 0
    signals = []

    close = df["close"].iloc[-1]
    ema20 = r["ema20"]
    ema50 = r["ema50"]
    ema200 = r["ema200"]
    trend = r["trend"]

    # Восходящий тренд: цена над EMA200, EMA20 > EMA50
    if trend in ("strong_up", "up") and ema20 > ema50 > ema200:
        # Откат к EMA20 (точка входа в лонг)
        if close <= ema20 * 1.005:
            score += 3
            signals.append("Тренд вверх + откат к EMA20 ✅")
        # Откат к EMA50 (более глубокий, тоже валидный)
        elif close <= ema50 * 1.01:
            score += 2
            signals.append("Тренд вверх + откат к EMA50 ✅")
        else:
            score += 1
            signals.append("Восходящий тренд ✅")

    # Нисходящий тренд: цена под EMA200, EMA20 < EMA50
    elif trend in ("strong_down", "down") and ema20 < ema50:
        # Откат к EMA20 (точка входа в шорт)
        if close >= ema20 * 0.995:
            score -= 3
            signals.append("Тренд вниз + откат к EMA20 ❌")
        elif close >= ema50 * 0.99:
            score -= 2
            signals.append("Тренд вниз + откат к EMA50 ❌")
        else:
            score -= 1
            signals.append("Нисходящий тренд ❌")

    # Золотой/мёртвый крест усиливает сигнал
    if r["ema_cross"] == "golden":
        score += 2
        signals.append("Золотой крест EMA20/50 🌟")
    elif r["ema_cross"] == "death":
        score -= 2
        signals.append("Мёртвый крест EMA20/50 💀")

    # MACD подтверждает
    if r["macd_cross"] == "bullish" and score > 0:
        score += 1
        signals.append("MACD бычье пересечение подтверждает ✅")
    elif r["macd_cross"] == "bearish" and score < 0:
        score -= 1
        signals.append("MACD медвежье пересечение подтверждает ❌")

    # RSI не должен быть в крайней зоне при входе по тренду
    if 40 <= r["rsi"] <= 60 and score > 0:
        score += 0.5
        signals.append("RSI в нейтральной зоне (хороший откат) ✅")

    return round(score, 2), signals


def strategy_breakout(df, r):
    """
    Breakout (пробой диапазона/уровня + сужение волатильности).
    Логика: цена пробивает сопротивление/поддержку + ATR подтверждает импульс
            + объём выше среднего.
    Винрейт ~40-50%, много ложных пробоев — фильтруем объёмом и ATR.
    """
    score = 0
    signals = []

    close = df["close"].iloc[-1]
    supports = r["supports"]
    resistances = r["resistances"]

    atr = calc_atr(df, 14).iloc[-1]
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]
    last_volume = df["volume"].iloc[-1]

    # Пробой сопротивления вверх
    if len(resistances) > 0:
        res = resistances[0]
        if close > res * 1.002:  # пробой с небольшим запасом
            score += 2
            signals.append(f"Пробой сопротивления {fmt_price(res)} ✅")

            # Подтверждение объёмом
            if last_volume > avg_volume * 1.5:
                score += 2
                signals.append("Объём выше среднего в 1.5x — пробой сильный ✅")
            elif last_volume > avg_volume * 1.2:
                score += 1
                signals.append("Объём немного выше среднего ✅")
            else:
                score -= 1
                signals.append("⚠️ Объём слабый — возможен ложный пробой")

            # ATR подтверждает импульс
            candle_body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
            if candle_body > atr * 0.7:
                score += 1
                signals.append("Сильная свеча пробоя (тело > 0.7 ATR) ✅")

    # Пробой поддержки вниз
    if len(supports) > 0:
        sup = supports[0]
        if close < sup * 0.998:
            score -= 2
            signals.append(f"Пробой поддержки {fmt_price(sup)} ❌")

            if last_volume > avg_volume * 1.5:
                score -= 2
                signals.append("Объём выше среднего — пробой сильный ❌")
            elif last_volume > avg_volume * 1.2:
                score -= 1
                signals.append("Объём немного выше среднего ❌")
            else:
                score += 1
                signals.append("⚠️ Объём слабый — возможен ложный пробой вниз")

            candle_body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
            if candle_body > atr * 0.7:
                score -= 1
                signals.append("Сильная свеча пробоя вниз ❌")

    # Сужение BB (волатильность сжимается перед пробоем)
    bb_upper, bb_mid, bb_lower = calc_bollinger(df["close"])
    bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1]
    bb_width_prev = (bb_upper.iloc[-20] - bb_lower.iloc[-20]) / bb_mid.iloc[-20] if len(df) > 20 else bb_width
    if bb_width < bb_width_prev * 0.7:
        signals.append("BB сужены — волатильность сжата, возможен пробой ⚡")

    return round(score, 2), signals


def strategy_divergence_swing(df, r):
    """
    Свинг по дивергенции (RSI/MACD vs цена).
    Логика: дивергенция RSI + подтверждение MACD + паттерн разворота.
    Винрейт ~45-55%, сигнал редкий, но качественный.
    """
    score = 0
    signals = []

    close = df["close"]
    rsi_series = calc_rsi(close)
    macd_line, macd_sig, macd_hist = calc_macd(close)

    # Бычья дивергенция RSI: цена делает новый минимум, RSI — нет
    if len(close) >= 20:
        price_low1 = close.iloc[-1]
        price_low2 = close.iloc[-15:-5].min()
        rsi_low1 = rsi_series.iloc[-1]
        rsi_low2 = rsi_series.iloc[-15:-5].min()

        if price_low1 < price_low2 and rsi_low1 > rsi_low2:
            score += 3
            signals.append("Бычья дивергенция RSI (новый минимум цены, RSI выше) ✅")

        # Медвежья дивергенция RSI: цена делает новый максимум, RSI — нет
        price_high1 = close.iloc[-1]
        price_high2 = close.iloc[-15:-5].max()
        rsi_high1 = rsi_series.iloc[-1]
        rsi_high2 = rsi_series.iloc[-15:-5].max()

        if price_high1 > price_high2 and rsi_high1 < rsi_high2:
            score -= 3
            signals.append("Медвежья дивергенция RSI (новый максимум цены, RSI ниже) ❌")

    # Дивергенция MACD-гистограммы
    if len(macd_hist) >= 10:
        hist_now = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-8:-3].min()
        price_now = close.iloc[-1]
        price_prev_low = close.iloc[-8:-3].min()

        if price_now < price_prev_low and hist_now > hist_prev:
            score += 2
            signals.append("Бычья дивергенция MACD-гистограммы ✅")

        hist_prev_high = macd_hist.iloc[-8:-3].max()
        price_prev_high = close.iloc[-8:-3].max()
        if price_now > price_prev_high and hist_now < hist_prev_high:
            score -= 2
            signals.append("Медвежья дивергенция MACD-гистограммы ❌")

    # Паттерн разворота подтверждает
    for p in r["patterns"]:
        if score > 0 and "bullish" in p:
            score += 1
            signals.append(f"Паттерн разворота: {p} ✅")
        elif score < 0 and "bearish" in p:
            score -= 1
            signals.append(f"Паттерн разворота: {p} ❌")

    # RSI в зоне перепроданности/перекупленности усиливает
    if score > 0 and r["rsi"] < 40:
        score += 1
        signals.append(f"RSI {r['rsi']:.1f} — зона перепроданности ✅")
    if score < 0 and r["rsi"] > 60:
        score -= 1
        signals.append(f"RSI {r['rsi']:.1f} — зона перекупленности ❌")

    return round(score, 2), signals


def strategy_multitf_confirm(r1h, r4h):
    """
    Мультитаймфрейм подтверждение (1H + 4H).
    Логика: 4H задаёт направление (тренд/зона), 1H даёт точку входа.
    Фильтрует шум младшего ТФ старшим.
    Обычно повышает качество сигналов других стратегий.
    """
    score = 0
    signals = []

    # 4H задаёт главный тренд
    if r4h["trend"] in ("strong_up", "up"):
        score += 2
        signals.append("4H: восходящий тренд (главное направление) ✅")
    elif r4h["trend"] in ("strong_down", "down"):
        score -= 2
        signals.append("4H: нисходящий тренд (главное направление) ❌")

    # 1H подтверждает направление 4H
    tf_align = False
    if r4h["trend"] in ("strong_up", "up") and r1h["trend"] in ("strong_up", "up"):
        score += 2
        signals.append("1H и 4H тренды совпадают — сильное подтверждение ✅")
        tf_align = True
    elif r4h["trend"] in ("strong_down", "down") and r1h["trend"] in ("strong_down", "down"):
        score -= 2
        signals.append("1H и 4H тренды совпадают — сильное подтверждение ❌")
        tf_align = True
    elif r4h["trend"] in ("strong_up", "up") and r1h["trend"] in ("strong_down", "down"):
        signals.append("⚠️ 1H против 4H — возможен откат, входить осторожно")
    elif r4h["trend"] in ("strong_down", "down") and r1h["trend"] in ("strong_up", "up"):
        signals.append("⚠️ 1H против 4H — возможен откат, входить осторожно")

    # RSI на обоих ТФ в одном направлении
    if r4h["rsi"] < 40 and r1h["rsi"] < 45:
        score += 1.5
        signals.append(f"RSI перепродан на обоих ТФ (4H: {r4h['rsi']:.0f}, 1H: {r1h['rsi']:.0f}) ✅")
    elif r4h["rsi"] > 60 and r1h["rsi"] > 55:
        score -= 1.5
        signals.append(f"RSI перекуплен на обоих ТФ (4H: {r4h['rsi']:.0f}, 1H: {r1h['rsi']:.0f}) ❌")

    # MACD на обоих ТФ совпадают
    macd_bull_4h = r4h["macd"] > r4h["macd_signal"]
    macd_bull_1h = r1h["macd"] > r1h["macd_signal"]
    if macd_bull_4h and macd_bull_1h:
        score += 1
        signals.append("MACD бычий на 1H и 4H ✅")
    elif not macd_bull_4h and not macd_bull_1h:
        score -= 1
        signals.append("MACD медвежий на 1H и 4H ❌")

    # EMA кросс на 4H — самый сильный фильтр
    if r4h["ema_cross"] == "golden":
        score += 2
        signals.append("4H Золотой крест — мощный бычий сигнал 🌟")
    elif r4h["ema_cross"] == "death":
        score -= 2
        signals.append("4H Мёртвый крест — мощный медвежий сигнал 💀")

    # Дивергенция на 4H — редкий но качественный сигнал
    if r4h["divergence"]:
        if "Бычья" in r4h["divergence"]:
            score += 2
            signals.append(f"4H {r4h['divergence']} ✅")
        elif "Медвежья" in r4h["divergence"]:
            score -= 2
            signals.append(f"4H {r4h['divergence']} ❌")

    return round(score, 2), signals, tf_align


def run_all_strategies(df1h, r1h, df4h=None, r4h=None):
    """
    Запускает все стратегии и возвращает сводную таблицу результатов.
    """
    results = {}

    results["mean_reversion"] = strategy_mean_reversion(df1h, r1h)
    results["trend_pullback"] = strategy_trend_pullback(df1h, r1h)
    results["breakout"] = strategy_breakout(df1h, r1h)
    results["divergence_swing"] = strategy_divergence_swing(df1h, r1h)

    if r4h is not None:
        mtf_score, mtf_signals, tf_align = strategy_multitf_confirm(r1h, r4h)
        results["multitf"] = (mtf_score, mtf_signals)
    else:
        tf_align = False

    # Суммарный взвешенный счёт всех стратегий
    weights = {
        "mean_reversion":  1.0,
        "trend_pullback":  1.2,
        "breakout":        0.8,
        "divergence_swing": 1.0,
        "multitf":         1.5,
    }
    total = sum(results[k][0] * weights.get(k, 1.0) for k in results)

    return results, total, tf_align


def fmt_strategy_score(score):
    if score >= 3:
        return f"🟢 +{score:.1f}"
    elif score >= 1:
        return f"🟡 +{score:.1f}"
    elif score <= -3:
        return f"🔴 {score:.1f}"
    elif score <= -1:
        return f"🟠 {score:.1f}"
    else:
        return f"⚪ {score:.1f}"


def full_analysis(df):
    """Возвращает dict с результатами анализа."""
    close = df["close"]
    result = {}

    # RSI
    rsi = calc_rsi(close)
    result["rsi"] = rsi.iloc[-1]

    # MACD
    macd, signal, hist = calc_macd(close)
    result["macd"] = macd.iloc[-1]
    result["macd_signal"] = signal.iloc[-1]
    result["macd_hist"] = hist.iloc[-1]
    result["macd_cross"] = "bullish" if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2] else \
                           "bearish" if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2] else "none"

    # Bollinger
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    result["bb_upper"] = bb_upper.iloc[-1]
    result["bb_mid"] = bb_mid.iloc[-1]
    result["bb_lower"] = bb_lower.iloc[-1]
    result["bb_pos"] = "above" if close.iloc[-1] > bb_upper.iloc[-1] else \
                       "below" if close.iloc[-1] < bb_lower.iloc[-1] else "inside"

    # EMA
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)
    result["ema20"] = ema20.iloc[-1]
    result["ema50"] = ema50.iloc[-1]
    result["ema200"] = ema200.iloc[-1]

    # EMA cross
    if ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] <= ema50.iloc[-2]:
        result["ema_cross"] = "golden"
    elif ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-2] >= ema50.iloc[-2]:
        result["ema_cross"] = "death"
    else:
        result["ema_cross"] = "none"

    # Тренд по EMA
    p = close.iloc[-1]
    if p > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
        result["trend"] = "strong_up"
    elif p > ema50.iloc[-1] > ema200.iloc[-1]:
        result["trend"] = "up"
    elif p < ema20.iloc[-1] and p < ema50.iloc[-1] and p < ema200.iloc[-1]:
        result["trend"] = "strong_down"
    elif p < ema50.iloc[-1]:
        result["trend"] = "down"
    else:
        result["trend"] = "neutral"

    # Паттерны
    result["patterns"] = detect_candle_patterns(df)

    # Уровни
    result["supports"], result["resistances"] = find_support_resistance(df)

    # Дивергенция
    result["divergence"] = check_divergence(df)

    return result

TIMEFRAME_WEIGHTS = {
    "5m": 0.5,
    "15m": 0.75,
    "1h": 1.0,
    "4h": 2.0,
}

def generate_signal_multi(results: dict):
    """
    results: {"5m": r5, "15m": r15, "1h": r1h, "4h": r4h}
    Считает общий сигнал с учётом весов таймфреймов.
    Возвращает: signal, strength, emoji, score, reasons, max_score
    """
    score = 0.0
    reasons = []
    max_score = 0.0

    for tf, r in results.items():
        if r is None:
            continue
        w = TIMEFRAME_WEIGHTS.get(tf, 1.0)
        max_score += w * 9  # примерный потолок вклада одного ТФ

        # Тренд
        if r["trend"] in ("strong_up", "up"):
            score += w * 2
            reasons.append(f"{tf} тренд восходящий ✅")
        elif r["trend"] in ("strong_down", "down"):
            score -= w * 2
            reasons.append(f"{tf} тренд нисходящий ❌")

        # RSI
        if r["rsi"] < 35:
            score += w * 1.5
            reasons.append(f"{tf} RSI перепродан ({r['rsi']:.1f}) ✅")
        elif r["rsi"] > 65:
            score -= w * 1.5
            reasons.append(f"{tf} RSI перекуплен ({r['rsi']:.1f}) ❌")

        # MACD
        if r["macd_cross"] == "bullish":
            score += w * 1.5
            reasons.append(f"{tf} MACD бычье пересечение ✅")
        elif r["macd_cross"] == "bearish":
            score -= w * 1.5
            reasons.append(f"{tf} MACD медвежье пересечение ❌")
        elif r["macd"] > r["macd_signal"]:
            score += w * 0.5
        else:
            score -= w * 0.5

        # EMA cross
        if r["ema_cross"] == "golden":
            score += w * 2
            reasons.append(f"{tf} Золотой крест EMA20/50 🌟")
        elif r["ema_cross"] == "death":
            score -= w * 2
            reasons.append(f"{tf} Мёртвый крест EMA20/50 💀")

        # Bollinger
        if r["bb_pos"] == "below":
            score += w * 0.5
        elif r["bb_pos"] == "above":
            score -= w * 0.5

        # Паттерны
        for p in r["patterns"]:
            if "bullish" in p:
                score += w * 0.5
                reasons.append(f"{tf} паттерн: {p} ✅")
            elif "bearish" in p:
                score -= w * 0.5
                reasons.append(f"{tf} паттерн: {p} ❌")

        # Дивергенция
        if r["divergence"]:
            if "Бычья" in r["divergence"]:
                score += w * 1.5
                reasons.append(f"{tf} {r['divergence']} ✅")
            elif "Медвежья" in r["divergence"]:
                score -= w * 1.5
                reasons.append(f"{tf} {r['divergence']} ❌")

    norm = (score / max_score) * 15 if max_score else 0  # нормируем к старой шкале ±15

    if norm >= 5:
        signal, strength, emoji = "🟢 ЛОНГ", "Сильный", "🚀"
    elif norm >= 2:
        signal, strength, emoji = "🟡 ЛОНГ", "Слабый", "📈"
    elif norm <= -5:
        signal, strength, emoji = "🔴 ШОРТ", "Сильный", "💥"
    elif norm <= -2:
        signal, strength, emoji = "🟡 ШОРТ", "Слабый", "📉"
    else:
        signal, strength, emoji = "⚪ НЕЙТРАЛЬНО", "Нет сигнала", "⏸"

    return signal, strength, emoji, norm, reasons


    """Генерирует итоговый сигнал на основе 1h и 4h анализа."""
    score = 0
    reasons = []

    # ── 4h (вес x2) ──
    # Тренд
    if r4h["trend"] in ("strong_up", "up"):
        score += 2
        reasons.append("4h тренд восходящий ✅")
    elif r4h["trend"] in ("strong_down", "down"):
        score -= 2
        reasons.append("4h тренд нисходящий ❌")

    # RSI 4h
    if r4h["rsi"] < 35:
        score += 2
        reasons.append(f"4h RSI перепродан ({r4h['rsi']:.1f}) ✅")
    elif r4h["rsi"] > 65:
        score -= 2
        reasons.append(f"4h RSI перекуплен ({r4h['rsi']:.1f}) ❌")

    # MACD 4h
    if r4h["macd_cross"] == "bullish":
        score += 2
        reasons.append("4h MACD бычье пересечение ✅")
    elif r4h["macd_cross"] == "bearish":
        score -= 2
        reasons.append("4h MACD медвежье пересечение ❌")
    elif r4h["macd"] > r4h["macd_signal"]:
        score += 1
        reasons.append("4h MACD > Signal ✅")
    else:
        score -= 1
        reasons.append("4h MACD < Signal ❌")

    # EMA cross 4h
    if r4h["ema_cross"] == "golden":
        score += 3
        reasons.append("4h Золотой крест EMA20/50 🌟")
    elif r4h["ema_cross"] == "death":
        score -= 3
        reasons.append("4h Мёртвый крест EMA20/50 💀")

    # ── 1h ──
    # RSI 1h
    if r1h["rsi"] < 35:
        score += 1
        reasons.append(f"1h RSI перепродан ({r1h['rsi']:.1f}) ✅")
    elif r1h["rsi"] > 65:
        score -= 1
        reasons.append(f"1h RSI перекуплен ({r1h['rsi']:.1f}) ❌")

    # MACD 1h
    if r1h["macd_cross"] == "bullish":
        score += 1
        reasons.append("1h MACD бычье пересечение ✅")
    elif r1h["macd_cross"] == "bearish":
        score -= 1
        reasons.append("1h MACD медвежье пересечение ❌")

    # Bollinger 1h
    if r1h["bb_pos"] == "below":
        score += 1
        reasons.append("1h Цена ниже нижней BB (отскок?) ✅")
    elif r1h["bb_pos"] == "above":
        score -= 1
        reasons.append("1h Цена выше верхней BB (перегрев?) ❌")

    # Паттерны 1h
    for p in r1h["patterns"]:
        if "bullish" in p:
            score += 1
            reasons.append(f"Паттерн: {p} ✅")
        elif "bearish" in p:
            score -= 1
            reasons.append(f"Паттерн: {p} ❌")

    # Дивергенция
    if r1h["divergence"]:
        if "Бычья" in r1h["divergence"]:
            score += 2
            reasons.append(r1h["divergence"] + " ✅")
        elif "Медвежья" in r1h["divergence"]:
            score -= 2
            reasons.append(r1h["divergence"] + " ❌")

    # Итог
    if score >= 5:
        signal = "🟢 ЛОНГ"
        strength = "Сильный"
        emoji = "🚀"
    elif score >= 2:
        signal = "🟡 ЛОНГ"
        strength = "Слабый"
        emoji = "📈"
    elif score <= -5:
        signal = "🔴 ШОРТ"
        strength = "Сильный"
        emoji = "💥"
    elif score <= -2:
        signal = "🟡 ШОРТ"
        strength = "Слабый"
        emoji = "📉"
    else:
        signal = "⚪ НЕЙТРАЛЬНО"
        strength = "Нет сигнала"
        emoji = "⏸"

    return signal, strength, emoji, score, reasons

def build_analysis_chart(symbol, df1h, df4h, r1h, r4h):
    """Строит график с индикаторами для анализа."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), facecolor="#0d1117",
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{symbol} — Технический анализ", color="white", fontsize=14, fontweight="bold")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

    def plot_candles_with_indicators(ax_price, ax_rsi, ax_macd, df, r, label):
        close = df["close"]
        times = df["open_time"]

        # Свечи
        for _, row in df.tail(60).iterrows():
            color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
            ax_price.plot([row["open_time"], row["open_time"]], [row["low"], row["high"]], color=color, linewidth=0.6)
            ax_price.bar(row["open_time"], abs(row["close"] - row["open"]),
                        bottom=min(row["open"], row["close"]),
                        color=color, width=pd.Timedelta(minutes=int(label[0])*45), alpha=0.9)

        # EMA
        ema20 = calc_ema(close, 20)
        ema50 = calc_ema(close, 50)
        ax_price.plot(times, ema20, color="#f0b90b", linewidth=1, label="EMA20", alpha=0.8)
        ax_price.plot(times, ema50, color="#2196F3", linewidth=1, label="EMA50", alpha=0.8)

        # Bollinger
        bb_u, bb_m, bb_l = calc_bollinger(close)
        ax_price.plot(times, bb_u, color="#9C27B0", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_price.plot(times, bb_l, color="#9C27B0", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_price.fill_between(times, bb_l, bb_u, alpha=0.05, color="#9C27B0")

        ax_price.set_title(label, color="white", fontsize=10)
        ax_price.legend(facecolor="#161b22", labelcolor="white", fontsize=7)

        # RSI
        rsi = calc_rsi(close)
        ax_rsi.plot(times, rsi, color="#FF9800", linewidth=1)
        ax_rsi.axhline(70, color="#ef5350", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_rsi.axhline(30, color="#26a69a", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_rsi.fill_between(times, rsi, 70, where=(rsi >= 70), alpha=0.3, color="#ef5350")
        ax_rsi.fill_between(times, rsi, 30, where=(rsi <= 30), alpha=0.3, color="#26a69a")
        ax_rsi.set_ylabel("RSI", color="#8b949e", fontsize=7)
        ax_rsi.set_ylim(0, 100)

        # MACD
        macd, sig, hist = calc_macd(close)
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
        ax_macd.bar(times, hist, color=colors, alpha=0.7)
        ax_macd.plot(times, macd, color="#2196F3", linewidth=1, label="MACD")
        ax_macd.plot(times, sig, color="#FF9800", linewidth=1, label="Signal")
        ax_macd.axhline(0, color="#8b949e", linewidth=0.5)
        ax_macd.set_ylabel("MACD", color="#8b949e", fontsize=7)
        ax_macd.legend(facecolor="#161b22", labelcolor="white", fontsize=6)

    plot_candles_with_indicators(axes[0][0], axes[1][0], axes[2][0], df1h, r1h, "1H таймфрейм")
    plot_candles_with_indicators(axes[0][1], axes[1][1], axes[2][1], df4h, r4h, "4H таймфрейм")

    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    buf.seek(0)
    plt.close()
    return buf

# ─── Chart builder (simple) ───────────────────────────────────────────────────

def build_chart(symbol, klines, interval_label="1h / 3 дня"):
    df = klines_to_df(klines)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [3, 1]}, facecolor="#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    for _, row in df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax1.plot([row["open_time"], row["open_time"]], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax1.bar(row["open_time"], abs(row["close"] - row["open"]),
                bottom=min(row["open"], row["close"]),
                color=color, width=pd.Timedelta(minutes=45), alpha=0.9)

    close = df["close"]
    if len(df) >= 20:
        sma20 = close.rolling(20).mean()
        ax1.plot(df["open_time"], sma20, color="#f0b90b", linewidth=1.2, label="SMA20", alpha=0.8)
        ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=9)

    vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax2.bar(df["open_time"], df["volume"], color=vol_colors, alpha=0.7)

    for ax in [ax1, ax2]:
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    ax1.set_title(f"{symbol}  •  {interval_label}", color="white", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Объём", color="#8b949e", fontsize=8)

    last_price = close.iloc[-1]
    change = ((last_price - close.iloc[0]) / close.iloc[0]) * 100
    color_txt = "#26a69a" if change >= 0 else "#ef5350"
    sign = "+" if change >= 0 else ""
    ax1.text(0.01, 0.97, f"{fmt_price(last_price)}  {sign}{change:.2f}%",
             transform=ax1.transAxes, color=color_txt, fontsize=11, fontweight="bold", va="top")

    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    buf.seek(0)
    plt.close()
    return buf

# ─── Market scanner ────────────────────────────────────────────────────────────

STABLECOINS = {
    "USDCUSDT", "USD1USDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT",
    "DAIUSDT", "USDPUSDT", "GUSDUSDT", "EURUSDT", "USTCUSDT",
    "PYUSDUSDT", "USDEUSDT", "FRAXUSDT", "USDDUSDT",
}

def get_all_usdt_symbols(min_volume=5_000_000, max_symbols=60):
    """Возвращает список самых ликвидных USDT-пар по 24h объёму (без стейблкоинов)."""
    try:
        r = requests.get(f"{BINANCE_API}/ticker/24hr", timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"get_all_usdt_symbols error: {e}")
        return []

    pairs = [
        d for d in data
        if d["symbol"].endswith("USDT")
        and not d["symbol"].endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))
        and d["symbol"] not in STABLECOINS
        and float(d["quoteVolume"]) > min_volume
    ]
    pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:max_symbols]]


def quick_score_symbol(symbol):
    """Быстрый скоринг одного символа на 1H для сканера рынка. Блокирующая функция."""
    try:
        klines = get_klines(symbol, "1h", 100)
        if not klines or len(klines) < 60:
            return None
        df = klines_to_df(klines)
        r = full_analysis(df)

        score = 0.0
        if r["trend"] in ("strong_up", "up"):
            score += 2
        elif r["trend"] in ("strong_down", "down"):
            score -= 2

        if r["rsi"] < 35:
            score += 1.5
        elif r["rsi"] > 65:
            score -= 1.5

        if r["macd_cross"] == "bullish":
            score += 1.5
        elif r["macd_cross"] == "bearish":
            score -= 1.5
        elif r["macd"] > r["macd_signal"]:
            score += 0.5
        else:
            score -= 0.5

        if r["ema_cross"] == "golden":
            score += 2
        elif r["ema_cross"] == "death":
            score -= 2

        if r["bb_pos"] == "below":
            score += 0.5
        elif r["bb_pos"] == "above":
            score -= 0.5

        if r["divergence"]:
            if "Бычья" in r["divergence"]:
                score += 1.5
            elif "Медвежья" in r["divergence"]:
                score -= 1.5

        price = df["close"].iloc[-1]
        change_24 = ((price - df["close"].iloc[-25]) / df["close"].iloc[-25]) * 100 if len(df) > 25 else 0

        # Доп. защита от плоских/пеговых активов (вдруг не попали в список стейблов)
        recent_std_pct = df["close"].tail(48).pct_change().std() * 100
        if recent_std_pct < 0.05:
            return None

        atr = calc_atr(df, 14).iloc[-1]
        risk = atr * 1.5

        if score > 0:
            sl = price - risk
            tp1 = price + risk * 1.5
            tp2 = price + risk * 3.0
        elif score < 0:
            sl = price + risk
            tp1 = price - risk * 1.5
            tp2 = price - risk * 3.0
        else:
            sl = tp1 = tp2 = None

        return {
            "symbol": symbol,
            "score": score,
            "price": price,
            "rsi": r["rsi"],
            "trend": r["trend"],
            "change_24h_approx": change_24,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
        }
    except Exception as e:
        logger.warning(f"quick_score_symbol {symbol} error: {e}")
        return None


def scan_market(min_volume=5_000_000, max_symbols=60, max_workers=8):
    """Сканирует рынок и возвращает (longs, shorts) — списки кандидатов, отсортированные по score."""
    symbols = get_all_usdt_symbols(min_volume=min_volume, max_symbols=max_symbols)
    if not symbols:
        return [], []

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(quick_score_symbol, symbols):
            if res:
                results.append(res)
            time.sleep(0.02)  # небольшая пауза, чтобы не упереться в rate limit

    results.sort(key=lambda x: x["score"], reverse=True)
    longs  = [r for r in results if r["score"] > 1.5][:MAX_CANDIDATES_LONG]
    shorts = sorted([r for r in results if r["score"] < -1.5], key=lambda x: x["score"])[:MAX_CANDIDATES_SHORT]
    return longs, shorts


async def scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "🌐 Сканирую рынок (топ-60 ликвидных пар, 1H ТФ)... это займёт ~20-40 сек."
    )

    loop = asyncio.get_event_loop()
    longs, shorts = await loop.run_in_executor(None, scan_market)

    if not longs and not shorts:
        await msg.edit_text("❌ Не удалось просканировать рынок. Попробуй позже.")
        return

    lines = [f"🌐 *Сканирование рынка* | {datetime.utcnow().strftime('%H:%M UTC')}\n"]

    lines.append("🟢 *Топ кандидатов на ЛОНГ:*\n")
    if longs:
        for r in longs:
            lines.append(
                f"• *{r['symbol'].replace('USDT','')}* счёт `{r['score']:+.1f}` | "
                f"RSI `{r['rsi']:.1f}` | Цена: {fmt_price(r['price'])}"
            )
            if r.get("sl"):
                lines.append(
                    f"   🛑 SL: `{fmt_price(r['sl'])}`  🎯 TP1: `{fmt_price(r['tp1'])}`  TP2: `{fmt_price(r['tp2'])}`"
                )
    else:
        lines.append("— нет явных кандидатов")

    lines.append("\n🔴 *Топ кандидатов на ШОРТ:*\n")
    if shorts:
        for r in shorts:
            lines.append(
                f"• *{r['symbol'].replace('USDT','')}* счёт `{r['score']:+.1f}` | "
                f"RSI `{r['rsi']:.1f}` | Цена: {fmt_price(r['price'])}"
            )
            if r.get("sl"):
                lines.append(
                    f"   🛑 SL: `{fmt_price(r['sl'])}`  🎯 TP1: `{fmt_price(r['tp1'])}`  TP2: `{fmt_price(r['tp2'])}`"
                )
    else:
        lines.append("— нет явных кандидатов")

    lines.append(
        "\n_Скан основан на 1H ТФ (тренд, RSI, MACD, EMA-кросс, BB, дивергенция)._\n"
        "Используй `/analyze СИМВОЛ` для детального разбора по 5m/15m/1h/4h перед входом.\n"
        "⚠️ Это не финансовый совет."
    )

    await msg.edit_text("\n".join(lines), parse_mode="Markdown")




HELP_SECTIONS = {
    "main": (
        "👋 *Binance Market Bot*\n\n"
        "Бот показывает рыночные данные, технический анализ и сигналы "
        "ЛОНГ/ШОРТ по монетам с Binance. Выбери раздел кнопкой ниже 👇\n\n"
        "⚠️ _Бот НЕ торгует автоматически и не даёт финансовых советов. "
        "Все сигналы — информационные, решение и риск всегда на тебе._"
    ),
    "data": (
        "📊 *Рыночные данные*\n\n"
        "• `/price BTC` — цена и 24ч статистика\n"
        "• `/chart BTC` — свечной график (1H, 3 дня)\n"
        "• `/futures BTC` — данные фьючерсов: mark price, базис, OI, funding\n"
        "• `/top` — топ-5 gainers и losers за 24ч\n"
        "• `/info BTC` — полная сводка: спот + фьючерсы в одном сообщении"
    ),
    "analysis": (
        "🧠 *Технический анализ*\n\n"
        "• `/analyze BTC` — разбор на 5m/15m/1H/4H: RSI, MACD, Bollinger, "
        "EMA-кроссы, паттерны свечей, уровни, дивергенции\n"
        "  └ Даёт общий взвешенный сигнал ЛОНГ / ШОРТ / НЕЙТРАЛЬНО + SL/TP + график\n\n"
        "• `/strategy BTC` — запускает 5 стратегий сразу с разбивкой:\n"
        "  └ Mean Reversion (отскок от уровня, винрейт ~60%)\n"
        "  └ Trend Pullback (откат в тренде, R:R ~2-4)\n"
        "  └ Breakout (пробой уровня с объёмом)\n"
        "  └ Divergence Swing (свинг по дивергенции RSI/MACD)\n"
        "  └ MultiTF 1H+4H (подтверждение старшим ТФ)\n\n"
        "• `/scan` — сканирует топ-60 ликвидных пар рынка "
        "и находит топ-5 кандидатов на лонг и шорт с SL/TP\n\n"
        "_Сигналы — статистическое преимущество, не гарантия._"
    ),
    "alerts": (
        "🔔 *Алерты (бот сам напишет тебе)*\n\n"
        "• `/subscribe BTC ETH` — подписаться на сильные сигналы по монетам "
        "(проверка каждые 10 мин)\n"
        "• `/unsubscribe BTC` — отписаться (без аргументов — от всех)\n"
        "• `/subscribe_market` — раз в час получать топ лонг/шорт по рынку\n"
        "• `/unsubscribe_market` — отключить рыночные алерты\n"
        "• `/mysubs` — посмотреть свои подписки\n\n"
        "_Подписки хранятся в памяти бота — при перезапуске сервера слетают, "
        "придётся подписаться заново._"
    ),
    "account": (
        "🔑 *Личный аккаунт Binance*\n\n"
        "• `/mode demo` / `/mode real` — выбрать режим: тестовая сеть (виртуальные деньги) или реальный Binance\n"
        "• `/setkey API_KEY SECRET` — подключить ключи для текущего режима\n"
        "• `/balance` — баланс спот-кошелька (только real)\n"
        "• `/positions` — открытые фьючерсные позиции и PnL\n"
        "• `/deletekey` — удалить ключи\n"
        "• `/autotrade on/off/status` — автоторговля по сильным сигналам (нужны ключи с правом Futures Trading)\n"
        "• `/autoportfolio` — открытые авто-позиции и live PnL\n\n"
        "⚠️ _Для real-режима создавай ключ только с нужными правами. НИКОГДА не давай право "
        "на вывод средств (Withdrawal). Используй `/setkey` только в личке "
        "с ботом, не в группах — ключи хранятся в памяти без шифрования._"
    ),
    "risk": (
        "⚠️ *Риски — прочитай перед тем, как торговать*\n\n"
        "• Риск не больше 0.5–1% капитала на сделку\n"
        "• Только деньги, потерю которых можешь себе позволить\n"
        "• Без плеча или минимальное плечо (x2–x3) на фьючерсах\n"
        "• Будь готов к просадке 20–30% — это нормальная часть статистики, "
        "не сигнал, что система сломалась\n"
        "• Не входи против сигнала на эмоциях\n"
        "• Веди журнал сигналов хотя бы первый месяц перед крупными суммами\n\n"
        "Бот не финансовый консультант. Все решения и ответственность — твои."
    ),
}

def help_keyboard(active="main"):
    buttons = [
        [InlineKeyboardButton("📊 Данные", callback_data="help:data"),
         InlineKeyboardButton("🧠 Анализ", callback_data="help:analysis")],
        [InlineKeyboardButton("🔔 Алерты", callback_data="help:alerts"),
         InlineKeyboardButton("🔑 Аккаунт", callback_data="help:account")],
        [InlineKeyboardButton("⚠️ Риски", callback_data="help:risk")],
    ]
    if active != "main":
        buttons.append([InlineKeyboardButton("« Назад", callback_data="help:main")])
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        HELP_SECTIONS["main"], parse_mode="Markdown", reply_markup=help_keyboard()
    )


async def analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Использование: `/analyze BTC`", parse_mode="Markdown")
        return

    symbol = normalize_symbol(ctx.args[0])
    msg = await update.message.reply_text(
        f"🔍 Анализирую *{symbol}* на 5m / 15m / 1H / 4H...", parse_mode="Markdown"
    )

    tf_limits = {"5m": 200, "15m": 200, "1h": 200, "4h": 200}
    klines = {}
    for tf, limit in tf_limits.items():
        klines[tf] = get_klines(symbol, tf, limit)

    if not klines["1h"] or not klines["4h"]:
        await msg.edit_text("❌ Не удалось получить данные. Проверь тикер.")
        return

    dfs = {}
    results = {}
    for tf, kl in klines.items():
        if not kl or len(kl) < 30:
            results[tf] = None
            continue
        df = klines_to_df(kl)
        dfs[tf] = df
        results[tf] = full_analysis(df)

    r1h = results.get("1h")
    r4h = results.get("4h")
    r15m = results.get("15m")
    r5m = results.get("5m")

    signal, strength, emoji, score, reasons = generate_signal_multi(results)

    current_price = dfs["1h"]["close"].iloc[-1]
    supports = r4h["supports"]
    resistances = r4h["resistances"]

    atr_1h = calc_atr(dfs["1h"], 14).iloc[-1]
    risk = atr_1h * 1.5  # риск = 1.5 ATR(14) на 1H

    if "ЛОНГ" in signal:
        sl = current_price - risk
        tp1 = current_price + risk * 1.5   # R:R 1.5
        tp2 = current_price + risk * 3.0   # R:R 3.0
        direction_text = "📈 Входить в ЛОНГ"
    elif "ШОРТ" in signal:
        sl = current_price + risk
        tp1 = current_price - risk * 1.5
        tp2 = current_price - risk * 3.0
        direction_text = "📉 Входить в ШОРТ"
    else:
        sl = tp1 = tp2 = None
        direction_text = "⏸ Ждать чёткого сигнала"

    def tf_line(tf, r):
        if r is None:
            return f"• {tf}: нет данных"
        rsi_state = "🔴 перекуплен" if r["rsi"] > 65 else "🟢 перепродан" if r["rsi"] < 35 else "⚪ норма"
        macd_state = "▲ бычий" if r["macd"] > r["macd_signal"] else "▼ медвежий"
        trend_map = {
            "strong_up": "🚀 Сильный рост", "up": "📈 Рост",
            "strong_down": "💥 Сильное падение", "down": "📉 Падение",
            "neutral": "⚪ Боковик",
        }
        return (f"• *{tf}*: RSI `{r['rsi']:.1f}` {rsi_state} | MACD `{macd_state}` | "
                f"Тренд: {trend_map.get(r['trend'], r['trend'])}")

    lines = [
        f"🧠 *Анализ {symbol}* | {datetime.utcnow().strftime('%H:%M UTC')}\n",
        f"{'━'*28}",
        f"{emoji} *Общий сигнал: {signal}* ({strength})",
        f"📊 Счёт: `{score:+.1f}` / ±15 (взвешено по ТФ)",
        f"{'━'*28}\n",
        f"*💡 Что делать:* {direction_text}",
    ]

    if sl:
        lines += [
            f"🛑 *Stop Loss:* `{fmt_price(sl)}`",
            f"🎯 *TP1:* `{fmt_price(tp1)}`",
            f"🎯 *TP2:* `{fmt_price(tp2)}`",
        ]

    lines += [
        f"\n*📐 По таймфреймам:*",
        tf_line("5m", r5m),
        tf_line("15m", r15m),
        tf_line("1h", r1h),
        tf_line("4h", r4h),
        f"\n*🕯 Паттерны (15m):*",
    ]

    if r15m:
        for p in r15m["patterns"]:
            lines.append(f"• {p}")
    else:
        lines.append("• — нет данных")

    if r1h and r1h["divergence"]:
        lines.append(f"\n*📡 Дивергенция (1H):* {r1h['divergence']}")

    lines += [
        f"\n*📍 Уровни (4H):*",
        f"• Поддержка: `{fmt_price(supports[0])}` / `{fmt_price(supports[1])}`",
        f"• Сопротивление: `{fmt_price(resistances[0])}` / `{fmt_price(resistances[1])}`",
        f"\n*⚡ Причины сигнала (топ-8):*",
    ]
    for r in reasons[:8]:
        lines.append(f"• {r}")

    lines.append(f"\n⚠️ _Это не финансовый совет. Торгуй с умом._")

    await msg.delete()
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    if r1h and r4h:
        buf = build_analysis_chart(symbol, dfs["1h"], dfs["4h"], r1h, r4h)
        await update.message.reply_photo(
            photo=buf, caption=f"📊 *{symbol}* — Технический анализ 1H + 4H", parse_mode="Markdown"
        )




async def strategy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /strategy BTC — запускает все 5 стратегий и показывает сводную таблицу сигналов.
    """
    if not ctx.args:
        await update.message.reply_text(
            "Использование: `/strategy BTC`\n\n"
            "Запускает 5 стратегий сразу:\n"
            "• Mean Reversion (отскок)\n"
            "• Trend Pullback (откат в тренде)\n"
            "• Breakout (пробой уровня)\n"
            "• Divergence Swing (свинг по дивергенции)\n"
            "• MultiTF Confirm (1H + 4H подтверждение)",
            parse_mode="Markdown"
        )
        return

    symbol = normalize_symbol(ctx.args[0])
    msg = await update.message.reply_text(
        f"🔬 Запускаю все стратегии для *{symbol}*...", parse_mode="Markdown"
    )

    klines_1h = get_klines(symbol, "1h", 200)
    klines_4h = get_klines(symbol, "4h", 200)

    if not klines_1h:
        await msg.edit_text("❌ Не удалось получить данные. Проверь тикер.")
        return

    df1h = klines_to_df(klines_1h)
    r1h  = full_analysis(df1h)

    df4h = r4h = None
    if klines_4h:
        df4h = klines_to_df(klines_4h)
        r4h  = full_analysis(df4h)

    strategy_results, total_score, tf_align = run_all_strategies(df1h, r1h, df4h, r4h)

    current_price = df1h["close"].iloc[-1]
    atr = calc_atr(df1h, 14).iloc[-1]
    # SL/TP теперь считаются по структуре (см. calc_structural_sl) — tighter стоп,
    # меньше выбивает шумом. Здесь direction определяем по знаку total_score,
    # т.к. /analyze показывает и нейтральные сигналы (где SL/TP не нужны).
    if total_score > 0:
        sl, risk = calc_structural_sl(current_price, atr, "long", df1h)
    elif total_score < 0:
        sl, risk = calc_structural_sl(current_price, atr, "short", df1h)
    else:
        sl = tp1 = tp2 = None
        risk = atr * 1.5  # только для отображения ATR-эквивалента

    # Итоговый сигнал
    if total_score >= 8:
        final_signal, final_emoji = "🟢 ЛОНГ (Сильный)", "🚀"
    elif total_score >= 3:
        final_signal, final_emoji = "🟡 ЛОНГ (Слабый)", "📈"
    elif total_score <= -8:
        final_signal, final_emoji = "🔴 ШОРТ (Сильный)", "💥"
    elif total_score <= -3:
        final_signal, final_emoji = "🟠 ШОРТ (Слабый)", "📉"
    else:
        final_signal, final_emoji = "⚪ НЕЙТРАЛЬНО", "⏸"

    # SL/TP на основе структуры (рассчитано выше через calc_structural_sl)
    if "ЛОНГ" in final_signal:
        tp1 = current_price + risk * 1.5
        tp2 = current_price + risk * 3.0
    elif "ШОРТ" in final_signal:
        tp1 = current_price - risk * 1.5
        tp2 = current_price - risk * 3.0
    else:
        sl = tp1 = tp2 = None

    strategy_names = {
        "mean_reversion":   "Mean Reversion  (отскок)",
        "trend_pullback":   "Trend Pullback  (по тренду)",
        "breakout":         "Breakout          (пробой)",
        "divergence_swing": "Divergence Swing (дивергенция)",
        "multitf":          "MultiTF 1H+4H   (подтверждение)",
    }

    lines = [
        f"🔬 *Анализ стратегий: {symbol}* | {datetime.utcnow().strftime('%H:%M UTC')}\n",
        f"Цена: `{fmt_price(current_price)}`  ATR: `{fmt_price(atr)}`\n",
        f"{'━'*30}",
        f"{final_emoji} *Итоговый сигнал: {final_signal}*",
        f"📊 Суммарный счёт: `{total_score:+.1f}`",
        f"{'━'*30}\n",
        f"*📋 Разбивка по стратегиям:*\n",
    ]

    for key, (sc, sigs) in strategy_results.items():
        name = strategy_names.get(key, key)
        lines.append(f"*{name}*  {fmt_strategy_score(sc)}")
        for s in sigs[:3]:  # топ-3 причины каждой стратегии
            lines.append(f"  └ {s}")
        lines.append("")

    if sl:
        lines += [
            f"{'━'*30}",
            f"🛑 *Stop Loss:* `{fmt_price(sl)}`",
            f"🎯 *TP1:* `{fmt_price(tp1)}`  (R:R 1.5)",
            f"🎯 *TP2:* `{fmt_price(tp2)}`  (R:R 3.0)",
        ]

    if tf_align:
        lines.append("\n✅ _1H и 4H тренды совпадают — сигнал надёжнее_")
    elif r4h:
        lines.append("\n⚠️ _1H и 4H тренды расходятся — входить осторожно_")

    lines.append("\n⚠️ _Это не финансовый совет. Торгуй с умом._")

    await msg.delete()
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # График 1H + 4H
    if r4h:
        buf = build_analysis_chart(symbol, df1h, df4h, r1h, r4h)
        await update.message.reply_photo(
            photo=buf,
            caption=f"📊 *{symbol}* — графики для анализа стратегий",
            parse_mode="Markdown"
        )


async def setkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        await update.message.delete()
    except:
        pass

    if len(ctx.args) != 2:
        await update.message.reply_text("Использование: `/setkey API_KEY API_SECRET`", parse_mode="Markdown")
        return

    api_key, api_secret = ctx.args[0], ctx.args[1]
    mode = USER_MODE.get(user_id, "real")
    error_detail = ""

    if mode == "demo":
        # Demo Trading ключи (demo.binance.com, вход через основной аккаунт Binance)
        # валидны на demo-fapi.binance.com, проверяем через futures-баланс (V3).
        params = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
        # ВАЖНО: строка для подписи должна быть в ТОМ ЖЕ порядке, в котором параметры
        # реально уйдут в запросе (params.items(), без sorted) — иначе подпись не совпадёт
        # с тем, что Binance пересчитает на своей стороне, и ключи будут отклонены всегда.
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": api_key}
        try:
            r = requests.get(f"{DEMO_FUTURES_API}/fapi/v3/balance", params=params, headers=headers, timeout=10)
            data = r.json()
            ok = isinstance(data, list)
            if not ok:
                error_detail = f"HTTP {r.status_code}: `{data}`"
                logger.warning(f"setkey demo check failed: status={r.status_code} body={data}")
        except Exception as e:
            logger.warning(f"setkey demo check error: {e}")
            error_detail = f"Исключение: `{e}`"
            ok = False
    else:
        # Для реального — через спотовый API, как раньше
        test = signed_request("GET", f"{BINANCE_API}/account", api_key, api_secret)
        ok = test is not None and "code" not in (test or {})
        if not ok:
            error_detail = f"`{test}`"

    if not ok:
        mode_hint = "демо (demo.binance.com — Demo Trading в твоём основном аккаунте)" if mode == "demo" else "реального Binance"
        await update.message.reply_text(
            f"❌ Неверные ключи или недостаточно прав.\n\n"
            f"Режим сейчас: *{'🧪 DEMO' if mode == 'demo' else '💰 REAL'}*\n"
            f"Убедись что ключи от {mode_hint}.\n\n"
            f"Сменить режим: `/mode demo` или `/mode real`\n\n"
            f"🔍 Ответ Binance: {error_detail or 'нет деталей'}",
            parse_mode="Markdown"
        )
        return

    USER_KEYS[user_id] = {"api_key": api_key, "api_secret": api_secret}
    mode_label = "🧪 DEMO" if mode == "demo" else "💰 REAL"
    await update.message.reply_text(
        f"✅ *API ключи подключены!* {mode_label}\nДоступны: `/balance`, `/positions`",
        parse_mode="Markdown"
    )


async def deletekey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in USER_KEYS:
        del USER_KEYS[user_id]
        await update.message.reply_text("✅ API ключи удалены.")
    else:
        await update.message.reply_text("У тебя нет сохранённых ключей.")


async def balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in USER_KEYS:
        await update.message.reply_text("❌ Сначала: `/setkey API_KEY API_SECRET`", parse_mode="Markdown")
        return

    keys = USER_KEYS[user_id]
    mode = USER_MODE.get(user_id, "real")

    if mode == "demo":
        # Demo-ключ выпущен под фьючерсы (demo-fapi) — спотового баланса там нет,
        # показываем фьючерсный баланс через тот же эндпоинт, что и /positions.
        msg = await update.message.reply_text("⏳ Получаю фьючерсный баланс (DEMO)...")
        bal = get_futures_balance(keys["api_key"], keys["api_secret"], chat_id=user_id)
        if bal is None:
            await msg.edit_text("❌ Ошибка. Проверь права API ключа (нужен Enable Futures).")
            return
        await msg.edit_text(
            f"💼 *Фьючерсный баланс (DEMO)*\n\n💰 Доступно: `${bal:,.2f} USDT`",
            parse_mode="Markdown"
        )
        return

    msg = await update.message.reply_text("⏳ Получаю баланс...")
    data = signed_request("GET", f"{BINANCE_API}/account", keys["api_key"], keys["api_secret"])

    if not data or "code" in data:
        await msg.edit_text("❌ Ошибка. Проверь права API ключа.")
        return

    balances = [b for b in data["balances"] if float(b["free"]) + float(b["locked"]) > 0]
    if not balances:
        await msg.edit_text("💼 Баланс пустой.")
        return

    lines = ["💼 *Спот баланс*\n"]
    total_usdt = 0.0
    for b in balances:
        asset = b["asset"]
        free = float(b["free"])
        locked = float(b["locked"])
        total = free + locked
        price_usdt = 1.0 if asset == "USDT" else (float((get_ticker_24h(f"{asset}USDT") or {}).get("lastPrice", 0)))
        value = total * price_usdt
        total_usdt += value
        if value > 0.01:
            lock_str = f" 🔒{locked:.4f}" if locked > 0 else ""
            lines.append(f"• *{asset}*: `{free:.4f}`{lock_str} ≈ `${value:,.2f}`")

    lines.append(f"\n💰 *Итого:* `${total_usdt:,.2f} USDT`")
    await msg.edit_text("\n".join(lines), parse_mode="Markdown")


async def positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in USER_KEYS:
        await update.message.reply_text("❌ Сначала: `/setkey API_KEY API_SECRET`", parse_mode="Markdown")
        return

    msg = await update.message.reply_text("⏳ Получаю позиции...")
    keys = USER_KEYS[user_id]

    # fapi/v2/positionRisk быстрее и надёжнее на demo чем fapi/v3/account
    data = futures_signed_request("GET", "fapi/v2/positionRisk", keys["api_key"], keys["api_secret"], chat_id=user_id)

    if not data or isinstance(data, dict) and "code" in data:
        await msg.edit_text("❌ Ошибка. Нужны права на Futures.", parse_mode="Markdown")
        return

    open_pos = [p for p in (data if isinstance(data, list) else []) if float(p.get("positionAmt", 0)) != 0]
    if not open_pos:
        await msg.edit_text("📭 Нет открытых позиций.")
        return

    lines = ["📊 *Открытые позиции*\n"]
    total_pnl = 0.0
    for p in open_pos:
        amt = float(p["positionAmt"])
        pnl = float(p.get("unrealizedProfit", 0))
        total_pnl += pnl
        side = "🟢 LONG" if amt > 0 else "🔴 SHORT"
        lines.append(f"*{p['symbol']}* {side}\n  Кол-во: `{abs(amt)}`  Вход: `{fmt_price(float(p['entryPrice']))}`  PnL: `${pnl:+,.2f}`\n")

    lines.append(f"*Общий PnL:* `${total_pnl:+,.2f} USDT`")
    await msg.edit_text("\n".join(lines), parse_mode="Markdown")


async def price(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Использование: `/price BTC`", parse_mode="Markdown")
        return
    symbol = normalize_symbol(ctx.args[0])
    msg = await update.message.reply_text(f"⏳ Получаю данные по *{symbol}*...", parse_mode="Markdown")
    data = get_ticker_24h(symbol)
    if not data or "code" in data:
        await msg.edit_text(f"❌ Монета `{symbol}` не найдена.", parse_mode="Markdown")
        return
    price_val = float(data["lastPrice"])
    change = float(data["priceChangePercent"])
    text = (
        f"💰 *{symbol}*\n\n"
        f"Цена:        `{fmt_price(price_val)}`\n"
        f"24h изм:  `{fmt_change(change)}`\n"
        f"24h макс: `{fmt_price(float(data['highPrice']))}`\n"
        f"24h мин:   `{fmt_price(float(data['lowPrice']))}`\n"
        f"Объём:      `${float(data['quoteVolume']):,.0f}`\n"
        f"Сделок:     `{int(data['count']):,}`\n"
    )
    kb = [[
        InlineKeyboardButton("📈 График", callback_data=f"chart:{symbol}"),
        InlineKeyboardButton("🔮 Фьючерсы", callback_data=f"futures:{symbol}"),
        InlineKeyboardButton("🧠 Анализ", callback_data=f"analyze:{symbol}"),
    ]]
    await msg.edit_text(text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))


async def chart(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Использование: `/chart BTC`", parse_mode="Markdown")
        return
    symbol = normalize_symbol(ctx.args[0])
    msg = await update.message.reply_text(f"📊 Строю график *{symbol}*...", parse_mode="Markdown")
    klines = get_klines(symbol, "1h", 72)
    if not klines:
        await msg.edit_text("❌ Не удалось получить данные.")
        return
    buf = build_chart(symbol, klines, "1h свечи / 3 дня")
    await msg.delete()
    await update.message.reply_photo(photo=buf, caption=f"📈 *{symbol}* — 3 дня (1h)", parse_mode="Markdown")


async def futures(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Использование: `/futures BTC`", parse_mode="Markdown")
        return
    symbol = normalize_symbol(ctx.args[0])
    msg = await update.message.reply_text(f"⏳ Получаю фьючерсные данные *{symbol}*...", parse_mode="Markdown")
    ticker = get_futures_ticker(symbol)
    oi = get_open_interest(symbol)
    funding = get_funding_rate(symbol)
    if not ticker or "code" in ticker:
        await msg.edit_text(f"❌ Фьючерс `{symbol}` не найден.", parse_mode="Markdown")
        return
    fut_price = float(ticker["lastPrice"])
    spot_data = get_ticker_24h(symbol)
    spot_price = float(spot_data["lastPrice"]) if spot_data and "code" not in spot_data else None
    basis = ((fut_price - spot_price) / spot_price * 100) if spot_price else None
    oi_usd = float(oi["openInterest"]) * fut_price if oi and "openInterest" in oi else None
    fr = float(funding["lastFundingRate"]) * 100 if funding and "lastFundingRate" in funding else None
    mark_price = float(funding["markPrice"]) if funding and "markPrice" in funding else None
    lines = [f"🔮 *{symbol} — Фьючерсы*\n",
             f"Цена: `{fmt_price(fut_price)}`  {fmt_change(float(ticker['priceChangePercent']))}"]
    if mark_price: lines.append(f"Mark: `{fmt_price(mark_price)}`")
    if basis is not None: lines.append(f"Базис: `{basis:+.4f}%`")
    if oi_usd: lines.append(f"OI: `${oi_usd:,.0f}`")
    if fr is not None: lines.append(f"Funding: `{'🟢' if fr>=0 else '🔴'} {fr:+.4f}%`")
    lines.append(f"Объём 24h: `${float(ticker['quoteVolume']):,.0f}`")
    await msg.edit_text("\n".join(lines), parse_mode="Markdown")


async def top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Загружаю топ монет...")
    gainers, losers = get_top_movers(5)
    if not gainers:
        await msg.edit_text("❌ Не удалось получить данные.")
        return
    lines = ["🏆 *Топ 5 Gainers (24h)*\n"]
    for d in gainers:
        sym = d["symbol"].replace("USDT","")
        lines.append(f"🟢 *{sym}* `{fmt_change(float(d['priceChangePercent']))}`  —  {fmt_price(float(d['lastPrice']))}")
    lines.append("\n💀 *Топ 5 Losers (24h)*\n")
    for d in losers:
        sym = d["symbol"].replace("USDT","")
        lines.append(f"🔴 *{sym}* `{fmt_change(float(d['priceChangePercent']))}`  —  {fmt_price(float(d['lastPrice']))}")
    await msg.edit_text("\n".join(lines), parse_mode="Markdown")


async def info(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Использование: `/info BTC`", parse_mode="Markdown")
        return
    symbol = normalize_symbol(ctx.args[0])
    msg = await update.message.reply_text(f"⏳ Собираю сводку по *{symbol}*...", parse_mode="Markdown")
    spot = get_ticker_24h(symbol)
    fut_ticker = get_futures_ticker(symbol)
    oi = get_open_interest(symbol)
    funding = get_funding_rate(symbol)
    if not spot or "code" in spot:
        await msg.edit_text(f"❌ Монета `{symbol}` не найдена.", parse_mode="Markdown")
        return
    price_v = float(spot["lastPrice"])
    lines = [f"📋 *{symbol} — Полная сводка*\n", "*━━ СПОТ ━━*",
             f"Цена: `{fmt_price(price_v)}`  {fmt_change(float(spot['priceChangePercent']))}",
             f"High: `{fmt_price(float(spot['highPrice']))}` Low: `{fmt_price(float(spot['lowPrice']))}`",
             f"Объём: `${float(spot['quoteVolume']):,.0f}`"]
    if fut_ticker and "code" not in fut_ticker:
        fut_price = float(fut_ticker["lastPrice"])
        basis = (fut_price - price_v) / price_v * 100
        lines += ["\n*━━ ФЬЮЧЕРСЫ ━━*",
                  f"Цена: `{fmt_price(fut_price)}`  Базис: `{basis:+.4f}%`"]
        if oi and "openInterest" in oi:
            lines.append(f"OI: `${float(oi['openInterest'])*fut_price:,.0f}`")
        if funding and "lastFundingRate" in funding:
            fr = float(funding["lastFundingRate"])*100
            lines.append(f"Funding: `{'🟢' if fr>=0 else '🔴'} {fr:+.4f}%`")
    lines.append(f"\n🕐 _{datetime.utcnow().strftime('%H:%M:%S')} UTC_")
    kb = [[
        InlineKeyboardButton("📈 График", callback_data=f"chart:{symbol}"),
        InlineKeyboardButton("🧠 Анализ", callback_data=f"analyze:{symbol}"),
    ]]
    await msg.edit_text("\n".join(lines), parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))


async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, symbol = query.data.split(":", 1)

    if action == "chart":
        await query.message.reply_text(f"📊 Строю график *{symbol}*...", parse_mode="Markdown")
        klines = get_klines(symbol, "1h", 72)
        if klines:
            buf = build_chart(symbol, klines, "1h свечи / 3 дня")
            await query.message.reply_photo(photo=buf, caption=f"📈 *{symbol}*", parse_mode="Markdown")

    elif action == "futures":
        class FU:
            message = query.message
        class FC:
            args = [symbol]
        await futures(FU(), FC())

    elif action == "analyze":
        class FU:
            message = query.message
            effective_user = query.from_user
        class FC:
            args = [symbol]
        await analyze(FU(), FC())

    elif action == "trade_confirm":
        trade_id = symbol  # здесь symbol содержит trade_id
        trade_info = PENDING_TRADES.pop(trade_id, None)
        if not trade_info:
            await query.message.edit_text("⚠️ Сигнал устарел или уже обработан.")
            return
        await query.message.edit_text(
            f"⏳ Открываю позицию *{trade_info['symbol'].replace('USDT','')}*...",
            parse_mode="Markdown"
        )
        await execute_trade(query.from_user.id, trade_info, ctx)

    elif action == "trade_cancel":
        trade_id = symbol
        PENDING_TRADES.pop(trade_id, None)
        await query.message.edit_text("❌ Сделка отменена.")

    elif action == "help":
        section = symbol  # после ":" лежит ключ раздела (main/data/analysis/...)
        try:
            await query.message.edit_text(
                HELP_SECTIONS.get(section, HELP_SECTIONS["main"]),
                parse_mode="Markdown",
                reply_markup=help_keyboard(active=section),
            )
        except Exception as e:
            # Telegram кидает BadRequest, если повторно нажали ту же кнопку
            # и текст/разметка не изменились — это не реальная ошибка, игнорируем.
            if "Message is not modified" not in str(e):
                logger.warning(f"help edit_text error: {e}")


# ─── Алерты (автоматические уведомления) ───────────────────────────────────────

async def subscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not ctx.args:
        await update.message.reply_text(
            "Использование: `/subscribe BTC ETH SOL`\n"
            "Бот будет присылать сообщение, когда по монете появится сильный сигнал ЛОНГ или ШОРТ (1H+4H).",
            parse_mode="Markdown"
        )
        return
    symbols = {normalize_symbol(s) for s in ctx.args}
    ALERT_SUBSCRIPTIONS.setdefault(chat_id, set()).update(symbols)
    await update.message.reply_text(
        f"✅ Подписка оформлена: {', '.join(s.replace('USDT','') for s in symbols)}\n"
        f"Алерт придёт при сильном сигнале (счёт ≥ {ALERT_SCORE_THRESHOLD:.0f} или ≤ -{ALERT_SCORE_THRESHOLD:.0f})."
    )


async def unsubscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not ctx.args:
        ALERT_SUBSCRIPTIONS.pop(chat_id, None)
        await update.message.reply_text("✅ Все подписки на монеты отключены.")
        return
    symbols = {normalize_symbol(s) for s in ctx.args}
    subs = ALERT_SUBSCRIPTIONS.get(chat_id, set())
    subs -= symbols
    await update.message.reply_text(
        f"✅ Отписка: {', '.join(s.replace('USDT','') for s in symbols)}"
    )


async def mysubs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subs = ALERT_SUBSCRIPTIONS.get(chat_id, set())
    market = "включена ✅" if chat_id in MARKET_SCAN_SUBSCRIBERS else "выключена ❌"
    lines = ["📋 *Твои подписки:*\n"]
    if subs:
        lines.append("Монеты: " + ", ".join(s.replace("USDT", "") for s in subs))
    else:
        lines.append("Монеты: нет")
    lines.append(f"Рыночный скан (топ лонг/шорт раз в час): {market}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def subscribe_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    MARKET_SCAN_SUBSCRIBERS.add(chat_id)
    await update.message.reply_text(
        "✅ Подписка на рыночный скан включена.\n"
        "Раз в час бот пришлёт топ кандидатов на лонг/шорт по всему рынку (если найдутся сильные сигналы)."
    )


async def unsubscribe_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    MARKET_SCAN_SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text("✅ Подписка на рыночный скан выключена.")


def check_symbol_signal(symbol):
    """Лёгкая проверка сигнала 1H+4H для алертов. Блокирующая функция."""
    try:
        k1h = get_klines(symbol, "1h", 100)
        k4h = get_klines(symbol, "4h", 100)
        if not k1h or not k4h:
            return None
        r1h = full_analysis(klines_to_df(k1h))
        r4h = full_analysis(klines_to_df(k4h))
        signal, strength, emoji, score, reasons = generate_signal_multi({"1h": r1h, "4h": r4h})
        price = klines_to_df(k1h)["close"].iloc[-1]
        return {"signal": signal, "strength": strength, "emoji": emoji,
                "score": score, "reasons": reasons[:5], "price": price}
    except Exception as e:
        logger.warning(f"check_symbol_signal {symbol} error: {e}")
        return None


async def alert_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Запускается периодически job_queue. Проверяет подписки по монетам."""
    if not ALERT_SUBSCRIPTIONS:
        return

    all_symbols = set()
    for subs in ALERT_SUBSCRIPTIONS.values():
        all_symbols.update(subs)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = dict(zip(
            all_symbols,
            await asyncio.gather(*[
                loop.run_in_executor(executor, check_symbol_signal, s) for s in all_symbols
            ])
        ))

    for chat_id, symbols in ALERT_SUBSCRIPTIONS.items():
        for symbol in symbols:
            res = results.get(symbol)
            if not res or abs(res["score"]) < ALERT_SCORE_THRESHOLD:
                continue

            key = (chat_id, symbol)
            if LAST_SIGNAL_SENT.get(key) == res["signal"]:
                continue  # уже отправляли именно этот сигнал — не спамим

            LAST_SIGNAL_SENT[key] = res["signal"]

            text = (
                f"🔔 *Алерт по {symbol.replace('USDT','')}*\n\n"
                f"{res['emoji']} *{res['signal']}* ({res['strength']})\n"
                f"Цена: `{fmt_price(res['price'])}`\n"
                f"Счёт: `{res['score']:+.1f}`\n\n"
                f"*Причины:*\n" + "\n".join(f"• {r}" for r in res["reasons"]) +
                f"\n\nИспользуй `/analyze {symbol.replace('USDT','')}` для деталей.\n"
                f"⚠️ Это не финансовый совет."
            )
            try:
                await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            except Exception as e:
                logger.warning(f"Не удалось отправить алерт {chat_id}: {e}")


async def market_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Запускается раз в час. Шлёт топ лонг/шорт подписчикам рыночного скана."""
    if not MARKET_SCAN_SUBSCRIBERS:
        return

    loop = asyncio.get_event_loop()
    longs, shorts = await loop.run_in_executor(None, scan_market)

    if not longs and not shorts:
        return

    lines = [f"🌐 *Авто-скан рынка* | {datetime.utcnow().strftime('%H:%M UTC')}\n"]
    lines.append("🟢 *Топ ЛОНГ:*")
    for r in longs[:3]:
        lines.append(f"• *{r['symbol'].replace('USDT','')}* `{r['score']:+.1f}` | {fmt_price(r['price'])}")
    lines.append("\n🔴 *Топ ШОРТ:*")
    for r in shorts[:3]:
        lines.append(f"• *{r['symbol'].replace('USDT','')}* `{r['score']:+.1f}` | {fmt_price(r['price'])}")
    lines.append("\n⚠️ Это не финансовый совет.")
    text = "\n".join(lines)

    for chat_id in list(MARKET_SCAN_SUBSCRIBERS):
        try:
            await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"Не удалось отправить авто-скан {chat_id}: {e}")


async def unknown(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❓ Неизвестная команда. Напиши /start")


async def global_error_handler(update, ctx: ContextTypes.DEFAULT_TYPE):
    logger.warning(f"Необработанная ошибка: {ctx.error}")


# ─── Автоторговля: helpers ────────────────────────────────────────────────────

def futures_signed_request(method, endpoint, api_key, api_secret, params=None, chat_id=None):
    """Подписанный запрос к Binance Futures API."""
    # ВАЖНО: делаем копию, а не мутируем словарь вызывающего кода. Иначе при
    # повторных попытках (retry) с тем же params-словарём в нём остаются
    # timestamp/recvWindow/signature от ПРОШЛОЙ попытки, которые попадают
    # в строку для подписи следующего запроса и портят сигнатуру —
    # именно это вызывало "Signature for this request is not valid" на SL/TP.
    params = dict(params) if params else {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    # Подписываем в том же порядке, в котором параметры реально уйдут в запросе
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": api_key}
    base = get_futures_api(chat_id) if chat_id else REAL_FUTURES_API
    url = f"{base}/{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == "POST":
            r = requests.post(url, params=params, headers=headers, timeout=10)
        elif method == "DELETE":
            r = requests.delete(url, params=params, headers=headers, timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = None
        if r.status_code >= 400:
            logger.error(f"Futures signed request error ({endpoint}): HTTP {r.status_code} {data}")
            # Возвращаем тело ответа биржи (там обычно код+текст ошибки Binance),
            # а не None — иначе вызывающий код теряет реальную причину сбоя.
            return data if isinstance(data, dict) else {"msg": f"HTTP {r.status_code}, нет тела ответа"}
        return data
    except Exception as e:
        logger.error(f"Futures signed request error ({endpoint}): {e}")
        return {"msg": f"Исключение запроса: {e}"}


def get_futures_balance(api_key, api_secret, chat_id=None):
    """Возвращает доступный USDT баланс на фьючерсном аккаунте."""
    mode = USER_MODE.get(chat_id, "real") if chat_id else "real"

    if mode == "demo":
        # demo-fapi поддерживает /fapi/v3/balance
        params = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": api_key}
        try:
            r = requests.get(f"{DEMO_FUTURES_API}/fapi/v3/balance", params=params, headers=headers, timeout=10)
            data = r.json()
            if isinstance(data, list):
                total = sum(float(a["availableBalance"]) for a in data if a.get("asset") in ("USDT", "USDC"))
                return total if total > 0 else None
        except Exception as e:
            logger.warning(f"get_futures_balance demo: {e}")
        return None

    data = futures_signed_request("GET", "fapi/v3/account", api_key, api_secret, chat_id=chat_id)
    if not data or "assets" not in data:
        return None
    total = sum(float(a["availableBalance"]) for a in data["assets"] if a["asset"] in ("USDT", "USDC"))
    return total if total > 0 else None


def get_symbol_info(symbol, chat_id=None):
    """Получает точность цены/количества и минимальный лот для символа."""
    try:
        base = get_futures_api(chat_id) if chat_id else REAL_FUTURES_API
        r = requests.get(f"{base}/fapi/v1/exchangeInfo", timeout=10)
        r.raise_for_status()
        for s in r.json()["symbols"]:
            if s["symbol"] == symbol:
                price_precision = s["pricePrecision"]
                qty_precision = s["quantityPrecision"]
                min_qty = None
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        min_qty = float(f["minQty"])
                return price_precision, qty_precision, min_qty
    except Exception as e:
        logger.error(f"get_symbol_info {symbol}: {e}")
    return 2, 3, 0.001


_FUTURES_SYMBOLS_CACHE: dict = {}   # base_url -> {"symbols": set, "time": float}
FUTURES_SYMBOLS_CACHE_TTL = 1800    # 30 минут — список инструментов меняется редко

def get_futures_symbols(chat_id=None):
    """
    Множество символов, у которых РЕАЛЬНО есть фьючерсный контракт на текущей
    бирже пользователя (demo или real — у них разные списки!). Кандидаты для
    автоторговли сейчас берутся из спотового тикера (get_all_usdt_symbols),
    и у части монет может не быть фьючерса вообще (только спот) — раньше это
    приводило к ошибке "Invalid symbol" уже на этапе открытия позиции.
    Кэшируется, чтобы не дёргать exchangeInfo на каждого кандидата.
    """
    base = get_futures_api(chat_id) if chat_id else REAL_FUTURES_API
    now = time.time()
    cached = _FUTURES_SYMBOLS_CACHE.get(base)
    if cached and now - cached["time"] < FUTURES_SYMBOLS_CACHE_TTL:
        return cached["symbols"]
    try:
        r = requests.get(f"{base}/fapi/v1/exchangeInfo", timeout=10)
        r.raise_for_status()
        symbols = {
            s["symbol"] for s in r.json().get("symbols", [])
            if s.get("status") == "TRADING"
        }
        _FUTURES_SYMBOLS_CACHE[base] = {"symbols": symbols, "time": now}
        return symbols
    except Exception as e:
        logger.warning(f"get_futures_symbols error: {e}")
        return cached["symbols"] if cached else set()


def set_leverage(symbol, leverage, api_key, api_secret, chat_id=None):
    return futures_signed_request("POST", "fapi/v1/leverage", api_key, api_secret,
                                   {"symbol": symbol, "leverage": leverage}, chat_id=chat_id)


def get_open_position_amt(symbol, api_key, api_secret, chat_id=None):
    """Возвращает positionAmt (float) для символа, или None если не удалось узнать."""
    data = futures_signed_request("GET", "fapi/v3/account", api_key, api_secret, chat_id=chat_id)
    if not data or "positions" not in data:
        return None
    for p in data["positions"]:
        if p.get("symbol") == symbol:
            try:
                return float(p["positionAmt"])
            except (TypeError, ValueError):
                return None
    return None


def get_open_positions_map(api_key, api_secret, chat_id=None):
    """Возвращает {symbol: positionAmt} по всем открытым фьючерсным позициям.
    Один запрос вместо N — дешевле по rate limit, чем get_open_position_amt на символ."""
    data = futures_signed_request("GET", "fapi/v3/account", api_key, api_secret, chat_id=chat_id)
    if not data or "positions" not in data:
        return {}
    out = {}
    for p in data["positions"]:
        try:
            amt = float(p.get("positionAmt", 0))
        except (TypeError, ValueError):
            continue
        if amt != 0:
            out[p["symbol"]] = amt
    return out


def get_realized_pnl(api_key, api_secret, symbol, start_ts_ms, chat_id=None):
    """Сумма реализованного PnL по закрытым сделкам по символу с момента start_ts_ms.
    Эндпоинт fapi/v1/userTrades отдаёт каждую закрытую часть позиции с realisedPnl.
    Используется для дневного лимита убытков: если позиция закрылась (SL/TP на бирже),
    бот об этом не узнаёт сам — поэтому polled-им историю сделок."""
    total = 0.0
    has_data = False
    try:
        data = futures_signed_request(
            "GET", "fapi/v1/userTrades", api_key, api_secret,
            {"symbol": symbol, "startTime": start_ts_ms, "limit": 1000},
            chat_id=chat_id,
        )
        if isinstance(data, list):
            for t in data:
                try:
                    pnl = float(t.get("realizedPnl", 0))
                    commission = float(t.get("commission", 0))
                except (TypeError, ValueError):
                    continue
                if pnl != 0:  # buyer=False записи закрытия позиции
                    total += pnl - commission
                    has_data = True
    except Exception as e:
        logger.warning(f"get_realized_pnl {symbol}: {e}")
    return total if has_data else None


def sync_open_autotrades(chat_id, api_key, api_secret):
    """Синхронизирует локальный OPEN_AUTOTRADES с реальными позициями биржи:
      - убирает закрытые на бирже позиции (и фиксирует их PnL в дневной лимит);
      - не трогает те, что ещё открыты.
    Без этого список рос бесконечно: SL/TP срабатывают на бирже, а бот об этом не знал."""
    trades = OPEN_AUTOTRADES.get(chat_id)
    if not trades:
        return

    live_positions = get_open_positions_map(api_key, api_secret, chat_id=chat_id)
    if not live_positions:
        # Запрос упал (None/пусто) — не доверяем, лучше ничего не удалять в этом цикле,
        # чем удалить реально открытую позицию. Пропускаем синхронизацию.
        return

    keys = USER_KEYS.get(chat_id)
    balance = 0.0
    if keys:
        balance = get_futures_balance(keys["api_key"], keys["api_secret"], chat_id=chat_id) or 0.0

    # Граница дня по UTC — чтобы запросить PnL только за сегодня.
    now = datetime.utcnow()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_ms = int(day_start.timestamp() * 1000)

    still_open = []
    closed_pnl_total = 0.0
    for t in trades:
        symbol = t.get("symbol")
        if symbol and symbol in live_positions:
            still_open.append(t)  # позиция ещё жива на бирже
        else:
            # Позиция закрылась на бирже — узнаём её реализованный PnL за сегодня.
            pnl = get_realized_pnl(api_key, api_secret, symbol, day_start_ms, chat_id=chat_id)
            if pnl is None:
                # Историю не получили — оставляем запись, разберёмся в следующем цикле,
                # чтобы не потерять убыточную сделку из подсчёта лимита.
                still_open.append(t)
                logger.warning(f"sync_open_autotrades {chat_id} {symbol}: не удалось получить PnL, оставляю запись")
                continue
            closed_pnl_total += pnl
            logger.info(f"sync_open_autotrades {chat_id} {symbol}: закрыта, PnL={pnl:+.2f} USDT")

    OPEN_AUTOTRADES[chat_id] = still_open

    # Фиксируем суммарный PnL всех закрывшихся в этом цикле позиций в дневной лимит.
    if closed_pnl_total != 0.0:
        record_trade_result(chat_id, closed_pnl_total, balance)


def cleanup_pending_trades(now_ts=None):
    """Удаляет неподтверждённые авто-сигналы старше PENDING_TRADE_TTL_SEC.
    Карточка в Telegram остаётся, но callback уже вернёт 'сигнал устарел'.
    Без этого PENDING_TRADES рос бесконечно при долгом аптайме."""
    if not PENDING_TRADES:
        return
    now_ts = now_ts if now_ts is not None else time.time()
    expired = [
        tid for tid, t in PENDING_TRADES.items()
        if now_ts - t.get("created_at", now_ts) > PENDING_TRADE_TTL_SEC
    ]
    for tid in expired:
        PENDING_TRADES.pop(tid, None)
    if expired:
        logger.info(f"cleanup_pending_trades: удалено {len(expired)} устаревших сигналов")


def _move_sl_order(symbol, old_order_id, new_sl_price, close_side, qty, qty_precision,
                   price_precision, api_key, api_secret, chat_id=None):
    """Переставляет SL-ордер: отменяет старый, ставит новый STOP_MARKET на new_sl_price.
    Возвращает (ok: bool, new_order_id). Если отмена/установка упали — возвращает (False, None),
    старый ордер при этом мог остаться висеть (это безопаснее, чем остаться без стопа)."""
    base_url = get_futures_api(chat_id)

    def _cancel(oid):
        params = {"symbol": symbol, "orderId": oid,
                  "timestamp": int(time.time() * 1000), "recvWindow": 5000}
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        params["signature"] = hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        try:
            requests.delete(f"{base_url}/fapi/v1/order",
                            params=params, headers={"X-MBX-APIKEY": api_key}, timeout=10)
        except Exception as e:
            logger.warning(f"_move_sl_order cancel {symbol} oid={oid}: {e}")

    def _place():
        params = {
            "symbol": symbol, "side": close_side, "type": "STOP_MARKET",
            "quantity": round(qty, qty_precision),
            "stopPrice": round(new_sl_price, price_precision),
            "positionSide": "BOTH", "reduceOnly": "true", "workingType": "MARK_PRICE",
            "timestamp": int(time.time() * 1000), "recvWindow": 5000,
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        params["signature"] = hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        r = requests.post(f"{base_url}/fapi/v1/order",
                          params=params, headers={"X-MBX-APIKEY": api_key}, timeout=10)
        return r.json()

    if old_order_id:
        _cancel(old_order_id)

    # Небольшой retry на постановку нового — биржа иногда 429'ит.
    for attempt in range(3):
        try:
            data = _place()
            if "orderId" in data:
                return True, data["orderId"]
            logger.warning(f"_move_sl_order place {symbol} попытка {attempt+1}: {data}")
        except Exception as e:
            logger.warning(f"_move_sl_order place {symbol} попытка {attempt+1}: {e}")
        time.sleep(1.0)
    return False, None


def trailing_stop_manage_one(trade, api_key, api_secret, chat_id):
    """Обрабатывает одну позицию из OPEN_AUTOTRADES:
       1) если TP1 ещё не сработал — ничего не делаем;
       2) когда TP1 сработал (позиция уменьшилась до ~qty_half) — переставляем SL на б/у;
       3) после б/у — трейлим SL за ценой на ATR_TRAIL_MULT × ATR.
    Возвращает обновлённый trade (или None, если позиция исчезла с биржи — её уберёт
    sync_open_autotrades). Все сетевые вызовы идут синхронно — функцию дёргают через
    run_in_executor, чтобы не блокировать event loop."""
    symbol = trade.get("symbol")
    direction = trade.get("direction") or ("long" if trade.get("qty", 0) > 0 else "short")
    entry = trade.get("entry")
    if not symbol or entry is None:
        return trade

    pos_amt = get_open_position_amt(symbol, api_key, api_secret, chat_id=chat_id)
    if pos_amt is None or abs(pos_amt) < 1e-9:
        # Позиция закрылась целиком — sync_open_autotrades её уберёт, тут только выходим.
        return None
    if abs(pos_amt) > abs(trade.get("qty", 0)) * 0.75 and not trade.get("tp1_filled"):
        # TP1 ещё не сработал — позиция ещё полного размера, трейлинг рано.
        return trade

    close_side = "SELL" if direction == "long" else "BUY"

    # Шаг 1: первый раз замечаем, что TP1 сработал → переводим SL на безубыток.
    if not trade.get("be_set"):
        be_buffer = trade.get("atr_at_open", 0) * 0.15  # небольшой буфер поверх entry
        if direction == "long":
            new_sl = entry + be_buffer
        else:
            new_sl = entry - be_buffer
        ok, new_oid = _move_sl_order(
            symbol, trade.get("sl_order_id"), new_sl, close_side,
            abs(pos_amt), trade.get("qty_precision", 3), trade.get("price_precision", 2),
            api_key, api_secret, chat_id=chat_id,
        )
        if ok:
            trade["sl_order_id"] = new_oid
            trade["be_set"] = True
            trade["trail_sl"] = new_sl
            trade["tp1_filled"] = True
            logger.info(f"trailing {chat_id} {symbol}: TP1 сработал, SL переведён в б/у ({new_sl})")
        return trade

    # Шаг 2: трейлинг — двигаем SL только в прибыльную сторону.
    klines = get_klines(symbol, "1h", 30)
    if not klines or len(klines) < 15:
        return trade
    df = klines_to_df(klines)
    atr_now = calc_atr(df, 14).iloc[-1]
    last_price = df["close"].iloc[-1]
    trail_dist = atr_now * ATR_TRAIL_MULT

    if direction == "long":
        new_sl = last_price - trail_dist
        if new_sl > trade["trail_sl"]:
            ok, new_oid = _move_sl_order(
                symbol, trade.get("sl_order_id"), new_sl, close_side,
                abs(pos_amt), trade.get("qty_precision", 3), trade.get("price_precision", 2),
                api_key, api_secret, chat_id=chat_id,
            )
            if ok:
                trade["sl_order_id"] = new_oid
                trade["trail_sl"] = new_sl
                logger.info(f"trailing {chat_id} {symbol}: SL поднят до {new_sl}")
    else:
        new_sl = last_price + trail_dist
        if new_sl < trade["trail_sl"]:
            ok, new_oid = _move_sl_order(
                symbol, trade.get("sl_order_id"), new_sl, close_side,
                abs(pos_amt), trade.get("qty_precision", 3), trade.get("price_precision", 2),
                api_key, api_secret, chat_id=chat_id,
            )
            if ok:
                trade["sl_order_id"] = new_oid
                trade["trail_sl"] = new_sl
                logger.info(f"trailing {chat_id} {symbol}: SL опущен до {new_sl}")
    return trade


async def trailing_stop_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Фоновый job: раз в 2 минуты проверяет открытые авто-позиции на предмет
    перевода в б/у после TP1 и последующего трейлинга. SL/TP на бирже фиксированы,
    без этой логики вторая половина позиции (до TP2) отдаёт всю прибыль при откате."""
    for chat_id in list(OPEN_AUTOTRADES.keys()):
        keys = USER_KEYS.get(chat_id)
        if not keys:
            continue
        trades = OPEN_AUTOTRADES.get(chat_id, [])
        if not trades:
            continue
        loop = asyncio.get_running_loop()
        updated = []
        for trade in trades:
            try:
                res = await loop.run_in_executor(
                    None, trailing_stop_manage_one, trade,
                    keys["api_key"], keys["api_secret"], chat_id,
                )
                if res is not None:
                    updated.append(res)
            except Exception as e:
                logger.warning(f"trailing_stop_job {chat_id} {trade.get('symbol')}: {e}")
                updated.append(trade)  # оставляем как есть, разберёмся в следующем цикле
        OPEN_AUTOTRADES[chat_id] = updated


def place_futures_market_order(symbol, side, quantity, qty_precision, api_key, api_secret, chat_id=None):
    """Открывает рыночный ордер на фьючерсах."""
    qty = round(quantity, qty_precision)
    return futures_signed_request("POST", "fapi/v1/order", api_key, api_secret, {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "positionSide": "BOTH",
    }, chat_id=chat_id)


def place_sl_tp_orders(symbol, direction, sl_price, tp_price, quantity,
                        price_precision, qty_precision, api_key, api_secret, chat_id=None,
                        attempts=3, retry_delay=1.5):
    """Ставит STOP_MARKET (SL) и TAKE_PROFIT_MARKET (TP) ордера.
    Делает несколько попыток на случай временного сбоя биржи (важно — без
    SL позиция остаётся полностью незащищённой)."""
    close_side = "SELL" if direction == "long" else "BUY"
    qty = round(quantity, qty_precision)
    sl_params = {
        "symbol": symbol, "side": close_side, "type": "STOP_MARKET",
        "quantity": qty, "stopPrice": round(sl_price, price_precision),
        "positionSide": "BOTH", "reduceOnly": "true", "workingType": "MARK_PRICE",
    }
    tp_params = {
        "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "quantity": qty, "stopPrice": round(tp_price, price_precision),
        "positionSide": "BOTH", "reduceOnly": "true", "workingType": "MARK_PRICE",
    }

    def _place_with_retry(params, label):
        last_res = None
        for attempt in range(1, attempts + 1):
            res = futures_signed_request("POST", "fapi/v1/order", api_key, api_secret, params, chat_id=chat_id)
            if res and "orderId" in res:
                return res
            last_res = res
            logger.warning(f"{label} попытка {attempt}/{attempts} не удалась: {res}")
            if attempt < attempts:
                time.sleep(retry_delay)
        return last_res

    sl_res = _place_with_retry(sl_params, f"SL {symbol}")
    tp_res = _place_with_retry(tp_params, f"TP {symbol}")
    return sl_res, tp_res


async def execute_trade(chat_id, trade_info, ctx):
    """Реально открывает позицию на Binance Futures после подтверждения пользователя."""
    keys = USER_KEYS.get(chat_id)
    if not keys:
        await ctx.bot.send_message(chat_id, "❌ API ключи не найдены.")
        return

    symbol    = trade_info["symbol"]
    direction = trade_info["direction"]
    sl        = trade_info["sl"]
    tp1       = trade_info["tp1"]
    tp2       = trade_info.get("tp2", tp1)
    entry     = trade_info["entry"]
    mode      = USER_MODE.get(chat_id, "real")
    mode_label = "🧪 DEMO" if mode == "demo" else "💰 REAL"

    balance = get_futures_balance(keys["api_key"], keys["api_secret"], chat_id=chat_id)
    if not balance or balance < 10:
        await ctx.bot.send_message(chat_id, "❌ Недостаточно средств (минимум $10).")
        return

    # ── Проверка дневного лимита убытков ──────────────────────────────────────
    allowed, pause_msg = check_daily_loss_limit(chat_id, balance)
    if not allowed:
        await ctx.bot.send_message(chat_id, f"🛑 *Дневной лимит убытков*\n\n{pause_msg}", parse_mode="Markdown")
        return

    risk_pct  = AUTOTRADE_RISK_PCT.get(chat_id, 1.0)
    risk_usdt = balance * risk_pct / 100
    price_precision, qty_precision, min_qty = get_symbol_info(symbol, chat_id=chat_id)

    atr_risk_price = abs(entry - sl)
    if atr_risk_price <= 0:
        await ctx.bot.send_message(chat_id, "❌ Ошибка расчёта риска (SL = цена входа).")
        return

    qty_total = risk_usdt / atr_risk_price
    qty_total = max(round(qty_total, qty_precision), min_qty or 0.001)

    # ── Частичное закрытие: 50% на TP1, 50% на TP2 ──────────────────────────
    qty_half = max(round(qty_total / 2, qty_precision), min_qty or 0.001)

    set_leverage(symbol, DEFAULT_LEVERAGE, keys["api_key"], keys["api_secret"], chat_id=chat_id)

    side  = "BUY" if direction == "long" else "SELL"
    order = place_futures_market_order(symbol, side, qty_total, qty_precision,
                                       keys["api_key"], keys["api_secret"], chat_id=chat_id)

    if not order or "orderId" not in order:
        err = (order or {}).get("msg", "нет ответа от биржи")
        await ctx.bot.send_message(chat_id, f"❌ Ошибка открытия позиции: `{err}`", parse_mode="Markdown")
        return

    fill_price = float(order.get("avgPrice") or entry)
    if fill_price == 0:
        fill_price = entry

    # Ждём подтверждения позиции биржей
    loop = asyncio.get_event_loop()
    position_confirmed = False
    for _ in range(5):
        await asyncio.sleep(1.0)
        pos_amt = await loop.run_in_executor(
            None, get_open_position_amt, symbol, keys["api_key"], keys["api_secret"], chat_id
        )
        if pos_amt is not None and abs(pos_amt) > 0:
            position_confirmed = True
            break

    if not position_confirmed:
        logger.warning(f"execute_trade {symbol}: позиция не подтвердилась за 5 попыток")

    # ── SL + TP1 + TP2 с retry ───────────────────────────────────────────────
    close_side = "SELL" if direction == "long" else "BUY"

    def _place_with_retry(params_base, label, attempts=4, delay=1.5):
        """Каждая попытка генерирует свежий timestamp — именно это исправляет
        'Signature for this request is not valid' при повторных вызовах."""
        last_res = None
        for attempt in range(1, attempts + 1):
            params = dict(params_base)
            params["timestamp"]  = int(time.time() * 1000)
            params["recvWindow"] = 5000
            qs  = "&".join(f"{k}={v}" for k, v in params.items())
            sig = hmac.new(keys["api_secret"].encode(), qs.encode(), hashlib.sha256).hexdigest()
            params["signature"] = sig
            headers = {"X-MBX-APIKEY": keys["api_key"]}
            base_url = get_futures_api(chat_id)
            try:
                r = requests.post(f"{base_url}/fapi/v1/order",
                                  params=params, headers=headers, timeout=10)
                data = r.json()
                if "orderId" in data:
                    return data
                last_res = data
                logger.warning(f"{label} попытка {attempt}/{attempts}: {data}")
            except Exception as e:
                last_res = {"msg": str(e)}
                logger.warning(f"{label} попытка {attempt}/{attempts} исключение: {e}")
            if attempt < attempts:
                time.sleep(delay)
        return last_res

    sl_base = {
        "symbol": symbol, "side": close_side, "type": "STOP_MARKET",
        "stopPrice": round(sl, price_precision),
        "positionSide": "BOTH", "closePosition": "true", "workingType": "MARK_PRICE",
    }
    tp1_base = {
        "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "quantity": round(qty_half, qty_precision),
        "stopPrice": round(tp1, price_precision),
        "positionSide": "BOTH", "reduceOnly": "true", "workingType": "MARK_PRICE",
    }
    tp2_base = {
        "symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET",
        "stopPrice": round(tp2, price_precision),
        "positionSide": "BOTH", "closePosition": "true", "workingType": "MARK_PRICE",
    }

    sl_res  = await loop.run_in_executor(None, _place_with_retry, sl_base,  f"SL  {symbol}")
    tp1_res = await loop.run_in_executor(None, _place_with_retry, tp1_base, f"TP1 {symbol}")
    tp2_res = await loop.run_in_executor(None, _place_with_retry, tp2_base, f"TP2 {symbol}")

    sl_ok  = sl_res  and "orderId" in sl_res
    tp1_ok = tp1_res and "orderId" in tp1_res
    tp2_ok = tp2_res and "orderId" in tp2_res

    def _err(res):
        msg = (res or {}).get("msg", "нет ответа")
        return str(msg).replace("`","").replace("*","").replace("_"," ")

    OPEN_AUTOTRADES.setdefault(chat_id, []).append({
        **trade_info,
        "entry":       fill_price,
        "qty":         qty_total,
        "qty_half":    qty_half,
        "order_id":    order["orderId"],
        "sl_order_id": sl_res.get("orderId")  if sl_ok  else None,
        "tp1_order_id":tp1_res.get("orderId") if tp1_ok else None,
        "tp2_order_id":tp2_res.get("orderId") if tp2_ok else None,
        "opened_at":   datetime.utcnow().isoformat(),
        # Состояние трейлинг-логики (обрабатывается trailing_stop_job):
        "tp1_filled":  False,                 # сработал ли TP1 (половина закрыта)
        "be_set":      False,                 # переведён ли стоп в безубыток
        "trail_sl":    sl,                    # текущий уровень трейлинг-стопа
        "atr_at_open": atr,                    # ATR на момент входа — для дистанции trail
    })

    dir_label = "🟢 LONG" if direction == "long" else "🔴 SHORT"
    notional  = qty_total * fill_price / DEFAULT_LEVERAGE
    daily_loss = AUTOTRADE_DAILY_LOSS.get(chat_id, 0.0)
    limit_usdt = balance * DAILY_LOSS_LIMIT_PCT / 100

    await ctx.bot.send_message(
        chat_id,
        f"✅ *Позиция открыта!* {mode_label}\n\n"
        f"{dir_label} *{symbol.replace('USDT', '')}*\n"
        f"💵 Вход: `{fmt_price(fill_price)}`\n"
        f"📦 Объём: `{qty_total}` (~`${notional:,.2f} USDT` маржи)\n"
        f"⚖️ Плечо: `x{DEFAULT_LEVERAGE}`\n\n"
        f"🛑 SL (100%): `{fmt_price(sl)}` {'✅' if sl_ok else '⚠️ ' + _err(sl_res)}\n"
        f"🎯 TP1 (50%): `{fmt_price(tp1)}` {'✅' if tp1_ok else '⚠️ ' + _err(tp1_res)}\n"
        f"🎯 TP2 (50%): `{fmt_price(tp2)}` {'✅' if tp2_ok else '⚠️ ' + _err(tp2_res)}\n\n"
        f"🛡 Дневной убыток: `${daily_loss:.2f}` / `${limit_usdt:.2f}`\n\n"
        f"_50% позиции закроется на TP1, 50% — на TP2_\n"
        f"Используй `/autoportfolio` для отслеживания.\n"
        f"⚠️ _Торговля связана с риском потери средств._",
        parse_mode="Markdown"
    )


# ─── Автоторговля: команды ────────────────────────────────────────────────────

async def autotrade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /autotrade on [risk%]  — включить (риск в % от баланса, по умолчанию 1%)
    /autotrade off         — выключить
    /autotrade status      — текущий статус
    """
    chat_id = update.effective_chat.id
    args = ctx.args

    if not args:
        await update.message.reply_text(
            "📟 *Автоторговля (фьючерсы)*\n\n"
            "• `/autotrade on` — включить (риск 1% на сделку)\n"
            "• `/autotrade on 2` — включить с риском 2%\n"
            "• `/autotrade off` — выключить\n"
            "• `/autotrade status` — текущий статус\n\n"
            "⚠️ Требует API ключ с правом *Futures Trading*.\n"
            "Добавь через `/setkey` если ещё не сделал.",
            parse_mode="Markdown"
        )
        return

    cmd = args[0].lower()

    if cmd == "off":
        AUTOTRADE_ENABLED[chat_id] = False
        await update.message.reply_text("🔴 Автоторговля выключена.")
        return

    if cmd == "status":
        enabled = AUTOTRADE_ENABLED.get(chat_id, False)
        risk = AUTOTRADE_RISK_PCT.get(chat_id, 1.0)
        open_trades = OPEN_AUTOTRADES.get(chat_id, [])
        pending = sum(1 for t in PENDING_TRADES.values() if True)  # все pending
        lines = [
            f"📟 *Статус автоторговли*\n",
            f"Состояние: {'🟢 Включена' if enabled else '🔴 Выключена'}",
            f"Риск на сделку: `{risk}%`",
            f"Плечо: `x{DEFAULT_LEVERAGE}`",
            f"Порог сигнала: `≥ {AUTOTRADE_SCORE_THRESHOLD}` из ±15",
            f"Открытых авто-позиций: `{len(open_trades)}`",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    if cmd == "on":
        if chat_id not in USER_KEYS:
            await update.message.reply_text(
                "❌ Сначала добавь API ключ: `/setkey API_KEY API_SECRET`\n"
                "Ключ должен иметь право *Futures Trading* (не Withdrawal!).",
                parse_mode="Markdown"
            )
            return

        risk = 1.0
        if len(args) > 1:
            try:
                risk = float(args[1])
                if not (0.1 <= risk <= 5):
                    await update.message.reply_text("❌ Риск должен быть от 0.1% до 5%.")
                    return
            except ValueError:
                await update.message.reply_text(
                    "❌ Неверный формат. Пример: `/autotrade on 1.5`",
                    parse_mode="Markdown"
                )
                return

        keys = USER_KEYS[chat_id]
        bal = get_futures_balance(keys["api_key"], keys["api_secret"], chat_id=chat_id)
        if bal is None:
            await update.message.reply_text(
                "❌ Не удалось получить фьючерсный баланс.\n"
                "Убедись что ключ имеет право *Futures Trading*.",
                parse_mode="Markdown"
            )
            return

        mode = USER_MODE.get(chat_id, "real")
        mode_label = "🧪 DEMO (testnet)" if mode == "demo" else "💰 REAL (реальные деньги)"
        AUTOTRADE_ENABLED[chat_id] = True
        AUTOTRADE_RISK_PCT[chat_id] = risk

        await update.message.reply_text(
            f"✅ *Автоторговля включена*\n\n"
            f"🌐 Режим: *{mode_label}*\n"
            f"💰 Баланс фьючерсов: `${bal:,.2f} USDT`\n"
            f"⚖️ Риск на сделку: `{risk}%` ≈ `${bal * risk / 100:,.2f} USDT`\n"
            f"🔢 Плечо: `x{DEFAULT_LEVERAGE}`\n"
            f"📊 Порог сигнала: score `≥ {AUTOTRADE_SCORE_THRESHOLD}` (из ±15)\n"
            f"🔄 Скан каждые 10 мин\n\n"
            f"При сильном сигнале бот пришлёт карточку с кнопками "
            f"✅ *Войти* / ❌ *Отмена*.\n\n"
            f"⚠️ _Это реальные сделки на реальные деньги. "
            f"Ты несёшь полную ответственность за результат._",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text(
        "❓ Неизвестная команда.\nИспользуй: `/autotrade on/off/status`",
        parse_mode="Markdown"
    )


async def autoportfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/autoportfolio — список открытых авто-позиций с live PnL."""
    chat_id = update.effective_chat.id
    trades = OPEN_AUTOTRADES.get(chat_id, [])
    if not trades:
        await update.message.reply_text("📭 Нет открытых авто-позиций.")
        return

    if chat_id not in USER_KEYS:
        await update.message.reply_text("❌ Ключи не найдены. `/setkey API_KEY API_SECRET`", parse_mode="Markdown")
        return

    keys = USER_KEYS[chat_id]
    data = futures_signed_request("GET", "fapi/v3/account", keys["api_key"], keys["api_secret"], chat_id=chat_id)
    live_pnl = {}
    if data and "positions" in data:
        for p in data["positions"]:
            if float(p.get("positionAmt", 0)) != 0:
                live_pnl[p["symbol"]] = float(p["unrealizedProfit"])

    lines = ["📊 *Авто-позиции*\n"]
    total_pnl = 0.0
    for t in trades:
        sym = t["symbol"]
        direction = "🟢 LONG" if t["direction"] == "long" else "🔴 SHORT"
        pnl = live_pnl.get(sym, 0.0)
        total_pnl += pnl
        pnl_str = f"📈 +${pnl:,.2f}" if pnl >= 0 else f"📉 -${abs(pnl):,.2f}"
        lines.append(
            f"*{sym.replace('USDT','')}* {direction}\n"
            f"  Вход: `{fmt_price(t['entry'])}` | "
            f"SL: `{fmt_price(t['sl'])}` | TP: `{fmt_price(t['tp1'])}`\n"
            f"  PnL: `{pnl_str}`\n"
        )
    pnl_total_str = f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"
    lines.append(f"*Итого PnL:* `{pnl_total_str} USDT`")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ─── Автоторговля: фоновый сканер ────────────────────────────────────────────

async def mode_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/mode [demo|real] — переключить режим торговли между testnet и реальным Binance."""
    chat_id = update.effective_chat.id
    current = USER_MODE.get(chat_id, "real")

    if not ctx.args:
        label = "🧪 DEMO (demo.binance.com)" if current == "demo" else "💰 REAL (реальный Binance)"
        await update.message.reply_text(
            f"🌐 *Текущий режим:* {label}\n\n"
            f"• `/mode demo` — переключить на демо (Demo Trading, виртуальные деньги)\n"
            f"• `/mode real` — переключить на реальный Binance\n\n"
            f"⚠️ При смене режима нужно ввести `/setkey` с ключами для нового режима.\n"
            f"Для демо ключи берутся на *demo.binance.com* (вход через твой обычный аккаунт Binance).",
            parse_mode="Markdown"
        )
        return

    mode = ctx.args[0].lower()
    if mode not in ("demo", "real"):
        await update.message.reply_text("❌ Используй: `/mode demo` или `/mode real`", parse_mode="Markdown")
        return

    USER_MODE[chat_id] = mode
    # Старый ключ почти наверняка не подходит для нового режима — убираем,
    # чтобы не было ситуации "real-ключ тихо используется на demo-эндпоинте".
    USER_KEYS.pop(chat_id, None)

    if mode == "demo":
        await update.message.reply_text(
            "🧪 *Режим DEMO включён*\n\n"
            "Бот будет использовать *demo-fapi.binance.com* (Demo Trading).\n\n"
            "Как получить демо-ключи:\n"
            "1. Зайди на *demo.binance.com* (или включи Demo Trading в основном приложении/сайте Binance)\n"
            "2. Войди обычным аккаунтом Binance (тот же логин, что и на binance.com)\n"
            "3. Открой *API Management* → *Create API* → разреши *Enable Reading* и *Enable Futures*\n"
            "4. Введи в боте: `/setkey API\\_KEY SECRET`\n\n"
            "Баланс — виртуальный, выдаётся/сбрасывается в интерфейсе Demo Trading.\n"
            "Старый ключ (если был) удалён — нужно подключить новый.",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "💰 *Режим REAL включён*\n\n"
            "Бот будет использовать реальный *Binance Futures*.\n"
            "Введи реальные API ключи через `/setkey`.\n\n"
            "⚠️ _Все сделки будут на реальные деньги._\n"
            "Старый ключ (если был) удалён — нужно подключить новый.",
            parse_mode="Markdown"
        )


async def autotrade_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Каждые 10 мин ищет сигналы. Комбинирует скан + 5 стратегий +
    BTC-режим фильтр + Funding Rate (только экстремальные) + дневной лимит."""
    import uuid
    active_users = [cid for cid, on in AUTOTRADE_ENABLED.items() if on]
    if not active_users:
        return

    # ── Получаем BTC режим один раз для всех ──────────────────────────────────
    loop = asyncio.get_running_loop()
    btc_trend = await loop.run_in_executor(None, get_btc_regime)
    logger.info(f"autotrade_scan_job: BTC trend = {btc_trend}")

    longs, shorts = await loop.run_in_executor(None, scan_market)
    candidates = longs + shorts

    # Чистим устаревшие неподтверждённые авто-сигналы (иначе PENDING_TRADES
    # растёт бесконечно при долгом аптайме — карточка остаётся, но callback
    # уже вернёт «сигнал устарел»).
    cleanup_pending_trades()

    for chat_id in active_users:
        if chat_id not in USER_KEYS:
            continue

        keys    = USER_KEYS[chat_id]
        balance = get_futures_balance(keys["api_key"], keys["api_secret"], chat_id=chat_id) or 0

        # ── Синхронизация локальных позиций с биржей ─────────────────────────
        # SL/TP срабатывают на бирже сами, бот об этом не знает — без этой
        # проверки OPEN_AUTOTRADES[chat_id] рос бесконечно, и символы, по
        # которым позиция уже закрылась, навсегда блокировали новые входы.
        # Заодно зафиксируем реализованный PnL закрывшихся сделок в дневной лимит.
        await loop.run_in_executor(
            None, sync_open_autotrades, chat_id, keys["api_key"], keys["api_secret"]
        )
        # Баланс мог измениться после учёта закрытых позиций — перечитываем.
        balance = get_futures_balance(keys["api_key"], keys["api_secret"], chat_id=chat_id) or 0

        # ── Дневной лимит убытков ─────────────────────────────────────────────
        allowed, pause_msg = check_daily_loss_limit(chat_id, balance)
        if not allowed:
            logger.info(f"autotrade {chat_id}: дневной лимит — пропускаем скан")
            try:
                await ctx.bot.send_message(chat_id, f"🛑 *Дневной лимит убытков*\n\n{pause_msg}",
                                           parse_mode="Markdown")
            except Exception:
                pass
            continue

        open_symbols = {t["symbol"] for t in OPEN_AUTOTRADES.get(chat_id, [])}
        futures_symbols = get_futures_symbols(chat_id=chat_id)

        # Собираем ВСЕХ кандидатов, прошедших фильтры, а не берём первого
        # попавшегося — раньше longs всегда шли первыми в списке candidates,
        # и из-за break после первого же подходящего лонга шорты в этом же
        # цикле часто даже не успевали проверяться.
        qualified = []

        for candidate in candidates:
            symbol     = candidate["symbol"]
            quick_score = candidate["score"]

            if symbol in open_symbols:
                continue
            # Кандидаты приходят из спотового скана — у части монет может не
            # быть фьючерсного контракта вообще (или его нет на demo), и тогда
            # открытие позиции упало бы с "Invalid symbol" уже на бирже.
            if futures_symbols and symbol not in futures_symbols:
                logger.info(f"autotrade {symbol}: нет фьючерсного контракта на текущей бирже ({USER_MODE.get(chat_id, 'real')}) — пропускаем")
                continue
            if abs(quick_score) < AUTOTRADE_SCORE_THRESHOLD:
                continue

            direction_quick = "long" if quick_score > 0 else "short"

            # ── BTC режим: блокируем лонги по альтам при сильном даунтренде ──
            if symbol != "BTCUSDT":
                if direction_quick == "long" and btc_trend == "strong_down":
                    logger.info(f"autotrade {symbol}: лонг заблокирован — BTC в strong_down")
                    continue
                if direction_quick == "short" and btc_trend == "strong_up":
                    logger.info(f"autotrade {symbol}: шорт заблокирован — BTC в strong_up")
                    continue

            # ── Funding Rate (только экстремальные значения) ──────────────────
            fr_pct, fr_signal = get_funding_signal(symbol)
            fr_note = ""
            if fr_signal is not None:
                if fr_signal != direction_quick:
                    # Funding явно против нашего направления — блокируем
                    logger.info(
                        f"autotrade {symbol}: funding {fr_pct:+.4f}% против направления {direction_quick} — пропускаем"
                    )
                    continue
                else:
                    # Funding подтверждает направление — бонус к счёту
                    fr_note = f"⚡ Funding Rate экстремальный ({fr_pct:+.4f}%) подтверждает {direction_quick.upper()}"

            # ── Детальный анализ 1H + 4H + все стратегии ─────────────────────
            k1h = get_klines(symbol, "1h", 200)
            k4h = get_klines(symbol, "4h", 200)
            if not k1h:
                continue

            df1h = klines_to_df(k1h)
            r1h  = full_analysis(df1h)
            df4h = r4h = None
            if k4h and len(k4h) >= 50:
                df4h = klines_to_df(k4h)
                r4h  = full_analysis(df4h)

            strat_results, strat_total, tf_align = run_all_strategies(df1h, r1h, df4h, r4h)

            mr_score,  _ = strat_results.get("mean_reversion",   (0, []))
            tp_score,  _ = strat_results.get("trend_pullback",    (0, []))
            bo_score,  _ = strat_results.get("breakout",          (0, []))
            ds_score,  _ = strat_results.get("divergence_swing",  (0, []))
            mtf_score, _ = strat_results.get("multitf",           (0, []))

            combined   = quick_score * 1.0 + strat_total * 0.5

            # ── Мульти-ТФ фильтр: смягчённая версия ────────────────────────────
            # Раньше направление стратегий обязано было СТРОГО совпадать с quick,
            # иначе сигнал дропался — это резало краевые (|strat_total|≈0) сигналы.
            # Теперь:
            #   - блокируем ТОЛЬКО при явном конфликте (strat_total заметно
            #     противоположен quick) — |strat_total| >= 1.0 в обратную сторону;
            #   - при нейтральных стратегиях (|strat_total| < 1.0) — пропускаем;
            #   - при совпадении — небольшой бонус к combined.
            strat_opp = (quick_score > 0 and strat_total <= -1.0) or \
                        (quick_score < 0 and strat_total >= 1.0)
            if strat_opp:
                logger.info(f"autotrade {symbol}: явный конфликт скан={direction_quick} vs стратегии total={strat_total:+.1f} — пропускаем")
                continue

            if (quick_score > 0 and strat_total >= 1.0) or (quick_score < 0 and strat_total <= -1.0):
                combined += 0.5  # бонус за подтверждение направления стратегиями

            if abs(combined) < AUTOTRADE_SCORE_THRESHOLD:
                continue

            direction = direction_quick
            price = df1h["close"].iloc[-1]
            atr   = calc_atr(df1h, 14).iloc[-1]
            # SL по структуре рынка (swing ± ATR-cap), не голый ATR*1.5 —
            # tighter стоп реже выбивается шумом, а cap не даёт раздуть риск.
            sl, risk = calc_structural_sl(price, atr, direction, df1h)

            if direction == "long":
                tp1, tp2 = price + risk * 1.5, price + risk * 3.0
                dir_label = "🟢 LONG"
            else:
                tp1, tp2 = price - risk * 1.5, price - risk * 3.0
                dir_label = "🔴 SHORT"

            strat_scores = [mr_score, tp_score, bo_score, ds_score, mtf_score]
            agree_count  = sum(
                1 for s in strat_scores
                if (direction == "long" and s > 0) or (direction == "short" and s < 0)
            )

            qualified.append({
                "symbol": symbol, "direction": direction, "dir_label": dir_label,
                "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
                "quick_score": quick_score, "strat_total": strat_total, "combined": combined,
                "agree_count": agree_count,
                "mr_score": mr_score, "tp_score": tp_score, "bo_score": bo_score,
                "ds_score": ds_score, "mtf_score": mtf_score,
                "tf_align": tf_align, "fr_note": fr_note,
            })

        if not qualified:
            continue

        # Выбираем сигнал с максимальным |combined| среди всех прошедших
        # фильтры — а не первый по порядку (раньше это всегда были лонги).
        best = max(qualified, key=lambda q: abs(q["combined"]))

        symbol, direction, dir_label = best["symbol"], best["direction"], best["dir_label"]
        price, sl, tp1, tp2 = best["price"], best["sl"], best["tp1"], best["tp2"]
        quick_score, strat_total, combined = best["quick_score"], best["strat_total"], best["combined"]
        agree_count = best["agree_count"]
        mr_score, tp_score, bo_score, ds_score, mtf_score = (
            best["mr_score"], best["tp_score"], best["bo_score"], best["ds_score"], best["mtf_score"]
        )
        tf_align, fr_note = best["tf_align"], best["fr_note"]

        trade_id = str(uuid.uuid4())[:8]
        PENDING_TRADES[trade_id] = {
            "symbol": symbol, "direction": direction,
            "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2,
            "score": combined,
            "created_at": time.time(),
        }

        risk_pct  = AUTOTRADE_RISK_PCT.get(chat_id, 1.0)
        risk_usdt = balance * risk_pct / 100
        mode      = USER_MODE.get(chat_id, "real")
        mode_label = "🧪 DEMO" if mode == "demo" else "💰 REAL"

        # BTC статус для карточки
        btc_map = {
            "strong_up": "🚀 Сильный рост", "up": "📈 Рост",
            "strong_down": "💥 Сильное падение", "down": "📉 Падение", "neutral": "⚪ Боковик"
        }
        btc_note = f"BTC: {btc_map.get(btc_trend, btc_trend)}"
        tf_note  = "✅ 1H и 4H совпадают" if tf_align else "⚠️ 1H/4H расходятся"

        def strat_line(name, sc):
            if (direction == "long" and sc > 0) or (direction == "short" and sc < 0):
                return f"  ✅ {name}: `{sc:+.1f}`"
            elif sc == 0:
                return f"  ⚪ {name}: нейтрально"
            return f"  ❌ {name}: `{sc:+.1f}` (против)"

        daily_loss = AUTOTRADE_DAILY_LOSS.get(chat_id, 0.0)
        limit_usdt = balance * DAILY_LOSS_LIMIT_PCT / 100

        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"✅ Войти (~${risk_usdt:.0f} риск)", callback_data=f"trade_confirm:{trade_id}"),
            InlineKeyboardButton("❌ Отмена", callback_data=f"trade_cancel:{trade_id}"),
        ]])

        extra_note = (
            f"\n_Выбран сильнейший сигнал из {len(qualified)}, прошедших фильтры в этом цикле._"
            if len(qualified) > 1 else ""
        )

        text = (
            f"🤖 *Авто-сигнал: {dir_label}* {mode_label}\n\n"
            f"💎 *{symbol.replace('USDT', '')}*  |  `{fmt_price(price)}`\n"
            f"📊 Скан: `{quick_score:+.1f}` | Стратегии: `{strat_total:+.1f}` | Итого: `{combined:+.1f}`\n"
            f"🔢 Согласны {agree_count}/5 стратегий\n\n"
            f"*📋 Стратегии:*\n"
            f"{strat_line('Mean Reversion', mr_score)}\n"
            f"{strat_line('Trend Pullback', tp_score)}\n"
            f"{strat_line('Breakout      ', bo_score)}\n"
            f"{strat_line('Div. Swing    ', ds_score)}\n"
            f"{strat_line('MultiTF 1H+4H ', mtf_score)}\n\n"
            f"📐 {tf_note}  |  {btc_note}\n"
            + (f"⚡ {fr_note}\n" if fr_note else "") +
            f"\n🛑 SL: `{fmt_price(sl)}`\n"
            f"🎯 TP1 (50%): `{fmt_price(tp1)}`\n"
            f"🎯 TP2 (50%): `{fmt_price(tp2)}`\n"
            f"⚖️ Плечо: `x{DEFAULT_LEVERAGE}`\n"
            f"💰 Риск: `~${risk_usdt:.2f}` ({risk_pct}%)\n"
            f"🛡 Дн. убыток: `${daily_loss:.2f}` / `${limit_usdt:.2f}`\n"
            f"⏳ _Актуально ~15 минут_{extra_note}"
        )

        try:
            await ctx.bot.send_message(chat_id=chat_id, text=text,
                                       parse_mode="Markdown", reply_markup=kb)
        except Exception as e:
            logger.warning(f"autotrade_scan_job {chat_id}: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Установи TELEGRAM_BOT_TOKEN")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler("strategy", strategy))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("mysubs", mysubs))
    app.add_handler(CommandHandler("subscribe_market", subscribe_market))
    app.add_handler(CommandHandler("unsubscribe_market", unsubscribe_market))
    app.add_handler(CommandHandler("setkey", setkey))
    app.add_handler(CommandHandler("deletekey", deletekey))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("positions", positions))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("futures", futures))
    app.add_handler(CommandHandler("top", top))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("autotrade", autotrade_cmd))
    app.add_handler(CommandHandler("autoportfolio", autoportfolio))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(global_error_handler)
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    # Фоновые задачи. Алерты и автоторговля разнесены по фазе (first=30 и first=330),
    # чтобы не бить Binance-лимиты одновременно — оба скана тяжёлые (по 60 символов).
    app.job_queue.run_repeating(alert_job,         interval=600,  first=30)
    app.job_queue.run_repeating(market_scan_job,   interval=3600, first=60)
    # Автоторговля: скан каждые 10 мин, но со сдвигом +5 мин относительно алертов.
    app.job_queue.run_repeating(autotrade_scan_job, interval=600,  first=330)
    # Трейлинг-стоп / перевод в б/у после TP1 — каждые 2 мин, лёгкий job.
    app.job_queue.run_repeating(trailing_stop_job,  interval=120,  first=120)

    logger.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
