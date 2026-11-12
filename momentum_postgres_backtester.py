import streamlit as st
import pandas as pd
import numpy as np
import psycopg

st.set_page_config(
    page_title="Upgraded X-Day Low / Y-Day High Backtester",
    layout="wide"
)

st.title("Simple Uptrend Pullback Backtester — Baseline vs. Institutional Upgrades")
st.markdown(
    "**Baseline (`10d Low -> 5d High`):** Stock is higher than 200 days ago, closes at an X-day low, "
    "sorted largest-to-smallest by Dollar Volume, and exits at a Y-day high close. "
    "Use this suite to test **4 targeted upgrades** (20d Avg Dollar Volume + Crash Filter, "
    "50d/200d Active Uptrend, SPY 200d Regime Gate, and 2-Down-Day + 15d Time-Stop) individually or combined."
)

# =========================================================
# 1. SIDEBAR PARAMETERS
# =========================================================
st.sidebar.header("1. PostgreSQL Connection")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.text_input("Port", value="5432")
db_name = st.sidebar.text_input("Database Name", value="stock_market_db")
db_user = st.sidebar.text_input("Username", value="postgres")
db_pass = st.sidebar.text_input("Password", type="password", value="")
db_table = st.sidebar.text_input("Table Name", value="stock_prices_daily")

st.sidebar.header("2. Mode & Date Range")
strat_mode = st.sidebar.selectbox(
    "Select Run Mode",
    options=[
        "Compare Baseline vs. 4 Upgraded Versions Side-by-Side",
        "Run Single Custom Strategy (Use Toggles Below)"
    ]
)
start_year = st.sidebar.number_input("Start Year", min_value=1995, max_value=2026, value=2006, step=1)
end_year = st.sidebar.number_input("End Year", min_value=1995, max_value=2026, value=2026, step=1)

st.sidebar.header("3. Entry (X-Day Low) & Exit (Y-Day High)")
top_n = st.sidebar.number_input("Max Active Stock Slots", min_value=1, max_value=30, value=10, step=1)
x_low_days = st.sidebar.slider("Entry: X-Day Low Close (Trading Days)", min_value=2, max_value=30, value=10, step=1)
y_high_days = st.sidebar.slider("Exit: Y-Day High Close (Trading Days)", min_value=2, max_value=30, value=5, step=1)

st.sidebar.header("4. Upgrade Toggles (For Single Custom Mode)")
dvol_sort_type = st.sidebar.selectbox(
    "Upgrade A: Dollar Volume Sort Metric",
    options=[
        "20-Day Average Dollar Volume (Avoids 1-Day Crash Spikes)",
        "Today's Dollar Volume (Original Baseline)"
    ]
)
max_1d_drop_pct = st.sidebar.slider(
    "Upgrade A2: Exclude If Today's 1-Day Crash Worse Than (-%)",
    min_value=3.0, max_value=50.0, value=8.0, step=1.0,
    help="Prevents buying stocks experiencing a single-day -8%+ earnings or news collapse today."
) / 100.0
use_1d_crash_filter = st.sidebar.checkbox("Enable 1-Day Crash Filter on Entry", value=True)

use_active_uptrend = st.sidebar.checkbox(
    "Upgrade B: Require Active Uptrend (Close > 200d SMA & 50d > 200d SMA)",
    value=True
)
use_market_regime = st.sidebar.checkbox(
    "Upgrade C: Broad Market Regime Filter (SPY / Market > 200d SMA)",
    value=True
)
require_two_down = st.sidebar.checkbox(
    "Upgrade D1: Require >= 2 Consecutive Down Days on Entry",
    value=True
)
use_max_hold = st.sidebar.checkbox(
    "Upgrade D2: Evict Stagnant 'Zombie' Trades (Max Hold Cap)",
    value=True
)
max_hold_days = st.sidebar.slider("Max Hold Cap (Trading Days)", min_value=5, max_value=60, value=15, step=1)

use_stop_loss = st.sidebar.checkbox("Optional: Add Hard Stop-Loss", value=False)
if use_stop_loss:
    stop_loss_pct = st.sidebar.slider("Stop-Loss (% Below Entry)", 2.0, 40.0, 10.0, 0.5) / 100.0
else:
    stop_loss_pct = 0.0

st.sidebar.header("5. Liquidity, Capital & Commissions")
min_adv = st.sidebar.number_input("Min 30-Day Avg Share Volume", min_value=0, value=300000, step=50000)
min_price = st.sidebar.number_input("Min Share Price ($)", min_value=1.0, value=5.0, step=1.0)
max_price = st.sidebar.number_input("Max Share Price (Reverse-Split Guard $)", min_value=100.0, value=2500.0, step=500.0)

starting_capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, value=100000.0, step=10000.0)
sizing_mode = st.sidebar.selectbox(
    "Position Sizing Mode",
    options=[
        "Compounding (Reinvest Profits Across Active Slots)",
        "Capped Slot Size (No Profit Reinvestment, Deduct Losses)"
    ]
)
commission_per_trade = st.sidebar.number_input("Commission per Order ($)", min_value=0.0, value=1.00, step=0.25, format="%.2f")

run_btn = st.sidebar.button("Load Data & Run Backtest", type="primary", use_container_width=True)


# =========================================================
# 2. CACHED POSTGRESQL MATRIX LOADER
# =========================================================
@st.cache_data(show_spinner=False)
def load_matrix_from_postgres(host, port, dbname, user, password, table, s_year, e_year):
    conn_str = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} LIMIT 1")
            col_names = [desc[0].lower() for desc in cur.description]

        ticker_col = next((c for c in ['ticker', 'symbol', 'code'] if c in col_names), col_names[0])
        date_col = next((c for c in ['date', 'trade_date', 'datetime', 'timestamp'] if c in col_names), col_names[1])
        adj_col = next((c for c in ['adj_close', 'adjusted_close', 'adjclose', 'close', 'close_price'] if c in col_names), 'close')
        vol_col = next((c for c in ['volume', 'vol'] if c in col_names), 'volume')

        query = f"""
            SELECT {ticker_col} AS ticker,
                   {date_col} AS date,
                   {adj_col}::real AS adj_close,
                   {vol_col}::real AS volume
            FROM {table}
            WHERE {date_col} >= '{s_year}-01-01'
              AND {date_col} <= '{e_year}-12-31'
              AND {adj_col} > 0
            ORDER BY {date_col}, {ticker_col}
        """
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return None, None, 0, adj_col

    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date', 'ticker'], keep='last')

    adj_close_mat = df.pivot(index='date', columns='ticker', values='adj_close').astype(np.float32)
    vol_mat = df.pivot(index='date', columns='ticker', values='volume').fillna(0.0).astype(np.float32)

    return adj_close_mat, vol_mat, len(df), adj_col


# =========================================================
# 3. PRECOMPUTE BASE & UPGRADE MATRICES
# =========================================================
def compute_base_matrices(adj_close_mat, vol_mat):
    px_ffill = adj_close_mat.ffill(limit=5)
    prev_c = px_ffill.shift(1)
    ret_1d = (px_ffill - prev_c) / prev_c

    # 2 consecutive down days (today < yesterday AND yesterday < day before)
    down_day = (ret_1d < 0.0)
    two_down_days = (down_day & down_day.shift(1).fillna(False))

    # Trend filters
    higher_200d = (px_ffill > px_ffill.shift(200))
    sma_200 = px_ffill.rolling(200, min_periods=150).mean()
    sma_50 = px_ffill.rolling(50, min_periods=40).mean()
    active_uptrend = higher_200d & (px_ffill > sma_200) & (sma_50 > sma_200)

    # Share volume & Dollar volume matrices
    adv_30d = vol_mat.rolling(30, min_periods=15).mean()
    dvol_today = (adj_close_mat * vol_mat).fillna(0.0)
    dvol_20d = dvol_today.rolling(20, min_periods=10).mean().fillna(0.0)

    # Broad Market Regime: Use SPY if present, otherwise equal-weight universe index
    if 'SPY' in px_ffill.columns and px_ffill['SPY'].notna().sum() > 250:
        mkt_series = px_ffill['SPY'].ffill()
    else:
        clean_ret = ret_1d.clip(lower=-0.50, upper=0.50).mean(axis=1).fillna(0.0)
        mkt_series = (1.0 + clean_ret).cumprod()

    mkt_sma_200 = mkt_series.rolling(200, min_periods=100).mean()
    mkt_bull_200 = (mkt_series >= mkt_sma_200).to_numpy()

    return {
        'px_ffill_df': px_ffill,
        'px_ffill': px_ffill.to_numpy(),
        'raw_px': adj_close_mat.to_numpy(),
        'ret_1d': ret_1d.to_numpy(),
        'two_down': two_down_days.to_numpy(),
        'higher_200d': higher_200d.to_numpy(),
        'active_uptrend': active_uptrend.to_numpy(),
        'adv_30d': adv_30d.to_numpy(),
        'dvol_today': dvol_today.to_numpy(),
        'dvol_20d': dvol_20d.to_numpy(),
        'mkt_bull_200': mkt_bull_200
    }


def compute_xy_triggers(px_ffill_df, x_days, y_days):
    w_x = max(1, int(x_days) - 1)
    prev_min_x = px_ffill_df.shift(1).rolling(window=w_x, min_periods=w_x).min()
    is_x_low = (px_ffill_df <= prev_min_x).to_numpy()

    w_y = max(1, int(y_days) - 1)
    prev_max_y = px_ffill_df.shift(1).rolling(window=w_y, min_periods=w_y).max()
    is_y_high = (px_ffill_df >= prev_max_y).to_numpy()

    return is_x_low, is_y_high


# =========================================================
# 4. SLOT SIMULATION ENGINE
# =========================================================
def simulate_xy_variant(
    dates, tickers, base, is_x_low_arr, is_y_high_arr,
    x_days, y_days, top_k,
    use_20d_dvol, use_crash_cap, crash_cap_pct,
    req_active_trend, req_mkt_regime, req_2_down,
    use_sl, sl_pct, use_max_h, max_h_days,
    min_sh_vol, min_px, max_px,
    start_cap, mode_sizing, comm_val, strat_label
):
    px_arr = base['px_ffill']
    raw_px_arr = base['raw_px']
    ret1d_arr = base['ret_1d']
    twodown_arr = base['two_down']
    higher200_arr = base['higher_200d']
    act_trend_arr = base['active_uptrend']
    adv_arr = base['adv_30d']
    dvol_arr = base['dvol_20d'] if use_20d_dvol else base['dvol_today']
    bull_arr = base['mkt_bull_200']

    n_days, _ = px_arr.shape
    start_idx = 200
    if start_idx >= n_days - 2:
        return pd.DataFrame(), pd.DataFrame()

    initial_slot_cap = float(start_cap) / top_k
    slot_cash = np.full(top_k, initial_slot_cap, dtype=np.float64)
    slot_equity_cap = np.full(top_k, initial_slot_cap, dtype=np.float64)
    slot_ticker_idx = np.full(top_k, -1, dtype=int)
    slot_entry_px = np.zeros(top_k, dtype=np.float64)
    slot_shares = np.zeros(top_k, dtype=np.float64)
    slot_entry_day = np.zeros(top_k, dtype=int)

    daily_equity = np.full(n_days, np.nan, dtype=np.float64)
    daily_equity[:start_idx] = start_cap

    trade_logs = []

    for d in range(start_idx, n_days):
        is_final_day = (d == n_days - 1)
        exited_today = set()

        # STEP 1: CHECK ACTIVE POSITIONS FOR Y-DAY HIGH EXIT (OR OPTIONAL TIME-STOP / STOP-LOSS)
        for s in range(top_k):
            t_idx = slot_ticker_idx[s]
            if t_idx != -1:
                cur_px = float(px_arr[d, t_idx])
                if not np.isfinite(cur_px) or cur_px <= 0:
                    cur_px = slot_entry_px[s]

                ret = (cur_px - slot_entry_px[s]) / slot_entry_px[s]
                days_held = d - slot_entry_day[s]
                hit_y_high = bool(is_y_high_arr[d, t_idx])

                exit_reason = None
                if is_final_day:
                    exit_reason = "End of Backtest"
                elif days_held >= 1 and hit_y_high:
                    exit_reason = f"{y_days}-Day High Close"
                elif use_sl and sl_pct > 0 and ret <= -sl_pct:
                    exit_reason = f"Stop Loss (-{sl_pct*100:.1f}%)"
                elif use_max_h and days_held >= max_h_days:
                    exit_reason = f"Time-Stop ({days_held}d)"

                if exit_reason is not None:
                    gross_pnl = (cur_px - slot_entry_px[s]) * slot_shares[s]
                    net_pnl = gross_pnl - (2.0 * comm_val)
                    proceeds = (slot_shares[s] * cur_px) - comm_val
                    slot_cash[s] += proceeds
                    slot_equity_cap[s] += net_pnl

                    trade_logs.append({
                        'Strategy': strat_label,
                        'Slot #': s + 1,
                        'Ticker': tickers[t_idx],
                        'Entry Date': dates[slot_entry_day[s]].strftime('%Y-%m-%d'),
                        'Exit Date': dates[d].strftime('%Y-%m-%d'),
                        'Days Held': int(days_held),
                        'Shares': int(slot_shares[s]),
                        'Entry Close ($)': round(float(slot_entry_px[s]), 2),
                        'Exit Close ($)': round(float(cur_px), 2),
                        'Return (%)': round(float(ret * 100.0), 2),
                        'Exit Reason': exit_reason,
                        'Net PnL ($)': round(float(net_pnl), 2)
                    })

                    exited_today.add(t_idx)
                    slot_ticker_idx[s] = -1
                    slot_shares[s] = 0.0

        # STEP 2: FILL EMPTY SLOTS AT TODAY'S CLOSE
        empty_slots = [s for s in range(top_k) if slot_ticker_idx[s] == -1]
        regime_ok = True if not req_mkt_regime else bool(bull_arr[d])

        if empty_slots and not is_final_day and regime_ok:
            c_row = raw_px_arr[d]
            trend_row = act_trend_arr[d] if req_active_trend else higher200_arr[d]
            xlow_row = is_x_low_arr[d]
            dvol_row = dvol_arr[d]
            adv_row = adv_arr[d]

            valid_mask = (
                np.isfinite(c_row) &
                np.isfinite(dvol_row) &
                np.isfinite(adv_row) &
                trend_row &
                xlow_row &
                (c_row >= min_px) &
                (c_row <= max_px) &
                (adv_row >= min_sh_vol) &
                (dvol_row > 0)
            )

            if use_crash_cap:
                r1d_row = ret1d_arr[d]
                valid_mask = valid_mask & np.isfinite(r1d_row) & (r1d_row > -crash_cap_pct)

            if req_2_down:
                valid_mask = valid_mask & twodown_arr[d]

            excluded_tickers = set(slot_ticker_idx[slot_ticker_idx != -1]).union(exited_today)
            for ex_idx in excluded_tickers:
                valid_mask[ex_idx] = False

            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 0:
                if "Compounding" in mode_sizing and len(empty_slots) > 1:
                    pooled_cash = sum(slot_cash[s] for s in empty_slots)
                    per_slot = pooled_cash / len(empty_slots)
                    for s in empty_slots:
                        slot_cash[s] = per_slot

                top_order = np.argsort(dvol_row[valid_indices])[::-1][:len(empty_slots)]
                chosen_indices = valid_indices[top_order]

                for j, t_idx in enumerate(chosen_indices):
                    s = empty_slots[j]
                    entry_px = float(c_row[t_idx])

                    if "Compounding" in mode_sizing:
                        target_alloc = max(slot_cash[s] - comm_val, 0.0)
                    else:
                        capped_budget = min(max(slot_equity_cap[s], 0.0), initial_slot_cap)
                        target_alloc = max(min(slot_cash[s], capped_budget) - comm_val, 0.0)

                    sh = np.floor(target_alloc / entry_px)
                    if sh >= 1:
                        slot_ticker_idx[s] = t_idx
                        slot_entry_px[s] = entry_px
                        slot_shares[s] = sh
                        slot_entry_day[s] = d
                        slot_cash[s] -= (sh * entry_px + comm_val)

        # STEP 3: DAILY MARK-TO-MARKET PORTFOLIO EQUITY
        day_total_eq = 0.0
        for s in range(top_k):
            t_idx = slot_ticker_idx[s]
            if t_idx != -1:
                cpx = float(px_arr[d, t_idx])
                if not np.isfinite(cpx) or cpx <= 0:
                    cpx = slot_entry_px[s]
                day_total_eq += slot_cash[s] + (slot_shares[s] * cpx)
            else:
                day_total_eq += slot_cash[s]
        daily_equity[d] = day_total_eq

    eq_df = pd.DataFrame({
        'Date': dates,
        'Equity': daily_equity
    }).dropna(subset=['Equity']).reset_index(drop=True)

    return eq_df, pd.DataFrame(trade_logs)


def summarize_performance(eq_df, trades_df, start_cap, label):
    eq_df = eq_df.copy()
    eq_df['Daily_Ret'] = eq_df['Equity'].pct_change().fillna(0.0)
    eq_df['Peak_Equity'] = eq_df['Equity'].cummax()
    eq_df['Drawdown_Pct'] = (eq_df['Equity'] - eq_df['Peak_Equity']) / eq_df['Peak_Equity'] * 100.0

    ending_eq = eq_df['Equity'].iloc[-1]
    net_profit = ending_eq - start_cap
    tot_ret_pct = (ending_eq / start_cap - 1.0) * 100.0
    max_dd_pct = eq_df['Drawdown_Pct'].min()

    n_years = max((eq_df['Date'].iloc[-1] - eq_df['Date'].iloc[0]).days / 365.25, 0.1)
    cagr_pct = ((max(ending_eq, 1.0) / start_cap) ** (1.0 / n_years) - 1.0) * 100.0
    sharpe = (eq_df['Daily_Ret'].mean() / eq_df['Daily_Ret'].std()) * np.sqrt(252) if eq_df['Daily_Ret'].std() > 0 else 0.0
    calmar = cagr_pct / abs(max_dd_pct) if max_dd_pct < 0 else 0.0

    win_rate = (trades_df['Net PnL ($)'] > 0).mean() * 100.0 if not trades_df.empty else 0.0
    avg_hold = trades_df['Days Held'].mean() if not trades_df.empty else 0.0

    summary = {
        'Strategy': label,
        'Total Return (%)': f"{tot_ret_pct:+.2f}%",
        'Ending Equity ($)': f"${ending_eq:,.2f}",
        'CAGR (%/yr)': f"{cagr_pct:+.2f}%",
        'Max Drawdown (%)': f"{max_dd_pct:.2f}%",
        'Ann. Sharpe': round(sharpe, 2),
        'Calmar (CAGR/DD)': round(calmar, 2),
        'Total Trades': len(trades_df),
        'Win Rate (%)': f"{win_rate:.1f}%",
        'Avg Days Held': f"{avg_hold:.1f}d"
    }
    return eq_df, summary


def build_yearly_table(eq_df, trades_df, start_cap):
    eq_df = eq_df.copy()
    eq_df['Year'] = eq_df['Date'].dt.year
    rows = []
    prev_end = start_cap

    for yr, grp in eq_df.groupby('Year'):
        yr_end = grp['Equity'].iloc[-1]
        yr_pnl = yr_end - prev_end
        yr_ret = (yr_pnl / prev_end) * 100.0 if prev_end > 0 else 0.0
        yr_peak = grp['Equity'].cummax()
        yr_dd = ((grp['Equity'] - yr_peak) / yr_peak * 100.0).min()

        if not trades_df.empty:
            yr_tr = trades_df[pd.to_datetime(trades_df['Exit Date']).dt.year == yr]
            n_tr = len(yr_tr)
            wr = (yr_tr['Net PnL ($)'] > 0).mean() * 100.0 if n_tr > 0 else 0.0
        else:
            n_tr, wr = 0, 0.0

        rows.append({
            'Year': int(yr),
            'Start Equity ($)': f"${prev_end:,.2f}",
            'End Equity ($)': f"${yr_end:,.2f}",
            'Net Profit ($)': f"${yr_pnl:+,.2f}",
            'Annual Return (%)': f"{yr_ret:+.2f}%",
            'Max Intra-Year DD (%)': f"{yr_dd:.2f}%",
            'Completed Trades': n_tr,
            'Win Rate (%)': f"{wr:.1f}%"
        })
        prev_end = yr_end
    return pd.DataFrame(rows)


# =========================================================
# 5. MAIN DASHBOARD EXECUTION
# =========================================================
if run_btn or 'postgres_loaded' in st.session_state:
    st.session_state['postgres_loaded'] = True
    try:
        with st.spinner(f"Querying {db_table} from PostgreSQL ({start_year}-{end_year})..."):
            adj_close_mat, vol_mat, total_rows, detected_adj_col = load_matrix_from_postgres(
                db_host, db_port, db_name, db_user, db_pass, db_table, start_year, end_year
            )
    except Exception as e:
        st.error(f"Database Connection / Query Error: {e}")
        st.stop()

    if adj_close_mat is None or adj_close_mat.empty:
        st.warning("No rows returned from PostgreSQL for the selected date range.")
        st.stop()

    st.success(
        f"Loaded {total_rows:,} daily rows across {adj_close_mat.shape[1]:,} tickers "
        f"({adj_close_mat.index.min().strftime('%Y-%m-%d')} to {adj_close_mat.index.max().strftime('%Y-%m-%d')}). "
        f"Active Price Column: {detected_adj_col}."
    )

    with st.spinner("Computing 200d SMAs, 20d Dollar Volume, Crash Filters & Market Regime..."):
        base_mats = compute_base_matrices(adj_close_mat, vol_mat)
        x_low_arr, y_high_arr = compute_xy_triggers(base_mats['px_ffill_df'], x_low_days, y_high_days)
        dates = adj_close_mat.index
        tickers = adj_close_mat.columns.to_numpy()

    # Format: (Label, use_20d_dvol, use_crash_cap, req_active_trend, req_mkt_regime, req_2_down, use_max_h)
    if "Single Custom" in strat_mode:
        configs = [
            (
                f"Custom ({x_low_days}d Low to {y_high_days}d High)",
                ("20-Day" in dvol_sort_type),
                use_1d_crash_filter,
                use_active_uptrend,
                use_market_regime,
                require_two_down,
                use_max_hold
            )
        ]
    else:
        configs = [
            (f"1. Original Baseline ({x_low_days}d Low to {y_high_days}d High)", False, False, False, False, False, False),
            ("2. + 20d Avg $Vol & No -8% Crash Day", True, True, False, False, False, False),
            ("3. + Active Uptrend (Close > 200d & 50d > 200d SMA)", True, True, True, False, False, False),
            ("4. + SPY 200d Market Regime Gate", True, True, True, True, False, False),
            (f"5. All Upgrades (+ 2 Down Days & {max_hold_days}d Time-Stop)", True, True, True, True, True, True)
        ]

    results_dict = {}
    for label, u_20d, u_crash, u_trend, u_reg, u_2dn, u_maxh in configs:
        with st.spinner(f"Simulating {label}..."):
            eq_d, tr_d = simulate_xy_variant(
                dates, tickers, base_mats, x_low_arr, y_high_arr,
                x_low_days, y_high_days, top_n,
                u_20d, u_crash, max_1d_drop_pct,
                u_trend, u_reg, u_2dn,
                use_stop_loss, stop_loss_pct, u_maxh, max_hold_days,
                min_adv, min_price, max_price,
                starting_capital, sizing_mode, commission_per_trade, label
            )
            if not eq_d.empty:
                eq_full, summary = summarize_performance(eq_d, tr_d, starting_capital, label)
                results_dict[label] = (eq_full, tr_d, summary)

    if not results_dict:
        st.warning("No trades matched your filter criteria.")
        st.stop()

    st.subheader("Leaderboard — Baseline vs. Institutional Upgrades")
    summary_df = pd.DataFrame([v[2] for v in results_dict.values()])
    st.dataframe(summary_df, use_container_width=True)

    combined_eq = pd.DataFrame()
    combined_dd = pd.DataFrame()
    for name, (eq_d, _, _) in results_dict.items():
        safe_col_name = name.replace(":", " -")
        combined_eq[safe_col_name] = eq_d.set_index('Date')['Equity']
        combined_dd[safe_col_name] = eq_d.set_index('Date')['Drawdown_Pct']

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.subheader("Daily Mark-to-Market Portfolio Equity ($)")
        st.line_chart(combined_eq)
    with c_chart2:
        st.subheader("Daily Underwater Drawdown (%)")
        st.line_chart(combined_dd)

    st.markdown("---")
    sys_tabs = st.tabs(list(results_dict.keys()))
    for idx, (sys_name, (eq_d, tr_d, sum_d)) in enumerate(results_dict.items()):
        with sys_tabs[idx]:
            st.subheader(f"{sys_name} — Detailed Breakdown")
            if not tr_d.empty:
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown("**Performance by Exit Reason**")
                    r_sum = tr_d.groupby('Exit Reason').agg(
                        Trades=('Net PnL ($)', 'count'),
                        Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean()*100:.1f}%"),
                        Avg_Return=('Return (%)', lambda x: f"{x.mean():+.2f}%"),
                        Avg_Days_Held=('Days Held', lambda x: f"{x.mean():.1f}d"),
                        Total_Net_PnL=('Net PnL ($)', lambda x: f"${x.sum():+,.2f}")
                    ).reset_index()
                    st.dataframe(r_sum, use_container_width=True)
                with b2:
                    st.markdown("**Top 10 Most Frequently Traded Stocks**")
                    top_t = tr_d.groupby('Ticker').agg(
                        Trades=('Net PnL ($)', 'count'),
                        Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean()*100:.1f}%"),
                        Total_Net_PnL=('Net PnL ($)', lambda x: f"${x.sum():+,.2f}")
                    ).sort_values('Trades', ascending=False).head(10).reset_index()
                    st.dataframe(top_t, use_container_width=True)

            st.markdown(f"#### Year-by-Year Return Table — {sys_name}")
            yr_table = build_yearly_table(eq_d, tr_d, starting_capital)
            st.dataframe(yr_table, use_container_width=True)

            st.markdown(f"#### Master Trade Log — {sys_name} ({len(tr_d):,} Trades)")
            st.dataframe(tr_d, use_container_width=True, height=400)
else:
    st.info("Enter your PostgreSQL password in the sidebar and click Load Data & Run Backtest.")

# END OF SCRIPT
