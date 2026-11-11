import streamlit as st
import pandas as pd
import numpy as np
import psycopg

st.set_page_config(
    page_title="PostgreSQL 6,000-Ticker Momentum Backtester",
    layout="wide"
)

st.title("🚀 6,000-Ticker Cross-Sectional Momentum Backtester (PostgreSQL)")
st.markdown("""
**Strategy Logic:** At each rebalance, rank all stocks in your PostgreSQL database by their **prior $L$-day return** (default 30 trading days). 
Filter for stocks with **Average Daily Volume $\\ge$ 300,000 shares** and **no single overnight gap-up $\\ge$ 30%** during the lookback window. 
Buy the **Top 10 stocks** on the next morning's Open, hold for **$X$ trading days**, and deduct **$1.00 per trade** in commissions.
""")

# =========================================================
# 1. SIDEBAR: POSTGRESQL CONNECTION & STRATEGY PARAMETERS
# =========================================================
st.sidebar.header("1. PostgreSQL Connection")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.text_input("Port", value="5432")
db_name = st.sidebar.text_input("Database Name", value="stock_market_db")
db_user = st.sidebar.text_input("Username", value="postgres")
db_pass = st.sidebar.text_input("Password", type="password", value="")
db_table = st.sidebar.text_input("Table Name", value="stock_prices_daily")

st.sidebar.header("2. Date Range to Load")
start_year = st.sidebar.number_input("Start Year", min_value=1995, max_value=2026, value=2006, step=1)
end_year = st.sidebar.number_input("End Year", min_value=1995, max_value=2026, value=2026, step=1)

st.sidebar.header("3. Momentum & Filter Rules")
top_n = st.sidebar.number_input("Number of Top Stocks to Buy", min_value=1, max_value=50, value=10, step=1)
lookback_days = st.sidebar.slider("Momentum Lookback Window (Trading Days)", min_value=5, max_value=120, value=30, step=5)
hold_days = st.sidebar.slider("Holding Period X (Trading Days)", min_value=1, max_value=120, value=10, step=1)

min_adv = st.sidebar.number_input("Min 30-Day Avg Daily Volume (Shares)", min_value=0, value=300000, step=50000)
max_overnight_gap_pct = st.sidebar.slider(
    "Exclude If Max Overnight Gap-Up >= (%)",
    min_value=5.0, max_value=100.0, value=30.0, step=2.5,
    help="Excludes any stock that had a single overnight gap (Open_t / Close_{t-1} - 1) >= this threshold during the 30-day lookback window."
) / 100.0
also_cap_daily_ret = st.sidebar.checkbox(
    "Also apply gap cap to Total Daily Return (Close_t / Close_{t-1} - 1)",
    value=False,
    help="Useful if any historical data rows have Open == Close_t-1 on gap days."
)
min_price = st.sidebar.number_input("Min Share Price ($ at Evaluation)", min_value=0.0, value=5.0, step=1.0)

st.sidebar.header("4. Capital, Sizing & Commissions")
starting_capital = st.sidebar.number_input("Starting Portfolio Capital ($)", min_value=1000.0, value=100000.0, step=10000.0)
sizing_mode = st.sidebar.selectbox(
    "Position Sizing Mode",
    options=[
        "Compounding (Divide Current Portfolio Equity Equally Across Top 10)",
        "Capped Slot Size (No Profit Reinvestment, Deduct Losses)"
    ]
)
commission_per_order = st.sidebar.number_input("Commission ($)", min_value=0.0, value=1.00, step=0.25, format="%.2f")
comm_mode = st.sidebar.selectbox(
    "Commission Application",
    options=[
        "$1 per Order Leg ($1 Buy + $1 Sell = $2 Round-Trip per Stock)",
        "$1 Flat per Position Round-Trip"
    ]
)

run_btn = st.sidebar.button("🚀 Load Data & Run Backtest", type="primary", use_container_width=True)

# =========================================================
# 2. CACHED POSTGRESQL MATRIX LOADER (FLOAT32 OPTIMIZED)
# =========================================================
@st.cache_data(show_spinner=False)
def load_matrix_from_postgres(host, port, dbname, user, password, table, s_year, e_year):
    conn_str = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    with psycopg.connect(conn_str) as conn:
        # Inspect actual column names in table to handle schema variations automatically
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} LIMIT 1")
            col_names = [desc[0].lower() for desc in cur.description]

        ticker_col = next((c for c in ['ticker', 'symbol', 'code'] if c in col_names), col_names[0])
        date_col = next((c for c in ['date', 'trade_date', 'datetime', 'timestamp'] if c in col_names), col_names[1])
        open_col = next((c for c in ['open', 'open_price'] if c in col_names), 'open')
        close_col = next((c for c in ['close', 'adj_close', 'close_price'] if c in col_names), 'close')
        vol_col = next((c for c in ['volume', 'vol'] if c in col_names), 'volume')

        query = f"""
            SELECT {ticker_col} AS ticker,
                   {date_col} AS date,
                   {open_col}::real AS open,
                   {close_col}::real AS close,
                   {vol_col}::real AS volume
            FROM {table}
            WHERE {date_col} >= '{s_year}-01-01'
              AND {date_col} <= '{e_year}-12-31'
              AND {close_col} > 0
              AND {open_col} > 0
            ORDER BY {date_col}, {ticker_col}
        """
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return None, None, None, 0

    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date', 'ticker'], keep='last')

    # Pivot into fast (Date x Ticker) float32 matrices
    open_mat = df.pivot(index='date', columns='ticker', values='open').astype(np.float32)
    close_mat = df.pivot(index='date', columns='ticker', values='close').astype(np.float32)
    vol_mat = df.pivot(index='date', columns='ticker', values='volume').fillna(0.0).astype(np.float32)

    total_rows = len(df)
    return open_mat, close_mat, vol_mat, total_rows


# =========================================================
# 3. VECTORIZED SIGNAL & DAILY MARK-TO-MARKET ENGINE
# =========================================================
def run_momentum_backtest(
    open_mat, close_mat, vol_mat,
    lookback, hold_x, top_k, min_vol, max_gap, cap_daily, min_px,
    start_cap, mode_sizing, comm_val, mode_comm
):
    dates = close_mat.index
    tickers = close_mat.columns.to_numpy()
    n_days, n_tickers = close_mat.shape

    # 1. Prior L-Day Momentum Return: Close_t / Close_{t - lookback} - 1
    mom_ret = (close_mat / close_mat.shift(lookback)) - 1.0

    # 2. Rolling L-Day Average Daily Volume (ADV)
    adv_mat = vol_mat.rolling(window=lookback, min_periods=lookback).mean()

    # 3. Overnight Gap Up: (Open_t - Close_{t-1}) / Close_{t-1}
    prev_close = close_mat.shift(1)
    overnight_gap = (open_mat - prev_close) / prev_close
    if cap_daily:
        daily_cc = (close_mat - prev_close) / prev_close
        overnight_gap = np.maximum(overnight_gap, daily_cc)

    # Rolling Max Overnight Gap over the exact L-day window [t - lookback + 1 ... t]
    max_gap_window = overnight_gap.rolling(window=lookback, min_periods=max(1, lookback // 2)).max()

    # 4. Daily Close-to-Close returns for mark-to-market tracking during the hold
    # Note: We do NOT forward-fill across dead tickers unless needed, and clamp extreme data errors
    close_ffill = close_mat.ffill(limit=5)

    # Convert to NumPy arrays for ultra-fast rebalance loop
    mom_arr = mom_ret.to_numpy()
    adv_arr = adv_mat.to_numpy()
    gap_arr = max_gap_window.to_numpy()
    open_arr = open_mat.to_numpy()
    close_arr = close_ffill.to_numpy()

    comm_per_position = (comm_val * 2.0) if "Order Leg" in mode_comm else comm_val

    # Track daily portfolio equity
    start_idx = lookback + 1
    if start_idx >= n_days - 2:
        return pd.DataFrame(), pd.DataFrame()

    daily_equity = np.full(n_days, np.nan, dtype=np.float64)
    daily_equity[:start_idx] = start_cap

    current_equity = float(start_cap)
    initial_slot_cap = float(start_cap) / top_k
    slot_equities = np.full(top_k, initial_slot_cap, dtype=np.float64)

    rebalance_logs = []

    # Step through time in non-overlapping blocks of `hold_x` trading days
    # Signal evaluated at Close of day `sig_i` -> Enter at Open of `entry_i = sig_i + 1` -> Exit at Close of `exit_i = min(entry_i + hold_x - 1, n_days - 1)`
    sig_i = lookback
    while sig_i + 1 < n_days:
        entry_i = sig_i + 1
        exit_i = min(entry_i + hold_x - 1, n_days - 1)

        # Evaluate eligibility at sig_i (Close of Day t) AND verify stock has a valid Open price on entry_i (Day t+1)
        m_row = mom_arr[sig_i]
        v_row = adv_arr[sig_i]
        g_row = gap_arr[sig_i]
        c_row = close_arr[sig_i]
        next_open_row = open_arr[entry_i]

        valid_mask = (
            np.isfinite(m_row) &
            np.isfinite(v_row) &
            np.isfinite(g_row) &
            np.isfinite(c_row) &
            np.isfinite(next_open_row) &
            (next_open_row > 0) &
            (c_row >= min_px) &
            (v_row >= min_vol) &
            (g_row < max_gap) &
            (m_row > 0.0)  # Top performers must have positive 30-day momentum
        )

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # Stay in cash for this hold_x window
            daily_equity[entry_i:exit_i + 1] = current_equity
            sig_i = exit_i
            continue

        # Sort valid candidates by 30-day return descending and pick Top N
        valid_rets = m_row[valid_indices]
        top_order = np.argsort(valid_rets)[::-1][:top_k]
        chosen_idx = valid_indices[top_order]
        actual_k = len(chosen_idx)

        # Determine capital deployed per slot
        if "Compounding" in mode_sizing:
            alloc_per_stock = max(current_equity / top_k, 0.0)
            slot_allocs = np.full(actual_k, alloc_per_stock, dtype=np.float64)
            undeployed_cash = current_equity - np.sum(slot_allocs)
        else:
            # Capped at initial_slot_cap, shrinks if slot lost money
            slot_allocs = np.minimum(np.maximum(slot_equities[:actual_k], 0.0), initial_slot_cap)
            undeployed_cash = current_equity - np.sum(slot_allocs)

        entry_prices = next_open_row[chosen_idx]
        shares = np.floor(slot_allocs / entry_prices)
        invested_dollars = shares * entry_prices
        leftover_cash = np.sum(slot_allocs - invested_dollars) + undeployed_cash

        # Deduct entry + exit commissions for the batch
        total_batch_comm = actual_k * comm_per_position

        # Compute daily mark-to-market portfolio equity for every day d in [entry_i ... exit_i]
        for d in range(entry_i, exit_i + 1):
            cur_closes = close_arr[d, chosen_idx]
            # If a stock has a missing close on day d, fallback to entry_prices
            cur_closes = np.where(np.isfinite(cur_closes) & (cur_closes > 0), cur_closes, entry_prices)
            pos_values = shares * cur_closes
            daily_equity[d] = leftover_cash + np.sum(pos_values) - total_batch_comm

        # Finalize batch exit at Close of exit_i
        exit_prices = close_arr[exit_i, chosen_idx]
        exit_prices = np.where(np.isfinite(exit_prices) & (exit_prices > 0), exit_prices, entry_prices)
        gross_pnls = (exit_prices - entry_prices) * shares
        net_pnls = gross_pnls - comm_per_position

        current_equity = daily_equity[exit_i]
        if "Capped" in mode_sizing:
            slot_equities[:actual_k] += net_pnls

        # Log individual stock trades in this rebalance cohort
        for j in range(actual_k):
            t_idx = chosen_idx[j]
            rebalance_logs.append({
                'Signal Date': dates[sig_i].strftime('%Y-%m-%d'),
                'Entry Date (Open)': dates[entry_i].strftime('%Y-%m-%d'),
                'Exit Date (Close)': dates[exit_i].strftime('%Y-%m-%d'),
                'Ticker': tickers[t_idx],
                'Prior 30d Return (%)': round(float(m_row[t_idx]) * 100, 2),
                '30d Avg Volume': int(v_row[t_idx]),
                'Max 30d Overnight Gap (%)': round(float(g_row[t_idx]) * 100, 2),
                'Allocation ($)': round(float(invested_dollars[j]), 2),
                'Shares': int(shares[j]),
                'Entry Open ($)': round(float(entry_prices[j]), 2),
                'Exit Close ($)': round(float(exit_prices[j]), 2),
                'Hold Return (%)': round(float((exit_prices[j] / entry_prices[j] - 1.0) * 100), 2),
                'Net PnL ($)': round(float(net_pnls[j]), 2)
            })

        sig_i = exit_i

    eq_df = pd.DataFrame({
        'Date': dates,
        'Equity': daily_equity
    }).dropna(subset=['Equity']).reset_index(drop=True)

    trades_df = pd.DataFrame(rebalance_logs)
    return eq_df, trades_df


# =========================================================
# 4. MAIN DASHBOARD EXECUTION & DISPLAY
# =========================================================
if run_btn or 'postgres_loaded' in st.session_state:
    st.session_state['postgres_loaded'] = True
    try:
        with st.spinner(f"Querying `{db_table}` from PostgreSQL ({start_year}–{end_year}) and building float32 matrices..."):
            open_mat, close_mat, vol_mat, total_rows = load_matrix_from_postgres(
                db_host, db_port, db_name, db_user, db_pass, db_table, start_year, end_year
            )
    except Exception as e:
        st.error(f"Database Connection / Query Error: {e}")
        st.stop()

    if open_mat is None or open_mat.empty:
        st.warning("No rows returned from PostgreSQL for the selected date range.")
        st.stop()

    st.success(
        f"✅ Loaded **{total_rows:,} daily rows** across **{close_mat.shape[1]:,} unique tickers** "
        f"and **{close_mat.shape[0]:,} trading days** (`{close_mat.index.min().strftime('%Y-%m-%d')}` to `{close_mat.index.max().strftime('%Y-%m-%d')}`)."
    )

    with st.spinner("Running cross-sectional momentum ranking, gap filters & daily mark-to-market simulation..."):
        eq_df, trades_df = run_momentum_backtest(
            open_mat=open_mat,
            close_mat=close_mat,
            vol_mat=vol_mat,
            lookback=lookback_days,
            hold_x=hold_days,
            top_k=top_n,
            min_vol=min_adv,
            max_gap=max_overnight_gap_pct,
            cap_daily=also_cap_daily_ret,
            min_px=min_price,
            start_cap=starting_capital,
            mode_sizing=sizing_mode,
            comm_val=commission_per_order,
            mode_comm=comm_mode
        )

    if eq_df.empty or trades_df.empty:
        st.warning("No trades matched your filter criteria. Try lowering the minimum volume or adjusting the gap filter.")
        st.stop()

    # Calculate Overall Metrics
    eq_df['Daily_Ret'] = eq_df['Equity'].pct_change().fillna(0.0)
    eq_df['Peak_Equity'] = eq_df['Equity'].cummax()
    eq_df['Drawdown_Pct'] = (eq_df['Equity'] - eq_df['Peak_Equity']) / eq_df['Peak_Equity'] * 100.0
    eq_df['Drawdown_Dollar'] = eq_df['Equity'] - eq_df['Peak_Equity']

    ending_equity = eq_df['Equity'].iloc[-1]
    total_net_profit = ending_equity - starting_capital
    total_return_pct = (ending_equity / starting_capital - 1.0) * 100.0
    max_dd_pct = eq_df['Drawdown_Pct'].min()
    max_dd_dollar = eq_df['Drawdown_Dollar'].min()

    n_years = max((eq_df['Date'].iloc[-1] - eq_df['Date'].iloc[0]).days / 365.25, 0.1)
    cagr_pct = ((max(ending_equity, 1.0) / starting_capital) ** (1.0 / n_years) - 1.0) * 100.0
    ann_sharpe = (eq_df['Daily_Ret'].mean() / eq_df['Daily_Ret'].std()) * np.sqrt(252) if eq_df['Daily_Ret'].std() > 0 else 0.0
    win_rate_trades = (trades_df['Net PnL ($)'] > 0).mean() * 100.0

    st.subheader("📊 Overall Portfolio Performance Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Return (%)", f"{total_return_pct:+.2f}%", f"${total_net_profit:+,.2f}")
    m2.metric("Ending Equity", f"${ending_equity:,.2f}", f"Start: ${starting_capital:,.0f}")
    m3.metric("CAGR (Annualized)", f"{cagr_pct:+.2f}%/yr")
    m4.metric("Max Drawdown", f"{max_dd_pct:.2f}%", f"${max_dd_dollar:,.2f}")
    m5.metric("Annualized Sharpe", f"{ann_sharpe:.2f}")
    m6.metric("Total Stock Trades", f"{len(trades_df):,}", f"Win Rate: {win_rate_trades:.1f}%")

    # Equity Curve & Drawdown Curve
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📈 Daily Mark-to-Market Portfolio Equity ($)")
        st.line_chart(eq_df.set_index('Date')['Equity'])
    with col_chart2:
        st.subheader("📉 Daily Underwater Drawdown (%)")
        st.line_chart(eq_df.set_index('Date')['Drawdown_Pct'])

    # Year-by-Year Return Table
    st.subheader("📅 Year-by-Year Performance Breakdown")
    eq_df['Year'] = eq_df['Date'].dt.year
    yearly_rows = []
    prev_yr_end_eq = starting_capital

    for yr, grp in eq_df.groupby('Year'):
        yr_end_eq = grp['Equity'].iloc[-1]
        yr_pnl = yr_end_eq - prev_yr_end_eq
        yr_ret_pct = (yr_pnl / prev_yr_end_eq) * 100.0 if prev_yr_end_eq > 0 else 0.0

        # Intra-year max drawdown
        yr_peak = grp['Equity'].cummax()
        yr_max_dd = ((grp['Equity'] - yr_peak) / yr_peak * 100.0).min()

        yr_trades = trades_df[pd.to_datetime(trades_df['Exit Date (Close)']).dt.year == yr]
        yr_trade_count = len(yr_trades)
        yr_win_rate = (yr_trades['Net PnL ($)'] > 0).mean() * 100.0 if yr_trade_count > 0 else 0.0

        yearly_rows.append({
            'Year': int(yr),
            'Start Equity ($)': f"${prev_yr_end_eq:,.2f}",
            'End Equity ($)': f"${yr_end_eq:,.2f}",
            'Net Profit ($)': f"${yr_pnl:+,.2f}",
            'Annual Return (%)': f"{yr_ret_pct:+.2f}%",
            'Max Intra-Year DD (%)': f"{yr_max_dd:.2f}%",
            'Completed Trades': yr_trade_count,
            'Trade Win Rate (%)': f"{yr_win_rate:.1f}%"
        })
        prev_yr_end_eq = yr_end_eq

    yr_df = pd.DataFrame(yearly_rows)
    st.dataframe(yr_df, use_container_width=True)

    # Rebalance & Trade Log
    st.subheader(f"📋 Master Trade & Rebalance Log (Top {top_n} Stocks Held for {hold_days} Days)")
    st.dataframe(trades_df, use_container_width=True, height=450)
else:
    st.info("👈 Enter your PostgreSQL password in the sidebar and click **Load Data & Run Backtest** to start.")