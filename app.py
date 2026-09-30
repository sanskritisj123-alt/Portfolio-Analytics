import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Portfolio Analysis Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
}

REQUIRED_COLUMNS = [
    "Ticker",
    "Quantity",
    "Purchase Price",
    "Purchase Date",
]


# ============================================================
# SESSION STATE
# ============================================================

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )

if "portfolio_name" not in st.session_state:
    st.session_state.portfolio_name = "My Portfolio"

if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = {}


# ============================================================
# FORMATTING
# ============================================================

def format_inr(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(value):
        return "N/A"

    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1e7:
        return f"{sign}₹{value / 1e7:.2f} Cr"

    if value >= 1e5:
        return f"{sign}₹{value / 1e5:.2f} L"

    return f"{sign}₹{value:,.0f}"


def format_pct(value):
    try:
        value = float(value)
    except Exception:
        return "N/A"

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.2%}"


def clean_ticker(value):
    return str(value).strip().upper()


# ============================================================
# PORTFOLIO INPUT / VALIDATION
# ============================================================

def normalize_portfolio(df):

    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = df.copy()

    mapping = {}

    for col in df.columns:

        key = str(col).strip().lower()

        if key == "ticker":
            mapping[col] = "Ticker"

        elif key in ["quantity", "qty", "shares"]:
            mapping[col] = "Quantity"

        elif key in [
            "purchase price",
            "buy price",
            "purchase_price",
            "price",
        ]:
            mapping[col] = "Purchase Price"

        elif key in [
            "purchase date",
            "buy date",
            "purchase_date",
            "date",
        ]:
            mapping[col] = "Purchase Date"

    df = df.rename(columns=mapping)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["Ticker"] = df["Ticker"].apply(clean_ticker)

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Purchase Price"] = pd.to_numeric(
        df["Purchase Price"],
        errors="coerce"
    )

    df["Purchase Date"] = pd.to_datetime(
        df["Purchase Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    df = df[
        (df["Ticker"] != "")
        & (df["Quantity"] > 0)
        & (df["Purchase Price"] >= 0)
    ]

    return df.reset_index(drop=True)


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers, period="5y"):

    tickers = tuple(sorted(set(tickers)))

    prices_dict = {}
    metadata = {}
    current_prices = {}
    failures = []

    for ticker in tickers:

        try:

            stock = yf.Ticker(ticker)

            hist = stock.history(
                period=period,
                auto_adjust=True
            )

            if hist is None or hist.empty:

                failures.append(ticker)
                continue

            close = hist["Close"].dropna()

            if close.empty:

                failures.append(ticker)
                continue

            prices_dict[ticker] = close

            current_prices[ticker] = float(
                close.iloc[-1]
            )

            try:
                info = stock.info
            except Exception:
                info = {}

            metadata[ticker] = {
                "Name": info.get(
                    "longName",
                    info.get("shortName", ticker)
                ),

                "Sector": info.get(
                    "sector",
                    "Unknown"
                ),

                "Industry": info.get(
                    "industry",
                    "Unknown"
                ),

                "Market Cap": info.get(
                    "marketCap",
                    np.nan
                ),

                "Currency": info.get(
                    "currency",
                    ""
                ),
            }

        except Exception:

            failures.append(ticker)

    if prices_dict:

        prices = pd.concat(
            prices_dict,
            axis=1
        )

        prices.columns.name = None

    else:

        prices = pd.DataFrame()

    return (
        prices,
        metadata,
        current_prices,
        failures,
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_benchmark_data(tickers, period="5y"):

    result = {}

    for ticker in tickers:

        try:

            data = yf.Ticker(ticker).history(
                period=period,
                auto_adjust=True
            )

            if data is not None and not data.empty:

                result[ticker] = (
                    data["Close"].dropna()
                )

        except Exception:

            continue

    if not result:
        return pd.DataFrame()

    return pd.concat(
        result,
        axis=1
    )


# ============================================================
# RISK / PERFORMANCE FUNCTIONS
# ============================================================

def annualized_return(returns):

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return np.nan

    years = len(returns) / 252

    if years <= 0:
        return np.nan

    total = (
        1 + returns
    ).prod()

    return float(
        total ** (1 / years) - 1
    )


def annualized_volatility(returns):

    returns = pd.Series(
        returns
    ).dropna()

    if len(returns) < 2:
        return np.nan

    return float(
        returns.std() * np.sqrt(252)
    )


def sharpe_ratio(
    returns,
    risk_free_rate=0.06
):

    ann_return = annualized_return(
        returns
    )

    ann_vol = annualized_volatility(
        returns
    )

    if not np.isfinite(ann_return):
        return np.nan

    if not np.isfinite(ann_vol) or ann_vol == 0:
        return np.nan

    return float(
        (ann_return - risk_free_rate)
        / ann_vol
    )


def sortino_ratio(
    returns,
    risk_free_rate=0.06
):

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return np.nan

    ann_return = annualized_return(
        returns
    )

    downside = returns[
        returns < 0
    ]

    if len(downside) < 2:
        return np.nan

    downside_deviation = (
        downside.std() * np.sqrt(252)
    )

    if downside_deviation == 0:
        return np.nan

    return float(
        (ann_return - risk_free_rate)
        / downside_deviation
    )


def max_drawdown(returns):

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return np.nan

    wealth = (
        1 + returns
    ).cumprod()

    peak = wealth.cummax()

    drawdown = (
        wealth / peak - 1
    )

    return float(
        drawdown.min()
    )


def value_at_risk(
    returns,
    confidence=0.95
):

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return np.nan

    return float(
        returns.quantile(
            1 - confidence
        )
    )


def conditional_var(
    returns,
    confidence=0.95
):

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:
        return np.nan

    var = returns.quantile(
        1 - confidence
    )

    tail = returns[
        returns <= var
    ]

    if tail.empty:
        return np.nan

    return float(
        tail.mean()
    )


def beta(
    portfolio_returns,
    benchmark_returns
):

    combined = pd.concat(
        [
            pd.Series(portfolio_returns),
            pd.Series(benchmark_returns),
        ],
        axis=1
    ).dropna()

    if len(combined) < 2:
        return np.nan

    p = combined.iloc[:, 0]
    b = combined.iloc[:, 1]

    if b.var() == 0:
        return np.nan

    return float(
        p.cov(b) / b.var()
    )


# ============================================================
# PORTFOLIO STATISTICS
# ============================================================

def portfolio_stats(
    weights,
    expected_returns,
    covariance,
    risk_free_rate
):

    weights = np.asarray(
        weights,
        dtype=float
    )

    mu = np.asarray(
        expected_returns,
        dtype=float
    )

    cov = np.asarray(
        covariance,
        dtype=float
    )

    ret = float(
        np.dot(weights, mu)
    )

    variance = float(
        weights.T @ cov @ weights
    )

    volatility = np.sqrt(
        max(variance, 0)
    )

    if volatility > 0:

        sharpe = (
            ret - risk_free_rate
        ) / volatility

    else:

        sharpe = np.nan

    return (
        ret,
        volatility,
        sharpe
    )


def optimize_weights(
    expected_returns,
    covariance,
    risk_free_rate,
    objective="max_sharpe",
    target_return=None,
    allow_short=False
):

    mu = np.asarray(
        expected_returns,
        dtype=float
    )

    cov = np.asarray(
        covariance,
        dtype=float
    )

    n = len(mu)

    initial = np.ones(n) / n

    if allow_short:

        bounds = [
            (-1, 1)
        ] * n

    else:

        bounds = [
            (0, 1)
        ] * n

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }
    ]

    if target_return is not None:

        constraints.append(
            {
                "type": "eq",
                "fun": lambda w:
                    np.dot(w, mu)
                    - target_return
            }
        )

    def objective_function(w):

        ret, vol, sr = portfolio_stats(
            w,
            mu,
            cov,
            risk_free_rate
        )

        if objective == "min_volatility":

            return vol

        if objective == "max_sharpe":

            if not np.isfinite(sr):
                return 1e6

            return -sr

        if objective == "target_return":

            return vol

        return vol

    try:

        result = minimize(
            objective_function,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 2000,
                "ftol": 1e-10
            }
        )

        if result.success:

            return result.x

    except Exception:

        pass

    return None


# ============================================================
# HOLDINGS
# ============================================================

def calculate_holdings(
    portfolio_df,
    current_prices,
    metadata
):

    rows = []

    for ticker, group in portfolio_df.groupby(
        "Ticker"
    ):

        quantity = group[
            "Quantity"
        ].sum()

        invested = (
            group["Quantity"]
            * group["Purchase Price"]
        ).sum()

        current_price = current_prices.get(
            ticker,
            np.nan
        )

        if np.isfinite(current_price):

            current_value = (
                quantity
                * current_price
            )

        else:

            current_value = np.nan

        if np.isfinite(current_value):

            pnl = (
                current_value
                - invested
            )

            return_pct = (
                pnl / invested
                if invested != 0
                else np.nan
            )

        else:

            pnl = np.nan
            return_pct = np.nan

        info = metadata.get(
            ticker,
            {}
        )

        rows.append(
            {
                "Ticker": ticker,

                "Company": info.get(
                    "Name",
                    ticker
                ),

                "Sector": info.get(
                    "Sector",
                    "Unknown"
                ),

                "Quantity": quantity,

                "Purchase Price": (
                    invested / quantity
                    if quantity > 0
                    else np.nan
                ),

                "Current Price": current_price,

                "Invested Value": invested,

                "Current Value": current_value,

                "P/L": pnl,

                "Return %": return_pct,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    total_current = result[
        "Current Value"
    ].sum()

    if total_current > 0:

        result["Portfolio Weight"] = (
            result["Current Value"]
            / total_current
        )

    else:

        result["Portfolio Weight"] = np.nan

    return result


def current_market_weights(
    portfolio_df,
    assets,
    current_prices
):

    values = pd.Series(
        0.0,
        index=assets,
        dtype=float
    )

    for ticker in assets:

        rows = portfolio_df[
            portfolio_df["Ticker"] == ticker
        ]

        if rows.empty:
            continue

        quantity = rows[
            "Quantity"
        ].sum()

        price = current_prices.get(
            ticker,
            np.nan
        )

        if np.isfinite(price):

            value = (
                quantity * price
            )

            if value > 0:

                values.loc[ticker] = value

    total = values.sum()

    if total <= 0:
        return None

    return values / total


# ============================================================
# PORTFOLIO RETURN SERIES
# ============================================================

def build_portfolio_returns(
    prices,
    portfolio_df,
    current_prices
):

    if prices.empty:
        return pd.Series(
            dtype=float
        )

    weights = current_market_weights(
        portfolio_df,
        list(prices.columns),
        current_prices
    )

    if weights is None:
        return pd.Series(
            dtype=float
        )

    returns = prices.pct_change()

    portfolio_returns = pd.Series(
        0.0,
        index=returns.index
    )

    for ticker in weights.index:

        if ticker in returns.columns:

            portfolio_returns += (
                returns[ticker].fillna(0)
                * weights[ticker]
            )

    return portfolio_returns.dropna()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Portfolio Terminal"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Portfolio Input",
        "Dashboard",
        "Holdings",
        "Performance",
        "Risk Analysis",
        "Benchmark",
        "Optimization",
        "Efficient Frontier",
        "Correlation",
        "Stock Analysis",
    ]
)

st.sidebar.divider()

if st.sidebar.button(
    "🔄 Refresh Market Data"
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# PORTFOLIO INPUT
# ============================================================

if page == "Portfolio Input":

    st.title(
        "📁 Portfolio Manager"
    )

    st.caption(
        "Create, upload, save and manage your investment portfolio."
    )

    # ========================================================
    # PORTFOLIO NAME
    # ========================================================

    st.subheader(
        "1️⃣ Portfolio Name"
    )

    portfolio_name = st.text_input(
        "Enter portfolio name",
        value=st.session_state.portfolio_name,
        placeholder="Example: My Equity Portfolio"
    )

    if portfolio_name.strip():

        st.session_state.portfolio_name = (
            portfolio_name.strip()
        )

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📂 Upload Portfolio",
            "✏️ Manual Entry",
            "💾 Saved Portfolios",
        ]
    )

    # ========================================================
    # UPLOAD PORTFOLIO
    # ========================================================

    with tab1:

        st.subheader(
            "Upload Portfolio"
        )

        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
            help=(
                "Required columns: Ticker, Quantity, "
                "Purchase Price and Purchase Date"
            )
        )

        if uploaded is not None:

            try:

                if uploaded.name.lower().endswith(
                    ".csv"
                ):

                    raw = pd.read_csv(
                        uploaded
                    )

                else:

                    raw = pd.read_excel(
                        uploaded
                    )

                normalized = normalize_portfolio(
                    raw
                )

                if normalized.empty:

                    st.error(
                        "No valid portfolio rows were found."
                    )

                else:

                    st.session_state.portfolio_df = (
                        normalized
                    )

                    st.success(
                        f"✅ Loaded "
                        f"{len(normalized)} "
                        f"holding rows successfully."
                    )

                    st.dataframe(
                        normalized,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as exc:

                st.error(
                    f"Unable to read the file: {exc}"
                )

    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    with tab2:

        st.subheader(
            "Enter Holdings Manually"
        )

        default_data = pd.DataFrame(
            [
                {
                    "Ticker": "RELIANCE.NS",
                    "Quantity": 10,
                    "Purchase Price": 2500,
                    "Purchase Date": pd.Timestamp(
                        "2025-04-15"
                    ),
                },

                {
                    "Ticker": "TCS.NS",
                    "Quantity": 5,
                    "Purchase Price": 3600,
                    "Purchase Date": pd.Timestamp(
                        "2025-05-10"
                    ),
                },
            ]
        )

        if not st.session_state.portfolio_df.empty:

            manual_source = (
                st.session_state
                .portfolio_df
                .copy()
            )

        else:

            manual_source = (
                default_data.copy()
            )

        manual = st.data_editor(
            manual_source,
            num_rows="dynamic",
            use_container_width=True,

            column_config={

                "Purchase Date":
                    st.column_config.DateColumn(
                        "Purchase Date",
                        format="YYYY-MM-DD"
                    ),

                "Quantity":
                    st.column_config.NumberColumn(
                        "Quantity",
                        min_value=0,
                        step=1
                    ),

                "Purchase Price":
                    st.column_config.NumberColumn(
                        "Purchase Price",
                        min_value=0,
                        step=0.01
                    ),
            },
        )

        if st.button(
            "✅ Update Portfolio",
            type="primary",
            use_container_width=True
        ):

            try:

                normalized = normalize_portfolio(
                    manual
                )

                if normalized.empty:

                    st.error(
                        "Please enter at least one valid holding."
                    )

                else:

                    st.session_state.portfolio_df = (
                        normalized
                    )

                    st.success(
                        "Portfolio updated successfully."
                    )

            except Exception as exc:

                st.error(
                    str(exc)
                )

    # ========================================================
    # SAVED PORTFOLIOS
    # ========================================================

    with tab3:

        st.subheader(
            "💾 Save & Load Portfolio"
        )

        if st.session_state.portfolio_df.empty:

            st.info(
                "Create or upload a portfolio first."
            )

        else:

            st.write(
                "Current Portfolio: "
                f"**{st.session_state.portfolio_name}**"
            )

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            with col1:

                if st.button(
                    "💾 Save Portfolio",
                    type="primary",
                    use_container_width=True
                ):

                    portfolio_to_save = (
                        st.session_state
                        .portfolio_df
                        .copy()
                    )

                    st.session_state.saved_portfolios[
                        st.session_state.portfolio_name
                    ] = portfolio_to_save

                    st.success(
                        f"✅ Portfolio "
                        f"'{st.session_state.portfolio_name}' "
                        f"saved successfully!"
                    )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            with col2:

                csv_data = (
                    st.session_state
                    .portfolio_df
                    .to_csv(
                        index=False
                    )
                )

                st.download_button(
                    label="⬇️ Download Portfolio",
                    data=csv_data,
                    file_name=(
                        st.session_state
                        .portfolio_name
                        .replace(" ", "_")
                        + ".csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

        st.divider()

        # ====================================================
        # LOAD PREVIOUSLY DOWNLOADED PORTFOLIO
        # ====================================================

        st.subheader(
            "📂 Load Previously Downloaded Portfolio"
        )

        saved_file = st.file_uploader(
            "Upload a previously saved portfolio",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
            key="saved_portfolio_uploader"
        )

        if saved_file is not None:

            try:

                if saved_file.name.lower().endswith(
                    ".csv"
                ):

                    saved_raw = pd.read_csv(
                        saved_file
                    )

                else:

                    saved_raw = pd.read_excel(
                        saved_file
                    )

                saved_normalized = normalize_portfolio(
                    saved_raw
                )

                if saved_normalized.empty:

                    st.error(
                        "The saved file does not contain "
                        "valid portfolio data."
                    )

                else:

                    st.session_state.portfolio_df = (
                        saved_normalized
                    )

                    filename = (
                        saved_file.name
                    )

                    portfolio_name_from_file = (
                        filename.rsplit(
                            ".",
                            1
                        )[0]
                    )

                    st.session_state.portfolio_name = (
                        portfolio_name_from_file
                    )

                    st.success(
                        f"✅ Portfolio "
                        f"'{portfolio_name_from_file}' "
                        f"loaded successfully!"
                    )

            except Exception as exc:

                st.error(
                    f"Unable to load portfolio: {exc}"
                )

        st.divider()

        # ====================================================
        # SESSION SAVED PORTFOLIOS
        # ====================================================

        st.subheader(
            "📚 Portfolios Saved in This Session"
        )

        if st.session_state.saved_portfolios:

            saved_names = list(
                st.session_state
                .saved_portfolios
                .keys()
            )

            selected_portfolio = st.selectbox(
                "Select Portfolio",
                saved_names
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "📥 Load Selected Portfolio",
                    use_container_width=True
                ):

                    st.session_state.portfolio_df = (
                        st.session_state
                        .saved_portfolios[
                            selected_portfolio
                        ]
                        .copy()
                    )

                    st.session_state.portfolio_name = (
                        selected_portfolio
                    )

                    st.success(
                        f"✅ "
                        f"'{selected_portfolio}' "
                        f"loaded successfully!"
                    )

            with col2:

                if st.button(
                    "🗑️ Delete Selected Portfolio",
                    use_container_width=True
                ):

                    del (
                        st.session_state
                        .saved_portfolios[
                            selected_portfolio
                        ]
                    )

                    st.success(
                        f"'{selected_portfolio}' "
                        f"deleted."
                    )

                    st.rerun()

        else:

            st.info(
                "No portfolios have been saved "
                "during this session."
            )

    # ========================================================
    # CURRENT PORTFOLIO PREVIEW
    # ========================================================

    if not st.session_state.portfolio_df.empty:

        st.divider()

        st.subheader(
            f"📊 Current Portfolio — "
            f"{st.session_state.portfolio_name}"
        )

        st.dataframe(
            st.session_state.portfolio_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Portfolio Summary"
        )

        summary_col1, summary_col2, summary_col3 = (
            st.columns(3)
        )

        summary_col1.metric(
            "Holding Rows",
            len(
                st.session_state.portfolio_df
            )
        )

        summary_col2.metric(
            "Unique Stocks",
            st.session_state
            .portfolio_df["Ticker"]
            .nunique()
        )

        total_investment = (
            st.session_state
            .portfolio_df["Quantity"]
            *
            st.session_state
            .portfolio_df["Purchase Price"]
        ).sum()

        summary_col3.metric(
            "Total Investment",
            format_inr(
                total_investment
            )
        )

        st.divider()

        if st.button(
            "🗑️ Clear Current Portfolio",
            use_container_width=True
        ):

            st.session_state.portfolio_df = (
                pd.DataFrame(
                    columns=REQUIRED_COLUMNS
                )
            )

            st.success(
                "Current portfolio cleared."
            )

            st.rerun()

    st.stop()


# ============================================================
# GLOBAL PORTFOLIO DATA
# ============================================================

portfolio_df = (
    st.session_state.portfolio_df
)

if portfolio_df.empty:

    st.warning(
        "No portfolio has been loaded. "
        "Go to Portfolio Input first."
    )

    st.stop()


tickers = (
    portfolio_df["Ticker"]
    .unique()
    .tolist()
)


with st.spinner(
    "Loading market data..."
):

    prices, metadata, current_prices, failures = (
        fetch_market_data(
            tuple(tickers),
            period="5y"
        )
    )


if failures:

    st.warning(
        "Unable to retrieve data for: "
        + ", ".join(failures)
    )


if prices.empty:

    st.error(
        "No market data could be retrieved. "
        "Check your ticker symbols and internet connection."
    )

    st.stop()


holdings = calculate_holdings(
    portfolio_df,
    current_prices,
    metadata
)


portfolio_returns = build_portfolio_returns(
    prices,
    portfolio_df,
    current_prices
)


if portfolio_returns.empty:

    st.error(
        "Unable to construct portfolio return data."
    )

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "📊 Portfolio Dashboard"
    )

    total_invested = holdings[
        "Invested Value"
    ].sum()

    total_current = holdings[
        "Current Value"
    ].sum()

    total_pnl = (
        total_current
        - total_invested
    )

    total_return = (
        total_pnl / total_invested
        if total_invested != 0
        else np.nan
    )

    volatility = annualized_volatility(
        portfolio_returns
    )

    sharpe = sharpe_ratio(
        portfolio_returns
    )

    drawdown = max_drawdown(
        portfolio_returns
    )

    best = (
        holdings.loc[
            holdings["Return %"].idxmax(),
            "Ticker"
        ]
        if not holdings.empty
        else "N/A"
    )

    worst = (
        holdings.loc[
            holdings["Return %"].idxmin(),
            "Ticker"
        ]
        if not holdings.empty
        else "N/A"
    )

    cols = st.columns(6)

    metrics = [
        (
            "Total Invested",
            format_inr(
                total_invested
            )
        ),

        (
            "Current Value",
            format_inr(
                total_current
            )
        ),

        (
            "P/L",
            format_inr(
                total_pnl
            )
        ),

        (
            "Return",
            format_pct(
                total_return
            )
        ),

        (
            "Volatility",
            format_pct(
                volatility
            )
        ),

        (
            "Sharpe",
            (
                f"{sharpe:.2f}"
                if np.isfinite(sharpe)
                else "N/A"
            )
        ),
    ]

    for col, (label, value) in zip(
        cols,
        metrics
    ):

        col.metric(
            label,
            value
        )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Portfolio Value"
        )

        value_df = pd.DataFrame(
            {
                "Metric": [
                    "Invested Value",
                    "Current Value",
                ],

                "Value": [
                    total_invested,
                    total_current,
                ],
            }
        )

        fig = px.bar(
            value_df,
            x="Metric",
            y="Value",
            text="Value",
            title="Invested vs Current Value"
        )

        fig.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Value (₹)",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader(
            "Portfolio Allocation"
        )

        allocation = holdings[
            [
                "Ticker",
                "Current Value"
            ]
        ].dropna()

        if not allocation.empty:

            fig = px.pie(
                allocation,
                names="Ticker",
                values="Current Value",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader(
        "Portfolio Highlights"
    )

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Holdings",
        len(holdings)
    )

    h2.metric(
        "Best Holding",
        best
    )

    h3.metric(
        "Worst Holding",
        worst
    )

    st.metric(
        "Maximum Drawdown",
        format_pct(
            drawdown
        )
    )


# ============================================================
# HOLDINGS
# ============================================================

elif page == "Holdings":

    st.title(
        "📋 Holdings"
    )

    if holdings.empty:

        st.info(
            "No holdings available."
        )

        st.stop()

    display = holdings.copy()

    for col in [
        "Purchase Price",
        "Current Price",
        "Invested Value",
        "Current Value",
        "P/L",
    ]:

        display[col] = display[
            col
        ].apply(
            format_inr
        )

    display["Return %"] = display[
        "Return %"
    ].apply(
        format_pct
    )

    display["Portfolio Weight"] = display[
        "Portfolio Weight"
    ].apply(
        format_pct
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "Performance":

    st.title(
        "📈 Performance Analysis"
    )

    period = st.selectbox(
        "Analysis Period",
        [
            "1M",
            "3M",
            "6M",
            "1Y",
            "3Y",
            "5Y",
            "Max",
        ],
        index=3
    )

    period_days = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "3Y": 1095,
        "5Y": 1825,
        "Max": None,
    }

    if period_days[period] is None:

        period_returns = (
            portfolio_returns.copy()
        )

    else:

        cutoff = (
            portfolio_returns.index.max()
            -
            pd.Timedelta(
                days=period_days[period]
            )
        )

        period_returns = (
            portfolio_returns[
                portfolio_returns.index >= cutoff
            ]
        )

    if period_returns.empty:

        st.warning(
            "Insufficient data for this period."
        )

        st.stop()

    cumulative = (
        1 + period_returns
    ).cumprod() - 1

    drawdown_series = (
        (1 + period_returns).cumprod()
        /
        (1 + period_returns)
        .cumprod()
        .cummax()
        - 1
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Cumulative Return",
        format_pct(
            cumulative.iloc[-1]
        )
    )

    m2.metric(
        "Annualized Return",
        format_pct(
            annualized_return(
                period_returns
            )
        )
    )

    m3.metric(
        "Volatility",
        format_pct(
            annualized_volatility(
                period_returns
            )
        )
    )

    current_sharpe = sharpe_ratio(
        period_returns
    )

    m4.metric(
        "Sharpe Ratio",
        (
            f"{current_sharpe:.2f}"
            if np.isfinite(current_sharpe)
            else "N/A"
        )
    )

    st.subheader(
        "Cumulative Return"
    )

    cumulative_df = pd.DataFrame(
        {
            "Date": cumulative.index,
            "Cumulative Return": cumulative.values,
        }
    )

    fig = px.line(
        cumulative_df,
        x="Date",
        y="Cumulative Return",
        title="Portfolio Cumulative Return"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Drawdown"
        )

        dd_df = pd.DataFrame(
            {
                "Date": drawdown_series.index,
                "Drawdown": drawdown_series.values,
            }
        )

        fig = px.area(
            dd_df,
            x="Date",
            y="Drawdown",
            title="Portfolio Drawdown"
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader(
            "Daily Returns"
        )

        ret_df = pd.DataFrame(
            {
                "Date": period_returns.index,
                "Daily Return": period_returns.values,
            }
        )

        fig = px.histogram(
            ret_df,
            x="Daily Return",
            nbins=50,
            title="Distribution of Daily Returns"
        )

        fig.update_xaxes(
            tickformat=".1%"
        )

        fig.update_layout(
            yaxis_title="Count",
            xaxis_title="Daily Return"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Count represents the number of trading days "
            "falling within each daily-return range."
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.title(
        "⚠️ Risk Analysis"
    )

    benchmark_ticker = st.selectbox(
        "Beta Benchmark",
        list(BENCHMARKS.keys()),
        index=0
    )

    risk_free_rate = st.number_input(
        "Risk-free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.25
    ) / 100

    benchmark_data = fetch_benchmark_data(
        (
            BENCHMARKS[
                benchmark_ticker
            ],
        ),
        period="5y"
    )

    benchmark_returns = pd.Series(
        dtype=float
    )

    if not benchmark_data.empty:

        benchmark_returns = (
            benchmark_data.iloc[:, 0]
            .pct_change()
            .dropna()
        )

    metrics = {

        "Annualized Return":
            annualized_return(
                portfolio_returns
            ),

        "Annualized Volatility":
            annualized_volatility(
                portfolio_returns
            ),

        "Sharpe Ratio":
            sharpe_ratio(
                portfolio_returns,
                risk_free_rate
            ),

        "Sortino Ratio":
            sortino_ratio(
                portfolio_returns,
                risk_free_rate
            ),

        "Maximum Drawdown":
            max_drawdown(
                portfolio_returns
            ),

        "95% Daily VaR":
            value_at_risk(
                portfolio_returns
            ),

        "95% Daily CVaR":
            conditional_var(
                portfolio_returns
            ),

        "Beta":
            beta(
                portfolio_returns,
                benchmark_returns
            )
            if not benchmark_returns.empty
            else np.nan,
    }

    cols = st.columns(4)

    for i, (
        label,
        value
    ) in enumerate(
        metrics.items()
    ):

        if label in [
            "Sharpe Ratio",
            "Sortino Ratio",
            "Beta",
        ]:

            display = (
                f"{value:.2f}"
                if np.isfinite(value)
                else "N/A"
            )

        else:

            display = format_pct(
                value
            )

        cols[
            i % 4
        ].metric(
            label,
            display
        )

    st.subheader(
        "Risk Summary"
    )

    risk_df = pd.DataFrame(
        {
            "Metric": list(
                metrics.keys()
            ),

            "Value": list(
                metrics.values()
            ),
        }
    )

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BENCHMARK
# ============================================================

elif page == "Benchmark":

    st.title(
        "📊 Benchmark Comparison"
    )

    benchmark_data = fetch_benchmark_data(
        tuple(
            BENCHMARKS.values()
        ),
        period="5y"
    )

    if benchmark_data.empty:

        st.warning(
            "Benchmark data could not be retrieved."
        )

        st.stop()

    combined = pd.DataFrame(
        {
            "Portfolio":
                portfolio_returns
        }
    )

    for name, ticker in BENCHMARKS.items():

        if ticker in benchmark_data.columns:

            combined[name] = (
                benchmark_data[ticker]
                .pct_change()
            )

    combined = combined.dropna(
        how="all"
    )

    normalized = (
        1 + combined.fillna(0)
    ).cumprod()

    normalized = (
        normalized
        / normalized.iloc[0]
    )

    fig = go.Figure()

    for column in normalized.columns:

        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode="lines",
                name=column
            )
        )

    fig.update_layout(
        title="Growth of ₹1",
        xaxis_title="Date",
        yaxis_title="Growth"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    rows = []

    for column in normalized.columns:

        series = combined[
            column
        ].dropna()

        rows.append(
            {
                "Benchmark": column,

                "Cumulative Return": (
                    (1 + series).prod() - 1
                    if not series.empty
                    else np.nan
                ),

                "Annualized Return":
                    annualized_return(
                        series
                    ),

                "Volatility":
                    annualized_volatility(
                        series
                    ),

                "Sharpe":
                    sharpe_ratio(
                        series
                    ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    st.subheader(
        "Benchmark Statistics"
    )

    st.dataframe(
        comparison.style.format(
            {
                "Cumulative Return": "{:.2%}",
                "Annualized Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# OPTIMIZATION
# ============================================================

elif page == "Optimization":

    st.title(
        "⚙️ Portfolio Optimization"
    )

    st.caption(
        "Mean-variance optimization using historical return "
        "and covariance estimates."
    )

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    if len(returns.columns) < 2:

        st.warning(
            "At least two securities are required."
        )

        st.stop()

    min_observations = 100

    valid_assets = [
        col
        for col in returns.columns
        if returns[col].count()
        >= min_observations
    ]

    if len(valid_assets) < 2:

        st.warning(
            "Not enough securities have sufficient "
            "historical data."
        )

        st.stop()

    returns = returns[
        valid_assets
    ]

    mu = returns.mean() * 252

    covariance = (
        returns.cov() * 252
    )

    risk_free_rate = st.number_input(
        "Risk-free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.25,
        key="optimization_rf"
    ) / 100

    allow_short = st.checkbox(
        "Allow short selling",
        value=False
    )

    method = st.selectbox(
        "Optimization Objective",
        [
            "Maximum Sharpe Ratio",
            "Minimum Volatility",
            "Target Return",
        ]
    )

    target_return = None

    if method == "Target Return":

        target_return = st.slider(
            "Target Annualized Return (%)",
            min_value=float(
                mu.min() * 100
            ),
            max_value=float(
                mu.max() * 100
            ),
            value=float(
                mu.mean() * 100
            ),
            step=0.25
        ) / 100

    objective_map = {

        "Maximum Sharpe Ratio":
            "max_sharpe",

        "Minimum Volatility":
            "min_volatility",

        "Target Return":
            "target_return",
    }

    weights = optimize_weights(
        mu.values,
        covariance.values,
        risk_free_rate,
        objective=objective_map[
            method
        ],
        target_return=target_return,
        allow_short=allow_short
    )

    if weights is None:

        st.error(
            "Optimization failed. "
            "Try different constraints."
        )

        st.stop()

    opt_return, opt_vol, opt_sharpe = (
        portfolio_stats(
            weights,
            mu.values,
            covariance.values,
            risk_free_rate
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Expected Return",
        format_pct(
            opt_return
        )
    )

    c2.metric(
        "Volatility",
        format_pct(
            opt_vol
        )
    )

    c3.metric(
        "Sharpe Ratio",
        (
            f"{opt_sharpe:.2f}"
            if np.isfinite(
                opt_sharpe
            )
            else "N/A"
        )
    )

    weights_df = pd.DataFrame(
        {
            "Ticker": valid_assets,
            "Weight": weights,
        }
    )

    weights_df = weights_df[
        weights_df["Weight"].abs()
        > 1e-6
    ].sort_values(
        "Weight",
        ascending=False
    )

    st.subheader(
        "Optimized Allocation"
    )

    st.dataframe(
        weights_df.style.format(
            {
                "Weight": "{:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        weights_df,
        x="Ticker",
        y="Weight",
        title="Optimized Portfolio Weights"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EFFICIENT FRONTIER
# ============================================================

elif page == "Efficient Frontier":

    st.title(
        "📈 Efficient Frontier"
    )

    st.caption(
        "Mean-variance risk–return analysis based on "
        "historical annualized returns and covariance."
    )

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    valid_assets = [
        ticker
        for ticker in returns.columns
        if returns[ticker].count() >= 100
    ]

    if len(valid_assets) < 2:

        st.warning(
            "At least two securities with sufficient "
            "historical data are required."
        )

        st.stop()

    returns = returns[
        valid_assets
    ]

    mu = returns.mean() * 252

    covariance = (
        returns.cov() * 252
    )

    mu = mu.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    valid_assets = [
        asset
        for asset in valid_assets
        if asset in mu.index
    ]

    covariance = covariance.loc[
        valid_assets,
        valid_assets
    ]

    mu = mu.loc[
        valid_assets
    ]

    control1, control2 = st.columns(2)

    with control1:

        risk_free_rate = st.number_input(
            "Risk-free Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.0,
            step=0.25,
            key="frontier_rf"
        ) / 100

    with control2:

        random_count = st.slider(
            "Random Portfolios",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000
        )

    n_assets = len(
        valid_assets
    )

    initial_weights = (
        np.ones(n_assets)
        / n_assets
    )

    bounds = [
        (0, 1)
        for _ in range(n_assets)
    ]

    base_constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1
        }
    ]

    def stats(weights):

        return portfolio_stats(
            weights,
            mu.values,
            covariance.values,
            risk_free_rate
        )

    def solve(
        objective,
        target=None
    ):

        constraints = list(
            base_constraints
        )

        if target is not None:

            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w:
                        np.dot(
                            w,
                            mu.values
                        )
                        - target
                }
            )

        try:

            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": 2000,
                    "ftol": 1e-10,
                }
            )

            if result.success:

                return result.x

        except Exception:

            return None

        return None

    # ========================================================
    # MINIMUM VOLATILITY
    # ========================================================

    min_vol_weights = solve(
        lambda w:
            stats(w)[1]
    )

    if min_vol_weights is None:

        st.error(
            "Unable to calculate the "
            "minimum-volatility portfolio."
        )

        st.stop()

    (
        min_vol_return,
        min_volatility,
        min_vol_sharpe
    ) = stats(
        min_vol_weights
    )

    # ========================================================
    # MAXIMUM SHARPE
    # ========================================================

    max_sharpe_weights = solve(
        lambda w:
            -(
                stats(w)[2]
                if np.isfinite(
                    stats(w)[2]
                )
                else -1e6
            )
    )

    if max_sharpe_weights is None:

        st.error(
            "Unable to calculate the "
            "maximum-Sharpe portfolio."
        )

        st.stop()

    (
        max_sharpe_return,
        max_sharpe_volatility,
        max_sharpe
    ) = stats(
        max_sharpe_weights
    )

    # ========================================================
    # CURRENT PORTFOLIO
    # ========================================================

    current_weights = current_market_weights(
        portfolio_df,
        valid_assets,
        current_prices
    )

    if current_weights is not None:

        current_weights = (
            current_weights
            .reindex(
                valid_assets
            )
            .fillna(0)
        )

        (
            current_return,
            current_volatility,
            current_sharpe
        ) = stats(
            current_weights.values
        )

        has_current = True

    else:

        current_return = np.nan
        current_volatility = np.nan
        current_sharpe = np.nan

        has_current = False

    # ========================================================
    # EFFICIENT FRONTIER CURVE
    # ========================================================

    frontier_target_returns = np.linspace(
        min_vol_return,
        max(
            float(mu.max()),
            min_vol_return
        ),
        100
    )

    frontier_x = []
    frontier_y = []

    for target in frontier_target_returns:

        frontier_weights = solve(
            lambda w:
                stats(w)[1],
            target=target
        )

        if frontier_weights is None:
            continue

        frontier_ret, frontier_vol, _ = (
            stats(
                frontier_weights
            )
        )

        if (
            np.isfinite(
                frontier_ret
            )
            and
            np.isfinite(
                frontier_vol
            )
        ):

            frontier_x.append(
                frontier_vol
            )

            frontier_y.append(
                frontier_ret
            )

    frontier_x = np.asarray(
        frontier_x
    )

    frontier_y = np.asarray(
        frontier_y
    )

    if len(frontier_x) > 1:

        order = np.argsort(
            frontier_x
        )

        frontier_x = (
            frontier_x[order]
        )

        frontier_y = (
            frontier_y[order]
        )

    # ========================================================
    # RANDOM PORTFOLIOS
    # ========================================================

    rng = np.random.default_rng(
        42
    )

    random_x = []
    random_y = []
    random_sharpe = []
    random_weights = []

    for _ in range(
        random_count
    ):

        weights = rng.dirichlet(
            np.ones(n_assets)
        )

        ret, vol, sr = stats(
            weights
        )

        if (
            np.isfinite(ret)
            and
            np.isfinite(vol)
            and
            np.isfinite(sr)
        ):

            random_x.append(vol)
            random_y.append(ret)
            random_sharpe.append(sr)
            random_weights.append(weights)

    random_x = np.asarray(
        random_x
    )

    random_y = np.asarray(
        random_y
    )

    random_sharpe = np.asarray(
        random_sharpe
    )

    random_weights = np.asarray(
        random_weights
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    st.subheader(
        "Portfolio Comparison"
    )

    k1, k2, k3 = st.columns(3)

    with k1:

        if has_current:

            st.metric(
                "⭐ Current Portfolio",
                format_pct(
                    current_return
                ),
                f"Risk {format_pct(current_volatility)}"
            )

        else:

            st.metric(
                "⭐ Current Portfolio",
                "N/A"
            )

    with k2:

        st.metric(
            "★ Maximum Sharpe",
            format_pct(
                max_sharpe_return
            ),
            f"Risk {format_pct(max_sharpe_volatility)}"
        )

    with k3:

        st.metric(
            "◆ Minimum Volatility",
            format_pct(
                min_vol_return
            ),
            f"Risk {format_pct(min_volatility)}"
        )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "Risk–Return Map"
    )

    if (
        "efficient_frontier_chart_version"
        not in st.session_state
    ):

        st.session_state.efficient_frontier_chart_version = 0

    clear_col, _ = st.columns(
        [1, 5]
    )

    with clear_col:

        if st.button(
            "✕ Clear Selection",
            key="clear_frontier_selection"
        ):

            st.session_state.efficient_frontier_chart_version += 1

            st.rerun()

    fig = go.Figure()

    random_point_data = np.column_stack(
        (
            np.arange(
                len(
                    random_sharpe
                )
            ),
            random_sharpe
        )
    )

    fig.add_trace(
        go.Scatter(
            x=random_x,
            y=random_y,
            mode="markers",
            name="Random Portfolios",

            marker=dict(
                size=6,
                opacity=0.35
            ),

            customdata=random_point_data,

            hovertemplate=(
                "<b>Random Portfolio #%{customdata[0]}</b><br>"
                "Risk: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                "Sharpe: %{customdata[1]:.2f}"
                "<extra></extra>"
            )
        )
    )

    if len(frontier_x) > 1:

        fig.add_trace(
            go.Scatter(
                x=frontier_x,
                y=frontier_y,
                mode="lines",
                name="Efficient Frontier",

                line=dict(
                    width=4,
                    shape="spline"
                ),

                hovertemplate=(
                    "<b>Efficient Frontier</b><br>"
                    "Risk: %{x:.2%}<br>"
                    "Return: %{y:.2%}"
                    "<extra></extra>"
                )
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[
                max_sharpe_volatility
            ],

            y=[
                max_sharpe_return
            ],

            mode="markers",
            name="★ Maximum Sharpe",

            marker=dict(
                size=18,
                symbol="star"
            ),

            hovertemplate=(
                "<b>Maximum Sharpe</b><br>"
                "Risk: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                f"Sharpe: {max_sharpe:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                min_volatility
            ],

            y=[
                min_vol_return
            ],

            mode="markers",
            name="◆ Minimum Volatility",

            marker=dict(
                size=16,
                symbol="diamond"
            ),

            hovertemplate=(
                "<b>Minimum Volatility</b><br>"
                "Risk: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                f"Sharpe: {min_vol_sharpe:.2f}"
                "<extra></extra>"
            )
        )
    )

    if has_current:

        fig.add_trace(
            go.Scatter(
                x=[
                    current_volatility
                ],

                y=[
                    current_return
                ],

                mode="markers",
                name="⭐ Current Portfolio",

                marker=dict(
                    size=20,
                    symbol="diamond-open",
                    line=dict(
                        width=3
                    )
                ),

                hovertemplate=(
                    "<b>⭐ Current Portfolio</b><br>"
                    "Risk: %{x:.2%}<br>"
                    "Return: %{y:.2%}<br>"
                    f"Sharpe: {current_sharpe:.2f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        height=680,

        xaxis_title=(
            "Annualized Risk (Volatility)"
        ),

        yaxis_title=(
            "Annualized Expected Return"
        ),

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),

        margin=dict(
            l=30,
            r=30,
            t=80,
            b=40
        )
    )

    fig.update_xaxes(
        tickformat=".0%",
        showgrid=True
    )

    fig.update_yaxes(
        tickformat=".0%",
        showgrid=True
    )

    chart_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=(
            "efficient_frontier_chart_"
            f"{st.session_state.efficient_frontier_chart_version}"
        )
    )

    # ========================================================
    # SELECTED RANDOM PORTFOLIO
    # ========================================================

    selected_random_index = None

    try:

        selected_points = (
            chart_event.selection.points
        )

        for point in selected_points:

            if point.get(
                "curve_number"
            ) == 0:

                point_index = point.get(
                    "point_index"
                )

                if point_index is not None:

                    selected_random_index = int(
                        point_index
                    )

                    break

    except Exception:

        selected_random_index = None

    if (
        selected_random_index is not None
        and
        0 <= selected_random_index
        < len(random_weights)
    ):

        selected_weights = (
            random_weights[
                selected_random_index
            ]
        )

        (
            selected_return,
            selected_volatility,
            selected_sharpe
        ) = stats(
            selected_weights
        )

        st.subheader(
            f"Selected Random Portfolio "
            f"#{selected_random_index}"
        )

        sw1, sw2, sw3 = st.columns(3)

        with sw1:

            st.metric(
                "Expected Return",
                format_pct(
                    selected_return
                )
            )

        with sw2:

            st.metric(
                "Risk (Volatility)",
                format_pct(
                    selected_volatility
                )
            )

        with sw3:

            st.metric(
                "Sharpe Ratio",
                (
                    f"{selected_sharpe:.2f}"
                    if np.isfinite(
                        selected_sharpe
                    )
                    else "N/A"
                )
            )

        selected_weights_df = pd.DataFrame(
            {
                "Ticker":
                    valid_assets,

                "Selected Portfolio Weight":
                    selected_weights,
            }
        )

        selected_weights_df = (
            selected_weights_df[
                selected_weights_df[
                    "Selected Portfolio Weight"
                ].abs()
                > 0.0001
            ]
            .sort_values(
                "Selected Portfolio Weight",
                ascending=False
            )
        )

        st.dataframe(
            selected_weights_df.style.format(
                {
                    "Selected Portfolio Weight":
                        "{:.2%}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Click any random portfolio point in the chart "
            "to view its exact asset weights."
        )

    else:

        st.info(
            "Click a point in the Random Portfolios "
            "cloud above to see its portfolio weights."
        )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    st.subheader(
        "Portfolio Comparison"
    )

    comparison_rows = []

    if has_current:

        comparison_rows.append(
            {
                "Portfolio":
                    "⭐ Current Portfolio",

                "Expected Return":
                    current_return,

                "Risk":
                    current_volatility,

                "Sharpe Ratio":
                    current_sharpe,
            }
        )

    comparison_rows.extend(
        [
            {
                "Portfolio":
                    "★ Maximum Sharpe",

                "Expected Return":
                    max_sharpe_return,

                "Risk":
                    max_sharpe_volatility,

                "Sharpe Ratio":
                    max_sharpe,
            },

            {
                "Portfolio":
                    "◆ Minimum Volatility",

                "Expected Return":
                    min_vol_return,

                "Risk":
                    min_volatility,

                "Sharpe Ratio":
                    min_vol_sharpe,
            },
        ]
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Expected Return":
                    "{:.2%}",

                "Risk":
                    "{:.2%}",

                "Sharpe Ratio":
                    "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # OPTIMIZED WEIGHTS
    # ========================================================

    st.subheader(
        "Optimized Portfolio Weights"
    )

    weights_table = pd.DataFrame(
        {
            "Ticker":
                valid_assets,

            "Maximum Sharpe":
                max_sharpe_weights,

            "Minimum Volatility":
                min_vol_weights,
        }
    )

    if has_current:

        weights_table[
            "Current Portfolio"
        ] = current_weights.values

    weights_table = weights_table[
        (
            weights_table.drop(
                columns=["Ticker"]
            )
            .abs()
            .max(axis=1)
            > 0.0001
        )
    ]

    formatting = {
        "Maximum Sharpe":
            "{:.2%}",

        "Minimum Volatility":
            "{:.2%}",
    }

    if has_current:

        formatting[
            "Current Portfolio"
        ] = "{:.2%}"

    st.dataframe(
        weights_table.style.format(
            formatting
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CORRELATION
# ============================================================

elif page == "Correlation":

    st.title(
        "🔗 Correlation Analysis"
    )

    returns = (
        prices
        .pct_change()
        .dropna()
    )

    if returns.shape[1] < 2:

        st.warning(
            "At least two securities are required."
        )

        st.stop()

    correlation = returns.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Return Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Covariance Matrix"
    )

    covariance = (
        returns.cov() * 252
    )

    st.dataframe(
        covariance.style.format(
            "{:.4f}"
        ),
        use_container_width=True
    )


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif page == "Stock Analysis":

    st.title(
        "🔎 Individual Stock Analysis"
    )

    available_tickers = [
        ticker
        for ticker in prices.columns
        if ticker in current_prices
    ]

    if not available_tickers:

        st.warning(
            "No stocks with valid market data are available."
        )

        st.stop()

    ticker = st.selectbox(
        "Select Stock",
        available_tickers
    )

    stock_prices = (
        prices[ticker]
        .dropna()
        .sort_index()
    )

    if stock_prices.empty:

        st.warning(
            f"No historical price data is "
            f"available for {ticker}."
        )

        st.stop()

    info = metadata.get(
        ticker,
        {}
    )

    company_name = info.get(
        "Name",
        ticker
    )

    sector = info.get(
        "Sector",
        "Unknown"
    )

    industry = info.get(
        "Industry",
        "Unknown"
    )

    market_cap = info.get(
        "Market Cap",
        np.nan
    )

    current_price = current_prices.get(
        ticker,
        np.nan
    )

    stock_returns = (
        stock_prices
        .pct_change()
        .dropna()
    )

    st.subheader(
        company_name
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Current Price",
            format_inr(
                current_price
            )
        )

    with c2:

        st.metric(
            "Sector",
            sector
        )

    with c3:

        st.metric(
            "Market Cap",
            format_inr(
                market_cap
            )
        )

    with c4:

        st.metric(
            "Annualized Volatility",
            format_pct(
                annualized_volatility(
                    stock_returns
                )
            )
        )

    st.caption(
        f"Ticker: {ticker}  •  "
        f"Industry: {industry}"
    )

    st.subheader(
        "Price History"
    )

    price_df = (
        stock_prices
        .rename("Price")
        .reset_index()
    )

    price_df.columns = [
        "Date",
        "Price"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["Price"],
            mode="lines",
            name=ticker,

            hovertemplate=(
                "Date: %{x|%d %b %Y}<br>"
                "Price: %{y:,.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        height=500,
        title=(
            f"{ticker} Historical Price"
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Stock Statistics"
    )

    stock_annual_return = (
        annualized_return(
            stock_returns
        )
    )

    stock_volatility = (
        annualized_volatility(
            stock_returns
        )
    )

    stock_sharpe = (
        sharpe_ratio(
            stock_returns
        )
    )

    stock_drawdown = (
        max_drawdown(
            stock_returns
        )
    )

    stock_var = (
        value_at_risk(
            stock_returns
        )
    )

    stock_cvar = (
        conditional_var(
            stock_returns
        )
    )

    stat_cols = st.columns(6)

    stat_cols[0].metric(
        "Annualized Return",
        format_pct(
            stock_annual_return
        )
    )

    stat_cols[1].metric(
        "Volatility",
        format_pct(
            stock_volatility
        )
    )

    stat_cols[2].metric(
        "Sharpe Ratio",
        (
            f"{stock_sharpe:.2f}"
            if np.isfinite(
                stock_sharpe
            )
            else "N/A"
        )
    )

    stat_cols[3].metric(
        "Max Drawdown",
        format_pct(
            stock_drawdown
        )
    )

    stat_cols[4].metric(
        "95% VaR",
        format_pct(
            stock_var
        )
    )

    stat_cols[5].metric(
        "95% CVaR",
        format_pct(
            stock_cvar
        )
    )

    statistics = pd.DataFrame(
        {
            "Metric": [
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Maximum Drawdown",
                "95% Daily VaR",
                "95% Daily CVaR",
            ],

            "Value": [
                stock_annual_return,
                stock_volatility,
                stock_sharpe,
                stock_drawdown,
                stock_var,
                stock_cvar,
            ]
        }
    )

    def format_stock_stat(row):

        value = row["Value"]

        if not np.isfinite(value):
            return "N/A"

        if row["Metric"] == "Sharpe Ratio":

            return f"{value:.2f}"

        return f"{value:.2%}"

    statistics["Value"] = statistics.apply(
        format_stock_stat,
        axis=1
    )

    st.dataframe(
        statistics,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Portfolio Analysis Terminal"
)

st.sidebar.caption(
    "Market data powered by Yahoo Finance."
)