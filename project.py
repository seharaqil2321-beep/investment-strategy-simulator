import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Investment Strategy Simulator", layout="wide")

# ---------------- AUTO REFRESH ----------------
st_autorefresh(interval=10000, key="refresh")

# ---------------- TITLE ----------------
st.title(" Personal Investment Strategy Simulator (Live)")
st.caption(f"Live Market Analysis | Last Updated: {datetime.now().strftime('%H:%M:%S')}")

# ---------------- SIDEBAR INPUTS ----------------
with st. sidebar:
    st.header("👤 User Profile")
    currency = st.selectbox("Currency", ["USD", "PKR", "EUR"])
    income = st.number_input("Monthly Income", min_value=0)
    savings = st.number_input("Current Savings", min_value=0)
    monthly_invest = st.number_input("Monthly Investment", min_value=0)
    years = st.slider("Investment Duration (Years)", 1, 30, 5)

    st.divider()
    st.header("🔍 Asset Selection")
    stock_symbol = st.text_input("Stock (e.g., AAPL)", "AAPL")
    crypto_symbol = st.text_input("Crypto (e.g., BTCUSDT)", "BTCUSDT")
    bond_symbol = st.text_input("Bond ETF (e.g., BND, TLT)", "BND")

# ---------------- CURRENCY CONVERSION ----------------
rates = {"USD": 1, "PKR": 278.34, "EUR": 0.86}
to_usd = 1 / rates[currency]

savings_usd = savings * to_usd
monthly_invest_usd = monthly_invest * to_usd

try:
    # 1. FETCH LIVE DATA
    stock_data = yf.Ticker(stock_symbol).history(period="1y")
    bond_data = yf.Ticker(bond_symbol).history(period="1y")

    # Guard rail: Check if Yahoo Finance actually returned data
    if stock_data.empty or bond_data.empty:
        st.error("Could not find data for the Stock or Bond symbol. Please check the tickers.")
        st.stop()

    # 2. BINANCE API FOR CRYPTO
    url = f"https://api.binance.com/api/v3/klines?symbol={crypto_symbol.upper()}&interval=1d&limit=365"
    res = requests.get(url)
    crypto_json = res.json()

    # Guard rail: Check if Binance returned a valid list of data
    if isinstance(crypto_json, list) and len(crypto_json) > 0:
        crypto_df = pd.DataFrame(crypto_json, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Vol",
            "CT", "QV", "NT", "TB", "TQ", "I"
        ])
        crypto_df["Close"] = crypto_df["Close"].astype(float)
    else:
        st.error(f"Crypto symbol '{crypto_symbol}' not found on Binance.")
        st.stop()

except Exception as e:
    st.error("A connection error occurred while fetching live data.")
    st.exception(e)
    st.stop()


    # ---------------- PREDICTION FUNCTION ----------------
    # Uses Linear Regression for trend and calculates SMA for technical context
    def get_model_predictions(series):
    # Check if we have enough data (at least 2 points to draw a line)
        if series.empty or len(series) < 2:
            return np.array([0] * 30), series # Return dummy zeros if no data
        
        series = series.ffill().dropna()
        y = series.values
        X = np.array(range(len(y))).reshape(-1, 1)
    
        model = LinearRegression().fit(X, y)
    
        future_X = np.array(range(len(y), len(y) + 30)).reshape(-1, 1)
        future_preds = model.predict(future_X)
    
        sma = series.rolling(window=50).mean()
        return future_preds, sma

    s_future, s_sma = get_model_predictions(stock_data['Close'])
    c_future, c_sma = get_model_predictions(crypto_df['Close'])
    b_future, b_sma = get_model_predictions(bond_data['Close'])
    
    # ---------------- RISK CALCULATION ----------------
    s_vol = stock_data['Close'].pct_change().dropna().std() * np.sqrt(252)
    c_vol = crypto_df['Close'].pct_change().std() * np.sqrt(365)
    b_vol = bond_data['Close'].pct_change().std() * np.sqrt(252)

    def get_risk_label(vol):
        if vol < 0.15: return "Low Risk"
        elif vol <= 0.35: return "Medium Risk"
        else: return "High Risk"
    def risk_badge(risk):
        if risk == "High Risk":
            color = "#ff4b4b"   # red
        elif risk == "Medium Risk":
            color = "#9e9e9e"   # neutral
        else:
            color = "#28a745"   # green
    
        return f"""
    <div style="
        background-color:{color};
        padding:4px 10px;
        border-radius:10px;
        colour: white;
        display:inline-block;
        font-size:12px;
        font-weight:600;">
        {risk}
    </div>
    """
    
    # ---------------- LIVE METRICS ----------------
    
    col1, col2, col3 = st.columns(3)

    stock_risk = get_risk_label(s_vol)
    crypto_risk = get_risk_label(c_vol)
    bond_risk = get_risk_label(b_vol)

    with col1:
        st.metric(f"{stock_symbol}", f"${stock_data['Close'].iloc[-1]:.2f}")
        st.markdown(risk_badge(stock_risk), unsafe_allow_html=True)

    with col2:
        st.metric(f"{crypto_symbol}", f"${crypto_df['Close'].iloc[-1]:.2f}")
        st.markdown(risk_badge(crypto_risk), unsafe_allow_html=True)

    with col3:
        st.metric(f"{bond_symbol}", f"${bond_data['Close'].iloc[-1]:.2f}")
        st.markdown(risk_badge(bond_risk), unsafe_allow_html=True)
        st.divider()

    # ---------------- CHARTS ----------------
    col_a, col_b, col_c = st.columns(3)

    def create_plot(title, history, sma, future):
        fig = go.Figure()
        # Actual Price
        fig.add_trace(go.Scatter(y=history, name="Actual Price", line=dict(color="#1f77b4")))
        # SMA
        fig.add_trace(go.Scatter(y=sma, name="50-Day SMA", line=dict(color="orange", width=1)))
        
        # --- IMPROVED: Connect the prediction to the last price point ---
        last_price = history.iloc[-1]
        # Put the last price at the start of the future array
        connected_future = np.insert(future, 0, last_price)
        # Shift index so it starts at the end of history
        future_idx = list(range(len(history) - 1, len(history) + len(future)))
        
        fig.add_trace(go.Scatter(x=future_idx, y=connected_future, name="30D Prediction", line=dict(dash='dash', color="green")))
        fig.update_layout(title=title, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        return fig
    # Display the charts
    col_a.plotly_chart(create_plot(f"{stock_symbol} Trend", stock_data['Close'], s_sma, s_future), use_container_width=True)
    col_b.plotly_chart(create_plot(f"{crypto_symbol} Trend", crypto_df['Close'], c_sma, c_future), use_container_width=True)
    col_c.plotly_chart(create_plot(f"{bond_symbol} Trend", bond_data['Close'], b_sma, b_future), use_container_width=True)

    # ---------------- RISK COMPARISON ----------------
    st.divider()
    st.subheader("📊 Portfolio Risk Assessment")

    vols = {"Stocks": s_vol, "Crypto": c_vol, "Bonds": b_vol}
    highest = max(vols, key=vols.get)
    lowest = min(vols, key=vols.get)

    st.error(f"⚠️ Highest Volatility: {highest} ({vols[highest]:.2%})")
    st.success(f"✅ Most Stable Asset: {lowest} ({vols[lowest]:.2%})")

    # ---------------- INVESTMENT GROWTH ----------------
    st.divider()
    st.subheader("💰 Wealth Projection")


    def get_annual_return(series, days=252):
        returns = series.pct_change().dropna()
        return returns.mean() * days

    months = years * 12
    inflation_rate = 0.05

    stock_return = get_annual_return(stock_data['Close'], 252)
    bond_return = get_annual_return(bond_data['Close'], 252)
    crypto_return = get_annual_return(crypto_df['Close'], 365) # Crypto is 24/7

# Limit extreme values
    stock_return = max(min(stock_return,0.8),-0.5)
    crypto_return = max(min(crypto_return,0.8),-0.5)
    bond_return = max(min(bond_return,0.8),-0.5)

    col1, col2, col3 = st.columns(3)

    def calculate_projection(rate):
        monthly_rate = rate / 12
    
        fv_lump = savings_usd * (1 + rate) ** years
    
        if monthly_rate != 0:
            fv_sip = monthly_invest_usd * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        else:
            fv_sip = monthly_invest_usd * months

        total = (fv_lump + fv_sip) * rates[currency]
        inflation_adj = total / ((1 + inflation_rate) ** years)

        return fv_lump, fv_sip, total, inflation_adj


# STOCKS
    with col1:
        fv_lump, fv_sip, total, inflation_adj = calculate_projection(stock_return)
    
        st.markdown("### 📈 Stocks")
        st.write(f"Estimated Annual Return: {stock_return:.2%}")
        st.success(f"Total Wealth Projection: {total:,.2f} {currency}")
        st.write(f"Lump Sum Growth: {(fv_lump * rates[currency]):,.2f}")
        st.write(f"SIP Growth: {(fv_sip * rates[currency]):,.2f}")
        st.warning(f"Inflation Adjusted: {inflation_adj:,.2f} {currency}")


# CRYPTO
    with col2:
        fv_lump, fv_sip, total, inflation_adj = calculate_projection(crypto_return)

        st.markdown("### ₿ Crypto")
        st.write(f"Estimated Annual Return: {crypto_return:.2%}")
        st.success(f"Total Wealth Projection: {total:,.2f} {currency}")
        st.write(f"Lump Sum Growth: {(fv_lump * rates[currency]):,.2f}")
        st.write(f"SIP Growth: {(fv_sip * rates[currency]):,.2f}")
        st.warning(f"Inflation Adjusted: {inflation_adj:,.2f} {currency}")


# BONDS
    with col3:
        fv_lump, fv_sip, total, inflation_adj = calculate_projection(bond_return)

        st.markdown("### 🏦 Bonds")
        st.write(f"Estimated Annual Return: {bond_return:.2%}")
        st.success(f"Total Wealth Projection: {total:,.2f} {currency}")
        st.write(f"Lump Sum Growth: {(fv_lump * rates[currency]):,.2f}")
        st.write(f"SIP Growth: {(fv_sip * rates[currency]):,.2f}")
        st.warning(f"Inflation Adjusted: {inflation_adj:,.2f} {currency}")
        

except Exception as e:
    st.error("Error fetching data.")
    st.exception(e)  

    
st.info(f"Disclaimer: Predictions are based on historical trends (Linear Regression) and do not guarantee future performance.")
    
    
