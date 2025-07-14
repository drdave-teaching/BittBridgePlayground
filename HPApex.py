import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import queue
import sys

# --- Insert all the code from the prompt here (except the __main__ block) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, xi, rho, r, T):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r
        self.T = T

    def simulate_paths(self, n_paths=1000, n_steps=252):
        dt = self.T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(n_paths)
            v[:, i + 1] = np.maximum(
                v[:, i] + self.kappa * (self.theta - v[:, i]) * dt +
                self.xi * np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * Z2,
                0
            )
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * v[:, i]) * dt +
                np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * Z1
            )
        return S, v

def get_stock_data(symbol, period="3y"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    data['Returns'] = data['Close'].pct_change().dropna()
    historical_vol = data['Returns'].std() * np.sqrt(252)
    return data, historical_vol

def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(period="5d")
        risk_free_rate = tnx_data['Close'].iloc[-1] / 100
        return risk_free_rate
    except:
        return 0.03

def calibrate_heston_simple(stock_data, historical_vol):
    returns = stock_data['Returns'].dropna()
    v0 = historical_vol**2
    theta = historical_vol**2
    kappa = 2.0
    xi = 0.3
    rho = -0.7
    return v0, kappa, theta, xi, rho

def analyze_stock_with_heston(symbol="^SPX", forecast_months=6):
    stock_data, historical_vol = get_stock_data(symbol)
    current_price = stock_data['Close'].iloc[-1]
    risk_free_rate = get_risk_free_rate()
    v0, kappa, theta, xi, rho = calibrate_heston_simple(stock_data, historical_vol)
    T = forecast_months / 12
    heston = HestonModel(current_price, v0, kappa, theta, xi, rho, risk_free_rate, T)
    S_paths, v_paths = heston.simulate_paths(n_paths=1000, n_steps=int(252 * T))
    return stock_data, S_paths, v_paths, heston

def create_visualizations(symbol, stock_data, S_paths, v_paths, heston):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Heston Model Analysis for {symbol}', fontsize=16, fontweight='bold')
    time_steps = np.linspace(0, heston.T, S_paths.shape[1])
    for i in range(min(50, S_paths.shape[0])):
        ax1.plot(time_steps, S_paths[i], alpha=0.3, color='lightblue', linewidth=0.5)
    percentiles = np.percentile(S_paths, [5, 25, 50, 75, 95], axis=0)
    ax1.plot(time_steps, percentiles[2], 'r-', linewidth=2, label='Median Path')
    ax1.fill_between(time_steps, percentiles[0], percentiles[4], alpha=0.2, color='red', label='90% Confidence')
    ax1.fill_between(time_steps, percentiles[1], percentiles[3], alpha=0.3, color='red', label='50% Confidence')
    ax1.axhline(y=heston.S0, color='black', linestyle='--', label='Current Price')
    ax1.set_title('Price Path Simulations')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    vol_paths = np.sqrt(v_paths)
    vol_percentiles = np.percentile(vol_paths, [5, 25, 50, 75, 95], axis=0)
    for i in range(min(20, vol_paths.shape[0])):
        ax2.plot(time_steps, vol_paths[i], alpha=0.3, color='green', linewidth=0.5)
    ax2.plot(time_steps, vol_percentiles[2], 'g-', linewidth=2, label='Median Volatility')
    ax2.fill_between(time_steps, vol_percentiles[0], vol_percentiles[4], alpha=0.2, color='green')
    ax2.axhline(y=np.sqrt(heston.theta), color='orange', linestyle='--', label='Long-term Vol')
    ax2.set_title('Volatility Evolution')
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    final_prices = S_paths[:, -1]
    ax3.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=heston.S0, color='red', linestyle='--', linewidth=2, label='Current Price')
    ax3.axvline(x=np.median(final_prices), color='green', linestyle='-', linewidth=2, label='Expected Price')
    ax3.set_title(f'Price Distribution in {heston.T:.1f} Years')
    ax3.set_xlabel('Stock Price ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    recent_data = stock_data.tail(252)
    ax4.plot(recent_data.index, recent_data['Close'], 'b-', linewidth=2, label='Historical Price')
    ax4.set_title('Historical Price Performance')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Stock Price ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

class DeepDifferentialHeston(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[128, 256, 128, 64], output_dim=1):
        super(DeepDifferentialHeston, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    def forward(self, x):
        return self.network(x)

class VolatilityForecastingNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=3, output_size=1):
        super(VolatilityForecastingNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8,
                                             dropout=0.1, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = self.fc_layers(attn_out[:, -1, :])
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EnhancedHestonML:
    def __init__(self, S0, r, T):
        self.S0 = S0
        self.r = r
        self.T = T
        self.param_network = DeepDifferentialHeston().to(device)
        self.vol_network = VolatilityForecastingNetwork().to(device)
        self.param_scaler = StandardScaler()
        self.vol_scaler = StandardScaler()
        self.training_history = {'param_loss': [], 'vol_loss': []}
    def generate_training_data(self, n_samples=50000):
        kappa_range = (0.1, 5.0)
        theta_range = (0.01, 0.5)
        xi_range = (0.1, 1.0)
        rho_range = (-0.9, -0.1)
        v0_range = (0.01, 0.5)
        np.random.seed(42)
        params = np.random.uniform(0, 1, (n_samples, 5))
        params[:, 0] = params[:, 0] * (kappa_range[1] - kappa_range[0]) + kappa_range[0]
        params[:, 1] = params[:, 1] * (theta_range[1] - theta_range[0]) + theta_range[0]
        params[:, 2] = params[:, 2] * (xi_range[1] - xi_range[0]) + xi_range[0]
        params[:, 3] = params[:, 3] * (rho_range[1] - rho_range[0]) + rho_range[0]
        params[:, 4] = params[:, 4] * (v0_range[1] - v0_range[0]) + v0_range[0]
        prices = []
        volatilities = []
        for i, param_set in enumerate(params):
            kappa, theta, xi, rho, v0 = param_set
            vol_approx = np.sqrt(v0 + theta) / 2
            price_approx = self.S0 * np.exp((self.r - 0.5 * vol_approx**2) * self.T +
                                          vol_approx * np.sqrt(self.T) * np.random.normal())
            prices.append(price_approx)
            volatilities.append(vol_approx)
        return np.array(params), np.array(prices), np.array(volatilities)
    def prepare_volatility_sequences(self, stock_data, sequence_length=20):
        returns = stock_data['Returns'].dropna()
        features = []
        for i in range(len(returns)):
            if i >= sequence_length:
                window_returns = returns.iloc[i-sequence_length:i]
                vol = window_returns.std() * np.sqrt(252)
                skew = window_returns.skew()
                kurt = window_returns.kurtosis()
                momentum = window_returns.mean()
                sma_5 = window_returns.tail(5).mean()
                sma_20 = window_returns.mean()
                vol_change = vol - (returns.iloc[i-sequence_length-1:i-1].std() * np.sqrt(252))
                extreme_moves = (np.abs(window_returns) > 2 * window_returns.std()).sum()
                feature_vector = [vol, skew, kurt, momentum, sma_5, sma_20,
                                vol_change, extreme_moves, len(window_returns), vol**2]
                features.append(feature_vector)
        return np.array(features[:-1]), np.array([f[0] for f in features[1:]])
    def train_networks(self, stock_data, epochs_param=100, epochs_vol=200):
        params, prices, _ = self.generate_training_data()
        params_norm = self.param_scaler.fit_transform(params)
        prices_norm = (prices - prices.mean()) / prices.std()
        X_param = torch.FloatTensor(params_norm).to(device)
        y_param = torch.FloatTensor(prices_norm.reshape(-1, 1)).to(device)
        X_train, X_val, y_train, y_val = train_test_split(
            X_param, y_param, test_size=0.2, random_state=42
        )
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512)
        optimizer_param = optim.AdamW(self.param_network.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler_param = optim.lr_scheduler.ReduceLROnPlateau(optimizer_param, patience=10)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        for epoch in range(epochs_param):
            self.param_network.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer_param.zero_grad()
                outputs = self.param_network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), 1.0)
                optimizer_param.step()
                train_loss += loss.item()
            self.param_network.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.param_network(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            self.training_history['param_loss'].append(avg_val_loss)
            scheduler_param.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.param_network.state_dict(), 'best_param_network.pth')
        X_vol, y_vol = self.prepare_volatility_sequences(stock_data)
        if len(X_vol) > 0:
            X_vol_norm = self.vol_scaler.fit_transform(X_vol)
            X_vol_tensor = torch.FloatTensor(X_vol_norm).unsqueeze(1).to(device)
            y_vol_tensor = torch.FloatTensor(y_vol.reshape(-1, 1)).to(device)
            X_vol_train, X_vol_val, y_vol_train, y_vol_val = train_test_split(
                X_vol_tensor, y_vol_tensor, test_size=0.2, random_state=42
            )
            vol_train_dataset = TensorDataset(X_vol_train, y_vol_train)
            vol_val_dataset = TensorDataset(X_vol_val, y_vol_val)
            vol_train_loader = DataLoader(vol_train_dataset, batch_size=64, shuffle=True)
            vol_val_loader = DataLoader(vol_val_dataset, batch_size=64)
            optimizer_vol = optim.AdamW(self.vol_network.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler_vol = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vol, patience=15)
            best_vol_loss = float('inf')
            for epoch in range(epochs_vol):
                self.vol_network.train()
                train_loss = 0
                for batch_X, batch_y in vol_train_loader:
                    optimizer_vol.zero_grad()
                    outputs = self.vol_network(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vol_network.parameters(), 1.0)
                    optimizer_vol.step()
                    train_loss += loss.item()
                self.vol_network.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in vol_val_loader:
                        outputs = self.vol_network(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                avg_train_loss = train_loss / len(vol_train_loader)
                avg_val_loss = val_loss / len(vol_val_loader)
                self.training_history['vol_loss'].append(avg_val_loss)
                scheduler_vol.step(avg_val_loss)
                if avg_val_loss < best_vol_loss:
                    best_vol_loss = avg_val_loss
                    torch.save(self.vol_network.state_dict(), 'best_vol_network.pth')
    def enhanced_forecast(self, stock_data, forecast_months=6, n_simulations=5000):
        try:
            self.param_network.load_state_dict(torch.load('best_param_network.pth'))
            self.vol_network.load_state_dict(torch.load('best_vol_network.pth'))
        except:
            pass
        self.param_network.eval()
        self.vol_network.eval()
        current_price = stock_data['Close'].iloc[-1]
        recent_returns = stock_data['Returns'].dropna().tail(20)
        current_vol = recent_returns.std() * np.sqrt(252)
        X_vol_current, _ = self.prepare_volatility_sequences(stock_data.tail(50))
        if len(X_vol_current) > 0:
            X_vol_norm = self.vol_scaler.transform(X_vol_current[-1:])
            X_vol_tensor = torch.FloatTensor(X_vol_norm).unsqueeze(1).to(device)
            with torch.no_grad():
                predicted_vol = self.vol_network(X_vol_tensor).cpu().numpy()[0, 0]
        else:
            predicted_vol = current_vol
        market_features = np.array([[
            2.0,
            predicted_vol**2,
            0.3,
            -0.7,
            current_vol**2
        ]])
        T = forecast_months / 12
        dt = T / 252
        n_steps = int(252 * T)
        S_paths = np.zeros((n_simulations, n_steps + 1))
        v_paths = np.zeros((n_simulations, n_steps + 1))
        S_paths[:, 0] = current_price
        v_paths[:, 0] = current_vol**2
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_simulations)
            Z2 = -0.7 * Z1 + np.sqrt(1 - 0.7**2) * np.random.standard_normal(n_simulations)
            vol_adjustment = 1 + 0.1 * np.sin(2 * np.pi * i / 252)
            kappa_t = 2.0 * vol_adjustment
            theta_t = predicted_vol**2
            xi_t = 0.3
            v_paths[:, i + 1] = np.maximum(
                v_paths[:, i] + kappa_t * (theta_t - v_paths[:, i]) * dt +
                xi_t * np.sqrt(np.maximum(v_paths[:, i], 0)) * np.sqrt(dt) * Z2,
                0.001
            )
            S_paths[:, i + 1] = S_paths[:, i] * np.exp(
                (self.r - 0.5 * v_paths[:, i]) * dt +
                np.sqrt(np.maximum(v_paths[:, i], 0)) * np.sqrt(dt) * Z1
            )
        return S_paths, v_paths, predicted_vol
    def create_enhanced_visualizations(self, symbol, stock_data, S_paths, v_paths, predicted_vol):
        # Defensive checks
        if S_paths is None or v_paths is None or len(S_paths) == 0 or np.isnan(S_paths).all():
            print("[DEBUG] S_paths is empty or all NaN, skipping plot.")
            return
        if np.isnan(S_paths).any():
            print(f"[DEBUG] S_paths contains NaN values. Shape: {S_paths.shape}")
            print("[DEBUG] Cleaning NaNs in S_paths for plotting.")
            S_paths = np.nan_to_num(S_paths, nan=np.nanmean(S_paths))
        if np.isnan(v_paths).any():
            print(f"[DEBUG] v_paths contains NaN values. Shape: {v_paths.shape}")
            print("[DEBUG] Cleaning NaNs in v_paths for plotting.")
            v_paths = np.nan_to_num(v_paths, nan=np.nanmean(v_paths))
        print(f"[DEBUG] S_paths stats: min={np.nanmin(S_paths)}, max={np.nanmax(S_paths)}, mean={np.nanmean(S_paths)}")
        print(f"[DEBUG] v_paths stats: min={np.nanmin(v_paths)}, max={np.nanmax(v_paths)}, mean={np.nanmean(v_paths)}")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Enhanced ML Heston Analysis for {symbol}', fontsize=16, fontweight='bold')
        time_steps = np.linspace(0, self.T, S_paths.shape[1])
        percentiles = np.percentile(S_paths, [5, 10, 25, 50, 75, 90, 95], axis=0)
        for i in range(min(30, S_paths.shape[0])):
            ax1.plot(time_steps, S_paths[i], alpha=0.2, color='lightblue', linewidth=0.5)
        ax1.plot(time_steps, percentiles[3], 'r-', linewidth=3, label='ML-Enhanced Median')
        ax1.fill_between(time_steps, percentiles[0], percentiles[6], alpha=0.15, color='red', label='90% Confidence')
        ax1.fill_between(time_steps, percentiles[1], percentiles[5], alpha=0.25, color='orange', label='80% Confidence')
        ax1.fill_between(time_steps, percentiles[2], percentiles[4], alpha=0.35, color='yellow', label='50% Confidence')
        ax1.axhline(y=self.S0, color='black', linestyle='--', linewidth=2, label='Current Price')
        ax1.set_title('ML-Enhanced Price Forecasts')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        vol_paths = np.sqrt(v_paths)
        vol_percentiles = np.percentile(vol_paths, [5, 25, 50, 75, 95], axis=0)
        for i in range(min(20, vol_paths.shape[0])):
            ax2.plot(time_steps, vol_paths[i], alpha=0.3, color='green', linewidth=0.5)
        ax2.plot(time_steps, vol_percentiles[2], 'g-', linewidth=3, label='Median Volatility')
        ax2.fill_between(time_steps, vol_percentiles[0], vol_percentiles[4], alpha=0.3, color='green')
        ax2.axhline(y=predicted_vol, color='purple', linestyle='--', linewidth=2, label='ML Predicted Vol')
        ax2.set_title('ML-Enhanced Volatility Evolution')
        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if self.training_history['param_loss']:
            ax3.plot(self.training_history['param_loss'], 'b-', label='Parameter Network')
        if self.training_history['vol_loss']:
            ax3.plot(self.training_history['vol_loss'], 'r-', label='Volatility Network')
        ax3.set_title('ML Training Progress')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        final_prices = S_paths[:, -1]
        returns = (final_prices - self.S0) / self.S0
        ax4.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax4.axvline(x=np.median(returns), color='green', linestyle='-', linewidth=2, label='Expected Return')
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        ax4.axvline(x=var_95, color='orange', linestyle=':', linewidth=2, label=f'VaR 95%: {var_95:.1%}')
        ax4.set_title('ML-Enhanced Return Distribution')
        ax4.set_xlabel('Return')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def run_enhanced_analysis(symbol="^SPX", forecast_months=6):
    print(f"[DEBUG] Entered run_enhanced_analysis for {symbol}, forecast_months={forecast_months}")
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period="3y")
    if stock_data.empty:
        print("[DEBUG] Stock data is empty! Check ticker or network.")
        return None, None, None, None
    stock_data['Returns'] = stock_data['Close'].pct_change()
    try:
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(period="5d")
        risk_free_rate = tnx_data['Close'].iloc[-1] / 100
    except:
        risk_free_rate = 0.03
    current_price = stock_data['Close'].iloc[-1]
    enhanced_heston = EnhancedHestonML(current_price, risk_free_rate, forecast_months/12)
    print(f"[DEBUG] About to train networks...")
    enhanced_heston.train_networks(stock_data, epochs_param=2, epochs_vol=2)  # <<-- reduce for test
    print(f"[DEBUG] About to forecast...")
    S_paths, v_paths, predicted_vol = enhanced_heston.enhanced_forecast(
        stock_data, forecast_months, n_simulations=10  # <<-- reduce for test
    )
    print(f"[DEBUG] Exiting run_enhanced_analysis.")
    return enhanced_heston, S_paths, v_paths, predicted_vol

def redirect_print_to_widget(widget):
    class RedirectText(object):
        def __init__(self, text_ctrl):
            self.output = text_ctrl
        def write(self, string):
            self.output.after(0, self.output.insert, tk.END, string)
            self.output.after(0, self.output.see, tk.END)
        def flush(self):
            pass
    sys.stdout = RedirectText(widget)
    sys.stderr = RedirectText(widget)

# --- Tkinter Dashboard ---

def heston_summary(symbol, current_price, historical_vol, risk_free_rate, v0, kappa, theta, xi, rho, percentiles, expected_return, downside_risk, upside_potential, prob_profit, avg_expected_return, outlook, future_vol):
    summary = f'''
Current Price: ${current_price:,.2f}
📈 Historical Volatility: {historical_vol*100:.1f}%
🏦 Risk-free Rate: {risk_free_rate*100:.1f}%

🎯 Heston Model Parameters:
   Initial Variance (v0): {v0:.4f}
   Mean Reversion Speed (κ): {kappa:.2f}
   Long-term Variance (θ): {theta:.4f}
   Vol of Vol (ξ): {xi:.2f}
   Correlation (ρ): {rho:.2f}

🚀 Simulating 12-month price paths...

📊 Price Forecast Results (in 12 months):
   5th Percentile:  ${percentiles[0]:,.2f}
   25th Percentile: ${percentiles[1]:,.2f}
   Median (50th):   ${percentiles[2]:,.2f}
   75th Percentile: ${percentiles[3]:,.2f}
   95th Percentile: ${percentiles[4]:,.2f}

💰 Expected Return: {expected_return*100:.1f}%
📉 Downside Risk (5th percentile): {downside_risk*100:.1f}%
📈 Upside Potential (95th percentile): {upside_potential*100:.1f}%

🎯 INVESTMENT INSIGHTS FOR SPOT INVESTORS
============================================================
📊 Probability of Profit: {prob_profit*100:.1f}%
💰 Average Expected Return: {avg_expected_return*100:.1f}%
⚖️  {outlook}

📈 Volatility Insights:
   Expected Future Volatility: {future_vol*100:.1f}%
   📊 Volatility expected to remain STABLE

🔄 To analyze a different stock, change the symbol variable
📅 To change forecast period, modify forecast_months parameter
'''
    return summary

def run_basic_heston(symbol, months, output_text, progress_callback=None, done_callback=None):
    try:
        stock_data, S_paths, v_paths, heston_model = analyze_stock_with_heston(symbol, forecast_months=months)
        # Calculate summary stats
        current_price = heston_model.S0
        historical_vol = stock_data['Returns'].std() * np.sqrt(252)
        risk_free_rate = get_risk_free_rate()
        v0 = heston_model.v0
        kappa = heston_model.kappa
        theta = heston_model.theta
        xi = heston_model.xi
        rho = heston_model.rho
        percentiles = np.percentile(S_paths[:, -1], [5, 25, 50, 75, 95])
        expected_return = (np.median(S_paths[:, -1]) - current_price) / current_price
        downside_risk = (percentiles[0] - current_price) / current_price
        upside_potential = (percentiles[4] - current_price) / current_price
        prob_profit = np.mean(S_paths[:, -1] > current_price)
        avg_expected_return = np.mean((S_paths[:, -1] - current_price) / current_price)
        outlook = "NEUTRAL OUTLOOK: Balanced risk-reward, consider market conditions"
        future_vol = np.std(S_paths[:, -1]) / np.mean(S_paths[:, -1])
        summary = heston_summary(symbol, current_price, historical_vol, risk_free_rate, v0, kappa, theta, xi, rho, percentiles, expected_return, downside_risk, upside_potential, prob_profit, avg_expected_return, outlook, future_vol)
        def gui_update():
            output_text.insert(tk.END, f"Basic Heston Model run for {symbol} ({months} months)\n")
            output_text.insert(tk.END, summary + '\n')
            create_visualizations(symbol, stock_data, S_paths, v_paths, heston_model)
            if done_callback:
                done_callback()
        output_text.after(0, gui_update)
    except Exception as e:
        output_text.after(0, lambda: output_text.insert(tk.END, f"Error: {e}\n"))
        if done_callback:
            output_text.after(0, done_callback)

def enhanced_heston_summary(symbol, current_price, historical_vol, risk_free_rate, percentiles, expected_return, downside_risk, upside_potential, prob_profit, avg_expected_return, outlook, future_vol):
    summary = f'''
Current Price: ${current_price:,.2f}
📈 Historical Volatility: {historical_vol*100:.1f}%
🏦 Risk-free Rate: {risk_free_rate*100:.1f}%

🚀 ML-Enhanced Heston Simulation Results:

📊 Price Forecast Results (in {symbol}):
   5th Percentile:  ${percentiles[0]:,.2f}
   25th Percentile: ${percentiles[1]:,.2f}
   Median (50th):   ${percentiles[2]:,.2f}
   75th Percentile: ${percentiles[3]:,.2f}
   95th Percentile: ${percentiles[4]:,.2f}

💰 Expected Return: {expected_return*100:.1f}%
📉 Downside Risk (5th percentile): {downside_risk*100:.1f}%
📈 Upside Potential (95th percentile): {upside_potential*100:.1f}%

🎯 INVESTMENT INSIGHTS FOR SPOT INVESTORS
============================================================
📊 Probability of Profit: {prob_profit*100:.1f}%
💰 Average Expected Return: {avg_expected_return*100:.1f}%
⚖️  {outlook}

📈 Volatility Insights:
   Expected Future Volatility: {future_vol*100:.1f}%
   📊 Volatility expected to remain STABLE
'''
    return summary

def run_enhanced_heston(symbol, months, output_text, progress_callback=None, done_callback=None):
    try:
        print(f"[DEBUG] Thread started for enhanced heston with symbol={symbol}, months={months}")
        def gui_update_start():
            output_text.insert(tk.END, f"Running Enhanced ML Heston Model for {symbol} ({months} months)...\n")
            output_text.see(tk.END)
        output_text.after(0, gui_update_start)
        print(f"[DEBUG] Calling run_enhanced_analysis...")
        enhanced_heston, S_paths, v_paths, predicted_vol = run_enhanced_analysis(symbol, forecast_months=months)
        print(f"[DEBUG] run_enhanced_analysis returned.")
        # Calculate summary stats for enhanced model
        if enhanced_heston is not None and S_paths is not None and v_paths is not None:
            current_price = enhanced_heston.S0
            # Use historical_vol from last 252 returns
            historical_vol = None
            try:
                stock_data, _ = get_stock_data(symbol)
                historical_vol = stock_data['Returns'].std() * np.sqrt(252)
            except:
                historical_vol = 0.0
            risk_free_rate = get_risk_free_rate()
            # Filter out simulation paths that are all NaN or contain any NaN
            valid_mask = ~np.isnan(S_paths[:, -1])
            valid_S = S_paths[valid_mask, -1]
            if valid_S.size == 0:
                # All paths are NaN, fallback to zeros or skip summary
                percentiles = [0, 0, 0, 0, 0]
                expected_return = downside_risk = upside_potential = prob_profit = avg_expected_return = future_vol = float('nan')
            else:
                percentiles = np.percentile(valid_S, [5, 25, 50, 75, 95])
                expected_return = (np.median(valid_S) - current_price) / current_price
                downside_risk = (percentiles[0] - current_price) / current_price
                upside_potential = (percentiles[4] - current_price) / current_price
                prob_profit = np.mean(valid_S > current_price)
                avg_expected_return = np.mean((valid_S - current_price) / current_price)
                future_vol = np.std(valid_S) / np.mean(valid_S)
            outlook = "NEUTRAL OUTLOOK: Balanced risk-reward, consider market conditions"
            summary = enhanced_heston_summary(symbol, current_price, historical_vol, risk_free_rate, percentiles, expected_return, downside_risk, upside_potential, prob_profit, avg_expected_return, outlook, future_vol)
        else:
            summary = "[DEBUG] Enhanced Heston simulation failed or returned no data."
        def gui_update_end():
            output_text.insert(tk.END, f"Enhanced ML Heston Model completed for {symbol}\n")
            output_text.insert(tk.END, summary + '\n')
            output_text.see(tk.END)
            try:
                print(f"[DEBUG] Calling create_enhanced_visualizations...")
                if enhanced_heston and S_paths is not None and v_paths is not None:
                    enhanced_heston.create_enhanced_visualizations(
                        symbol, enhanced_heston.S0, S_paths, v_paths, predicted_vol
                    )
                print(f"[DEBUG] Plotting complete.")
            except Exception as plot_e:
                output_text.insert(tk.END, f"Error during plotting: {plot_e}\n")
                output_text.see(tk.END)
            if done_callback:
                done_callback()
        output_text.after(0, gui_update_end)
    except Exception as e:
        print(f"[DEBUG] Exception in run_enhanced_heston: {e}")
        output_text.after(0, lambda: output_text.insert(tk.END, f"Error: {e}\n"))
        output_text.after(0, lambda: output_text.see(tk.END))
        if done_callback:
            output_text.after(0, done_callback)

def start_basic_heston_thread(symbol, months, output_text, progressbar, btns):
    output_text.delete(1.0, tk.END)
    progressbar['value'] = 0
    for btn in btns:
        btn['state'] = 'disabled'
    def done():
        progressbar['value'] = 100
        for btn in btns:
            btn['state'] = 'normal'
    t = threading.Thread(target=run_basic_heston, args=(symbol, months, output_text, None, done))
    t.start()

def start_enhanced_heston_thread(symbol, months, output_text, progressbar, btns):
    output_text.delete(1.0, tk.END)
    progressbar['value'] = 0
    for btn in btns:
        btn['state'] = 'disabled'
    def done():
        progressbar['value'] = 100
        for btn in btns:
            btn['state'] = 'normal'
    t = threading.Thread(target=run_enhanced_heston, args=(symbol, months, output_text, None, done))
    t.start()

def main_dashboard():
    root = tk.Tk()
    root.title("Stock Heston Model & ML Forecast Dashboard (Revamped)")
    frm = ttk.Frame(root, padding=20)
    frm.grid(row=0, column=0, sticky="nsew")
    ttk.Label(frm, text="Stock Ticker (e.g. AAPL, MSFT, ^SPX):").grid(row=0, column=0, sticky="w")
    symbol_entry = ttk.Entry(frm, width=15)
    symbol_entry.grid(row=0, column=1, sticky="w")
    symbol_entry.insert(0, "AAPL")
    ttk.Label(frm, text="Forecast Months:").grid(row=1, column=0, sticky="w")
    months_entry = ttk.Entry(frm, width=5)
    months_entry.grid(row=1, column=1, sticky="w")
    months_entry.insert(0, "6")
    progressbar = ttk.Progressbar(frm, orient="horizontal", length=300, mode="determinate", maximum=100)
    progressbar.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
    output_text = ScrolledText(frm, width=60, height=10, wrap="word")
    output_text.grid(row=4, column=0, columnspan=3, pady=10)
    redirect_print_to_widget(output_text)
    def on_basic():
        symbol = symbol_entry.get().strip().upper()
        try:
            months = int(months_entry.get())
        except:
            messagebox.showerror("Input Error", "Forecast months must be an integer.")
            return
        start_basic_heston_thread(symbol, months, output_text, progressbar, [btn_basic, btn_enhanced])
    def on_enhanced():
        symbol = symbol_entry.get().strip().upper()
        try:
            months = int(months_entry.get())
        except:
            messagebox.showerror("Input Error", "Forecast months must be an integer.")
            return
        start_enhanced_heston_thread(symbol, months, output_text, progressbar, [btn_basic, btn_enhanced])
    btn_basic = ttk.Button(frm, text="Run Basic Heston Model", command=on_basic)
    btn_basic.grid(row=3, column=0, pady=5, sticky="ew")
    btn_enhanced = ttk.Button(frm, text="Run Enhanced ML Heston Model", command=on_enhanced)
    btn_enhanced.grid(row=3, column=1, pady=5, sticky="ew")
    ttk.Label(frm, text="Output:").grid(row=5, column=0, sticky="w")
    root.mainloop()

if __name__ == "__main__":
    main_dashboard()
