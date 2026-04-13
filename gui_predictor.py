"""
Stock Direction Predictor — CNN Model GUI
Run from project root: python gui_predictor.py
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import yaml
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ══════════════════════════════════════════════════════════════
# Colors & Style
# ══════════════════════════════════════════════════════════════
BG_DARK    = "#0f0f1a"
BG_CARD    = "#1a1a2e"
BG_INPUT   = "#16213e"
FG_TEXT    = "#e0e0e0"
FG_LABEL   = "#8892b0"
ACCENT     = "#64ffda"
GREEN      = "#00e676"
RED        = "#ff1744"
BLUE       = "#448aff"
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_BIG   = ("Segoe UI", 36, "bold")
FONT_MED   = ("Segoe UI", 14)


class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Stock Direction Predictor — CNN Model")
        self.root.geometry("720x750")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # Load resources
        self.load_resources()

        # Build UI
        self.build_ui()

    # ──────────────────────────────────────────────────────────
    # Load config, data, model, and prepare scalers
    # ──────────────────────────────────────────────────────────
    def load_resources(self):
        # Config
        with open("configs/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # Processed data
        self.data = pd.read_csv("data/processed/processed_stock_data.csv")
        self.data["Date"] = pd.to_datetime(self.data["Date"])

        # CNN model
        model_path = "saved_models/deep_learning/qusay_models/cnn_classifier.keras"
        self.cnn_model = keras.models.load_model(model_path)

        # Feature columns
        self.feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "SMA_50", "RSI_14", "MACD", "MACD_Signal",
            "Bollinger_Upper", "Bollinger_Lower",
        ]
        self.window_size = self.config["data"]["window_size"]
        self.train_end = self.config["data"]["train_end_date"]

        # Fit one scaler per ticker (on train data only)
        self.scalers = {}
        for ticker in sorted(self.data["Ticker"].unique()):
            tdf = self.data[self.data["Ticker"] == ticker].sort_values("Date")
            train_mask = tdf["Date"] <= self.train_end
            scaler = MinMaxScaler()
            scaler.fit(tdf.loc[train_mask, self.feature_cols])
            self.scalers[ticker] = scaler

    # ──────────────────────────────────────────────────────────
    # Build the UI
    # ──────────────────────────────────────────────────────────
    def build_ui(self):
        # ── Title ──
        title_frame = tk.Frame(self.root, bg=BG_DARK)
        title_frame.pack(fill="x", padx=20, pady=(18, 5))
        tk.Label(
            title_frame, text="📊 Stock Direction Predictor",
            font=FONT_TITLE, fg=ACCENT, bg=BG_DARK
        ).pack(anchor="w")
        tk.Label(
            title_frame, text="1D-CNN Model  •  Predict tomorrow's price direction",
            font=("Segoe UI", 10), fg=FG_LABEL, bg=BG_DARK
        ).pack(anchor="w")

        # ── Input Card ──
        input_card = tk.Frame(self.root, bg=BG_CARD, highlightbackground="#2a2a4a",
                              highlightthickness=1)
        input_card.pack(fill="x", padx=20, pady=12)

        tk.Label(
            input_card, text="SELECT STOCK", font=FONT_HEAD,
            fg=FG_LABEL, bg=BG_CARD
        ).pack(anchor="w", padx=16, pady=(14, 4))

        # Ticker dropdown
        ticker_frame = tk.Frame(input_card, bg=BG_CARD)
        ticker_frame.pack(fill="x", padx=16, pady=(0, 6))

        self.ticker_var = tk.StringVar(value="AAPL")
        tickers = sorted(self.data["Ticker"].unique())

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.TCombobox",
                        fieldbackground=BG_INPUT, background=BG_INPUT,
                        foreground=FG_TEXT, selectbackground=BLUE,
                        arrowcolor=ACCENT)

        self.ticker_combo = ttk.Combobox(
            ticker_frame, textvariable=self.ticker_var,
            values=tickers, state="readonly", font=FONT_MED,
            style="Custom.TCombobox", width=12
        )
        self.ticker_combo.pack(side="left", pady=4)
        self.ticker_combo.bind("<<ComboboxSelected>>", lambda e: self.update_info())

        # Predict button
        self.predict_btn = tk.Button(
            ticker_frame, text="🔮  PREDICT", font=FONT_HEAD,
            bg=BLUE, fg="white", activebackground="#1565c0",
            activeforeground="white", relief="flat", cursor="hand2",
            padx=20, pady=6, command=self.predict
        )
        self.predict_btn.pack(side="right", pady=4)

        # Stock info label
        self.info_label = tk.Label(
            input_card, text="", font=("Segoe UI", 10),
            fg=FG_LABEL, bg=BG_CARD, justify="left"
        )
        self.info_label.pack(anchor="w", padx=16, pady=(0, 12))

        # ── Result Card ──
        self.result_card = tk.Frame(self.root, bg=BG_CARD,
                                    highlightbackground="#2a2a4a",
                                    highlightthickness=1)
        self.result_card.pack(fill="x", padx=20, pady=6)

        tk.Label(
            self.result_card, text="PREDICTION RESULT",
            font=FONT_HEAD, fg=FG_LABEL, bg=BG_CARD
        ).pack(anchor="w", padx=16, pady=(14, 0))

        # Direction label (big arrow)
        self.direction_label = tk.Label(
            self.result_card, text="—", font=FONT_BIG,
            fg=FG_LABEL, bg=BG_CARD
        )
        self.direction_label.pack(pady=(6, 0))

        # Confidence
        self.confidence_label = tk.Label(
            self.result_card, text="Select a stock and click PREDICT",
            font=FONT_MED, fg=FG_LABEL, bg=BG_CARD
        )
        self.confidence_label.pack(pady=(0, 4))

        # Confidence bar
        bar_frame = tk.Frame(self.result_card, bg=BG_CARD)
        bar_frame.pack(fill="x", padx=40, pady=(0, 14))

        self.bar_canvas = tk.Canvas(bar_frame, height=22, bg=BG_INPUT,
                                    highlightthickness=0)
        self.bar_canvas.pack(fill="x")

        # ── Data Preview Card ──
        preview_card = tk.Frame(self.root, bg=BG_CARD,
                                highlightbackground="#2a2a4a",
                                highlightthickness=1)
        preview_card.pack(fill="both", expand=True, padx=20, pady=(6, 18))

        tk.Label(
            preview_card, text="LATEST DATA (Last 5 Days of Window)",
            font=FONT_HEAD, fg=FG_LABEL, bg=BG_CARD
        ).pack(anchor="w", padx=16, pady=(14, 6))

        # Treeview for data table
        cols = ("Date", "Close", "Volume", "RSI_14", "MACD", "SMA_20")
        tree_frame = tk.Frame(preview_card, bg=BG_CARD)
        tree_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        style.configure("Custom.Treeview",
                        background=BG_INPUT, foreground=FG_TEXT,
                        fieldbackground=BG_INPUT, rowheight=26,
                        font=("Consolas", 10))
        style.configure("Custom.Treeview.Heading",
                        background=BG_CARD, foreground=ACCENT,
                        font=("Segoe UI", 10, "bold"))
        style.map("Custom.Treeview", background=[("selected", BLUE)])

        self.tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            height=5, style="Custom.Treeview"
        )
        for col in cols:
            self.tree.heading(col, text=col)
            w = 70 if col != "Date" else 100
            self.tree.column(col, width=w, anchor="center")

        self.tree.pack(fill="both", expand=True)

        # Initial info
        self.update_info()

    # ──────────────────────────────────────────────────────────
    # Update stock info when ticker changes
    # ──────────────────────────────────────────────────────────
    def update_info(self):
        ticker = self.ticker_var.get()
        tdf = self.data[self.data["Ticker"] == ticker].sort_values("Date")
        last_date = tdf["Date"].iloc[-1].strftime("%Y-%m-%d")
        last_close = tdf["Close"].iloc[-1]
        total_days = len(tdf)
        self.info_label.config(
            text=f"📅 Latest date: {last_date}   |   💰 Last close: ${last_close:.2f}   |   📊 Total: {total_days} days"
        )

        # Update data preview
        for row in self.tree.get_children():
            self.tree.delete(row)

        last_5 = tdf.tail(5)
        for _, row in last_5.iterrows():
            self.tree.insert("", "end", values=(
                row["Date"].strftime("%Y-%m-%d"),
                f"{row['Close']:.2f}",
                f"{int(row['Volume']):,}",
                f"{row['RSI_14']:.1f}",
                f"{row['MACD']:.3f}",
                f"{row['SMA_20']:.2f}",
            ))

    # ──────────────────────────────────────────────────────────
    # Run prediction
    # ──────────────────────────────────────────────────────────
    def predict(self):
        ticker = self.ticker_var.get()

        # Get ticker data
        tdf = self.data[self.data["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        if len(tdf) < self.window_size:
            self.direction_label.config(text="❌", fg=RED)
            self.confidence_label.config(text="Not enough data for this ticker")
            return

        # Scale features using the pre-fitted scaler
        scaler = self.scalers[ticker]
        scaled = scaler.transform(tdf[self.feature_cols])

        # Take the last window (30 days)
        last_window = scaled[-self.window_size:]
        X_input = last_window.reshape(1, self.window_size, len(self.feature_cols))

        # Predict
        prob = self.cnn_model.predict(X_input, verbose=0)[0][0]

        # Display result
        if prob > 0.5:
            direction = "▲ UP"
            color = GREEN
            confidence = prob
        else:
            direction = "▼ DOWN"
            color = RED
            confidence = 1 - prob

        self.direction_label.config(text=direction, fg=color)
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1%}   |   Raw probability: {prob:.4f}",
            fg=FG_TEXT
        )

        # Draw confidence bar
        self.bar_canvas.delete("all")
        canvas_w = self.bar_canvas.winfo_width()
        if canvas_w < 10:
            canvas_w = 600

        # DOWN portion (red) | UP portion (green)
        mid = int(canvas_w * (1 - prob))
        self.bar_canvas.create_rectangle(0, 0, mid, 22, fill=RED, outline="")
        self.bar_canvas.create_rectangle(mid, 0, canvas_w, 22, fill=GREEN, outline="")

        # Labels on bar
        self.bar_canvas.create_text(
            mid // 2, 11, text=f"DOWN {(1-prob):.0%}",
            fill="white", font=("Segoe UI", 9, "bold")
        )
        self.bar_canvas.create_text(
            mid + (canvas_w - mid) // 2, 11, text=f"UP {prob:.0%}",
            fill="white", font=("Segoe UI", 9, "bold")
        )

        # Update data preview
        self.update_info()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading model and data...")
    root = tk.Tk()
    app = StockPredictorApp(root)
    print("Ready! GUI is running.")
    root.mainloop()
