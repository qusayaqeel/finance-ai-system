# =============================================================================
# 📓 Reference Code for qusay_cnn.ipynb
# Copy each section (between ═══ markers) into a separate notebook cell
# =============================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Imports
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Load Config & Data
# ═══════════════════════════════════════════════════════════════════════════════

# Read project config
with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load processed data
data = pd.read_csv('../data/processed/processed_stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

print(f"📊 Dataset shape: {data.shape}")
print(f"📅 Date range: {data['Date'].min().date()} → {data['Date'].max().date()}")
print(f"🏢 Tickers: {list(data['Ticker'].unique())}")
print(f"\n📋 Columns: {list(data.columns)}")
data.head()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Data Exploration
# ═══════════════════════════════════════════════════════════════════════════════

# Check class balance (Target_Direction)
print("🎯 Target Distribution (all data):")
print(data['Target_Direction'].value_counts())
print(f"\nBalance ratio: {data['Target_Direction'].mean():.2%} UP vs {1 - data['Target_Direction'].mean():.2%} DOWN")

# Samples per ticker
print("\n📈 Samples per ticker:")
print(data.groupby('Ticker').size())


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Define Features & Config
# ═══════════════════════════════════════════════════════════════════════════════

# Features the model will use (12 features)
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal',
                'Bollinger_Upper', 'Bollinger_Lower']

TARGET_COL = 'Target_Direction'

# Config values
WINDOW_SIZE = config['data']['window_size']  # 30 days
TRAIN_END = config['data']['train_end_date']
VAL_END = config['data']['val_end_date']

print(f"✅ Features: {len(FEATURE_COLS)} columns")
print(f"✅ Window size: {WINDOW_SIZE} days")
print(f"✅ Train period: up to {TRAIN_END}")
print(f"✅ Val period: {TRAIN_END} → {VAL_END}")
print(f"✅ Test period: after {VAL_END}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Preprocessing — Scaling + Windowing + Chronological Split
# ═══════════════════════════════════════════════════════════════════════════════

def create_sequences(features, targets, window_size):
    """
    Create sliding window sequences.
    Input: features array (N, 12), targets array (N,)
    Output: X (samples, window_size, 12), y (samples,)
    """
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])  # 30 days of features
        y.append(targets[i])                    # Target of the last day
    return np.array(X), np.array(y)


# Process each ticker: Scale → Create Sequences → Split by Date
X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []
X_test_list, y_test_list = [], []

scalers = {}

for ticker in sorted(data['Ticker'].unique()):
    ticker_df = data[data['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

    # ---- Step 1: Fit scaler on TRAIN data only (prevent data leakage) ----
    train_mask = ticker_df['Date'] <= TRAIN_END
    scaler = MinMaxScaler()
    scaler.fit(ticker_df.loc[train_mask, FEATURE_COLS])
    scalers[ticker] = scaler

    # ---- Step 2: Scale ALL data using the train-fitted scaler ----
    scaled_features = scaler.transform(ticker_df[FEATURE_COLS])
    targets = ticker_df[TARGET_COL].values
    dates = ticker_df['Date'].values

    # ---- Step 3: Create sliding window sequences ----
    X_seq, y_seq = create_sequences(scaled_features, targets, WINDOW_SIZE)
    # Each sequence's "date" is the last day in the window
    seq_dates = dates[WINDOW_SIZE:]

    # ---- Step 4: Split sequences by date ----
    train_idx = seq_dates <= np.datetime64(TRAIN_END)
    val_idx = (seq_dates > np.datetime64(TRAIN_END)) & (seq_dates <= np.datetime64(VAL_END))
    test_idx = seq_dates > np.datetime64(VAL_END)

    X_train_list.append(X_seq[train_idx])
    y_train_list.append(y_seq[train_idx])
    X_val_list.append(X_seq[val_idx])
    y_val_list.append(y_seq[val_idx])
    X_test_list.append(X_seq[test_idx])
    y_test_list.append(y_seq[test_idx])

    print(f"  {ticker}: Train={train_idx.sum()}, Val={val_idx.sum()}, Test={test_idx.sum()}")

# Concatenate all tickers
X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_val = np.concatenate(X_val_list)
y_val = np.concatenate(y_val_list)
X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)

print(f"\n{'='*50}")
print(f"📊 Final Dataset Shapes:")
print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}  |  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}  |  y_test:  {y_test.shape}")
print(f"\n  Each sample = {WINDOW_SIZE} days × {len(FEATURE_COLS)} features")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Build Autoencoder (Conv1D)
# ═══════════════════════════════════════════════════════════════════════════════

# --- The Autoencoder learns to compress 30×12 input into a smaller
# --- representation and reconstruct it. This forces the network to
# --- learn the most important patterns in the data.

def build_autoencoder(window_size, n_features, latent_dim=32):
    """
    Conv1D Autoencoder for feature extraction.
    Input shape: (window_size, n_features) = (30, 12)
    Latent space: latent_dim = 32
    """

    # ============ ENCODER ============
    encoder_input = keras.Input(shape=(window_size, n_features), name='encoder_input')

    # Conv Block 1: (30, 12) → (30, 64) → (15, 64)
    x = layers.Conv1D(64, 3, activation='relu', padding='same', name='enc_conv1')(encoder_input)
    x = layers.BatchNormalization(name='enc_bn1')(x)
    x = layers.MaxPooling1D(2, name='enc_pool1')(x)  # → (15, 64)

    # Conv Block 2: (15, 64) → (15, 32) → (5, 32)
    x = layers.Conv1D(32, 3, activation='relu', padding='same', name='enc_conv2')(x)
    x = layers.BatchNormalization(name='enc_bn2')(x)
    x = layers.MaxPooling1D(3, name='enc_pool2')(x)  # → (5, 32)

    # Flatten and compress to latent space
    x = layers.Flatten(name='enc_flatten')(x)  # → 160
    latent = layers.Dense(latent_dim, activation='relu', name='latent_space')(x)  # → 32

    # Create encoder model
    encoder = Model(encoder_input, latent, name='encoder')

    # ============ DECODER ============
    # Rebuild from latent space back to original shape
    x = layers.Dense(5 * 32, activation='relu', name='dec_dense')(latent)  # → 160
    x = layers.Reshape((5, 32), name='dec_reshape')(x)  # → (5, 32)

    # DeConv Block 1: (5, 32) → (15, 32)
    x = layers.UpSampling1D(3, name='dec_up1')(x)  # → (15, 32)
    x = layers.Conv1D(32, 3, activation='relu', padding='same', name='dec_conv1')(x)
    x = layers.BatchNormalization(name='dec_bn1')(x)

    # DeConv Block 2: (15, 32) → (30, 64)
    x = layers.UpSampling1D(2, name='dec_up2')(x)  # → (30, 64)
    x = layers.Conv1D(64, 3, activation='relu', padding='same', name='dec_conv2')(x)
    x = layers.BatchNormalization(name='dec_bn2')(x)

    # Output: reconstruct original (30, 12) — sigmoid because data is MinMaxScaled to [0,1]
    decoder_output = layers.Conv1D(n_features, 3, activation='sigmoid', padding='same', name='decoder_output')(x)

    # Full autoencoder
    autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

    return autoencoder, encoder

# Build the model
autoencoder, encoder = build_autoencoder(WINDOW_SIZE, len(FEATURE_COLS), latent_dim=32)

# Compile
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'  # Mean Squared Error — measures reconstruction quality
)

# Print architecture
print("=" * 60)
print("🔧 AUTOENCODER Architecture")
print("=" * 60)
autoencoder.summary()

print("\n" + "=" * 60)
print("🔧 ENCODER Architecture (feature extractor)")
print("=" * 60)
encoder.summary()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Train Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════

# The autoencoder is UNSUPERVISED — input = output (reconstructs itself)
# No need for labels (y_train)

ae_history = autoencoder.fit(
    X_train, X_train,             # Input = Target (reconstruction)
    validation_data=(X_val, X_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ],
    verbose=1
)

print(f"\n✅ Autoencoder training complete!")
print(f"   Best val_loss: {min(ae_history.history['val_loss']):.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Autoencoder Results — Loss Curve
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(ae_history.history['loss'], label='Train Loss', linewidth=2)
ax.plot(ae_history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_title('Autoencoder — Reconstruction Loss (MSE)', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Autoencoder — Reconstruction Quality Visualization
# ═══════════════════════════════════════════════════════════════════════════════

# Pick a random test sample and compare original vs reconstructed
sample_idx = np.random.randint(0, len(X_test))
original = X_test[sample_idx]
reconstructed = autoencoder.predict(X_test[sample_idx:sample_idx+1], verbose=0)[0]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
features_to_plot = ['Close', 'RSI_14', 'MACD', 'Volume']
feature_indices = [FEATURE_COLS.index(f) for f in features_to_plot]

for i, (feat_name, feat_idx) in enumerate(zip(features_to_plot, feature_indices)):
    ax = axes[i // 2][i % 2]
    ax.plot(original[:, feat_idx], label='Original', linewidth=2, marker='o', markersize=3)
    ax.plot(reconstructed[:, feat_idx], label='Reconstructed', linewidth=2, linestyle='--', marker='x', markersize=3)
    ax.set_title(f'{feat_name} — Original vs Reconstructed', fontsize=12)
    ax.set_xlabel('Day')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Autoencoder Reconstruction Quality (Test Sample #{sample_idx})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Build 1D-CNN Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def build_cnn_classifier(window_size, n_features):
    """
    1D-CNN for binary classification (UP/DOWN prediction).
    Input shape: (window_size, n_features) = (30, 12)
    Output: probability of UP (sigmoid)
    """
    model = keras.Sequential([
        # --- Conv Block 1 ---
        layers.Conv1D(64, kernel_size=3, activation='relu',
                      input_shape=(window_size, n_features), name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling1D(pool_size=2, name='pool1'),

        # --- Conv Block 2 ---
        layers.Conv1D(128, kernel_size=3, activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling1D(pool_size=2, name='pool2'),

        # --- Conv Block 3 ---
        layers.Conv1D(64, kernel_size=3, activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn3'),

        # --- Global Average Pooling (better than Flatten for reducing overfitting) ---
        layers.GlobalAveragePooling1D(name='global_avg_pool'),

        # --- Classification Head ---
        layers.Dropout(0.3, name='dropout1'),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.2, name='dropout2'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='CNN_Classifier')

    return model

# Build
cnn_model = build_cnn_classifier(WINDOW_SIZE, len(FEATURE_COLS))

# Compile
cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("=" * 60)
print("🧠 1D-CNN CLASSIFIER Architecture")
print("=" * 60)
cnn_model.summary()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: Train 1D-CNN
# ═══════════════════════════════════════════════════════════════════════════════

cnn_history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ],
    verbose=1
)

print(f"\n✅ CNN training complete!")
print(f"   Best val_accuracy: {max(cnn_history.history['val_accuracy']):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: CNN Results — Loss & Accuracy Curves
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(cnn_history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(cnn_history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('CNN — Loss', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Crossentropy')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(cnn_history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(cnn_history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].set_title('CNN — Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.suptitle('1D-CNN Training Progress', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: Evaluation on TEST Set
# ═══════════════════════════════════════════════════════════════════════════════

# Predictions
y_pred_prob = cnn_model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Metrics
test_accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("🏆 CNN — TEST SET EVALUATION")
print("=" * 60)
print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)']))


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Confusion Matrix Visualization
# ═══════════════════════════════════════════════════════════════════════════════

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['DOWN (0)', 'UP (1)'],
            yticklabels=['DOWN (0)', 'UP (1)'],
            annot_kws={'size': 16}, ax=ax)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title(f'Confusion Matrix — Test Accuracy: {test_accuracy:.2%}', fontsize=14)
plt.tight_layout()
plt.show()

# Print summary
print(f"\n📝 Summary:")
print(f"   True Negatives (correctly predicted DOWN):  {cm[0][0]}")
print(f"   False Positives (predicted UP, was DOWN):   {cm[0][1]}")
print(f"   False Negatives (predicted DOWN, was UP):   {cm[1][0]}")
print(f"   True Positives (correctly predicted UP):    {cm[1][1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15: Save Models
# ═══════════════════════════════════════════════════════════════════════════════

import os

# Create directories
os.makedirs('../saved_models/deep_learning', exist_ok=True)

# Save Autoencoder (full model)
autoencoder.save('../saved_models/deep_learning/autoencoder.keras')
print("✅ Autoencoder saved → saved_models/deep_learning/autoencoder.keras")

# Save Encoder only (for feature extraction)
encoder.save('../saved_models/deep_learning/encoder.keras')
print("✅ Encoder saved → saved_models/deep_learning/encoder.keras")

# Save CNN Classifier
cnn_model.save('../saved_models/deep_learning/cnn_classifier.keras')
print("✅ CNN Classifier saved → saved_models/deep_learning/cnn_classifier.keras")

print(f"\n🎉 All models saved successfully!")
