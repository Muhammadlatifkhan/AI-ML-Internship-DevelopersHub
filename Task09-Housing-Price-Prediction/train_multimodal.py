"""
Task 9: Multimodal House Price Prediction
Combining CNN features from images with tabular data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 9: MULTIMODAL HOUSE PRICE PREDICTION")
print("=" * 60)

# ========== 1. CONFIGURATION ==========
BATCH_SIZE = 16  # Reduced for CPU
EPOCHS = 30
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n✅ Using device: {DEVICE}")

# ========== 2. LOAD TABULAR DATA ==========
print("\n[1] Loading tabular data...")

df = pd.read_csv('data/housing_data.csv')
print(f"✅ Loaded {len(df)} samples")
print(f"   Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

# ========== 3. PREPARE TABULAR FEATURES ==========
print("\n[2] Preparing tabular features...")

# Drop house_id and target
feature_cols = [col for col in df.columns if col not in ['house_id', 'price']]
X_tabular = df[feature_cols].values
y = df['price'].values

print(f"   Tabular features shape: {X_tabular.shape}")
print(f"   Feature names: {feature_cols[:5]}...")

# Scale tabular features
tabular_scaler = StandardScaler()
X_tabular_scaled = tabular_scaler.fit_transform(X_tabular)
print(f"✅ Tabular features scaled")

# ========== 4. NORMALIZE TARGET PRICES ==========
print("\n[3] Normalizing target prices...")

price_scaler = StandardScaler()
y_normalized = price_scaler.fit_transform(y.reshape(-1, 1)).flatten()
print(f"✅ Target prices normalized (mean=0, std=1)")
print(f"   Original price mean: ${y.mean():,.2f}")
print(f"   Original price std: ${y.std():,.2f}")

# ========== 5. TRAIN-TEST SPLIT ==========
print("\n[4] Splitting data...")

# Split indices
train_idx, test_idx = train_test_split(
    range(len(df)), test_size=0.2, random_state=42
)

X_train_tab = X_tabular_scaled[train_idx]
X_test_tab = X_tabular_scaled[test_idx]
y_train_norm = y_normalized[train_idx]
y_test_norm = y_normalized[test_idx]
y_train_orig = y[train_idx]
y_test_orig = y[test_idx]

print(f"✅ Train samples: {len(train_idx)}")
print(f"✅ Test samples: {len(test_idx)}")

# ========== 6. CREATE DATASET CLASS ==========
print("\n[5] Creating dataset with images...")

class HouseDataset(Dataset):
    def __init__(self, indices, image_dir, X_tabular, y_norm, transform=None):
        self.indices = indices
        self.image_dir = image_dir
        self.X_tabular = X_tabular
        self.y = y_norm
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'house_{original_idx + 1}.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Create blank image if not found
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
        image = self.transform(image)
        
        # Get tabular features
        tabular = torch.FloatTensor(self.X_tabular[idx])
        
        # Get normalized price
        price = torch.FloatTensor([self.y[idx]])
        
        return image, tabular, price

# Create datasets
train_dataset = HouseDataset(
    train_idx, 'data/images', X_train_tab, y_train_norm
)
test_dataset = HouseDataset(
    test_idx, 'data/images', X_test_tab, y_test_norm
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Datasets created successfully")

# ========== 7. BUILD MULTIMODAL MODEL ==========
print("\n[6] Building multimodal model...")

class MultimodalHousePriceModel(nn.Module):
    def __init__(self, tabular_input_dim, cnn_output_dim=512, hidden_dim=256):
        super(MultimodalHousePriceModel, self).__init__()
        
        # CNN for image feature extraction (ResNet18)
        self.cnn = models.resnet18(weights='IMAGENET1K_V1')
        # Remove the final classification layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.cnn_output_dim = cnn_output_dim
        
        # Tabular feature branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Fusion layer (combine image and tabular features)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: single price value
        )
        
    def forward(self, image, tabular):
        # Extract image features
        batch_size = image.shape[0]
        image_features = self.cnn(image)
        image_features = image_features.view(batch_size, -1)
        
        # Process tabular features
        tabular_features = self.tabular_branch(tabular)
        
        # Fuse features
        combined = torch.cat([image_features, tabular_features], dim=1)
        price = self.fusion(combined)
        
        return price.squeeze()

# Get input dimensions
tabular_input_dim = X_tabular_scaled.shape[1]
model = MultimodalHousePriceModel(tabular_input_dim).to(DEVICE)

print(f"✅ Model created!")
print(f"   Tabular input dim: {tabular_input_dim}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========== 8. SETUP TRAINING ==========
print("\n[7] Setting up training...")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ========== 9. TRAINING LOOP ==========
print("\n[8] Training multimodal model...")
print("-" * 60)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    train_batches = 0
    
    for images, tabulars, prices in train_loader:
        images = images.to(DEVICE)
        tabulars = tabulars.to(DEVICE)
        prices = prices.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images, tabulars)
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
    
    avg_train_loss = train_loss / train_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, tabulars, prices in test_loader:
            images = images.to(DEVICE)
            tabulars = tabulars.to(DEVICE)
            prices = prices.to(DEVICE)
            
            outputs = model(images, tabulars)
            loss = criterion(outputs, prices)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    val_losses.append(avg_val_loss)
    
    # Adjust learning rate
    scheduler.step(avg_val_loss)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"   Epoch [{epoch+1:2d}/{EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

print("✅ Training complete!")

# ========== 10. EVALUATE MODEL ==========
print("\n[9] Evaluating model performance...")

model.eval()
predictions_norm = []
actuals_norm = []

with torch.no_grad():
    for images, tabulars, prices in test_loader:
        images = images.to(DEVICE)
        tabulars = tabulars.to(DEVICE)
        
        outputs = model(images, tabulars)
        predictions_norm.extend(outputs.cpu().numpy())
        actuals_norm.extend(prices.cpu().numpy())

# Convert back to original price scale
predictions = price_scaler.inverse_transform(np.array(predictions_norm).reshape(-1, 1)).flatten()
actuals = price_scaler.inverse_transform(np.array(actuals_norm).reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print(f"\n📊 Performance Metrics:")
print(f"   MAE (Mean Absolute Error): ${mae:,.2f}")
print(f"   RMSE (Root Mean Square Error): ${rmse:,.2f}")
print(f"   R² Score: {r2:.4f}")
print(f"   MAPE (Mean Absolute % Error): {mape:.2f}%")

# ========== 11. SAVE MODEL ==========
print("\n[10] Saving model and scalers...")

os.makedirs('outputs', exist_ok=True)

# Save full model state
torch.save({
    'model_state_dict': model.state_dict(),
    'tabular_input_dim': tabular_input_dim,
    'tabular_scaler': tabular_scaler,
    'price_scaler': price_scaler,
    'feature_cols': feature_cols,
    'model_architecture': 'MultimodalHousePriceModel',
    'train_losses': train_losses,
    'val_losses': val_losses
}, 'outputs/multimodal_model.pth')

# Also save scalers separately for easy loading
import joblib
joblib.dump(tabular_scaler, 'outputs/tabular_scaler.pkl')
joblib.dump(price_scaler, 'outputs/price_scaler.pkl')

print("✅ Model saved to: outputs/multimodal_model.pth")
print("✅ Scalers saved to: outputs/")

# ========== 12. VISUALIZE RESULTS ==========
print("\n[11] Generating visualizations...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(train_losses, label='Train Loss', marker='o', linewidth=2, markersize=4)
axes[0, 0].plot(val_losses, label='Validation Loss', marker='s', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training History')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Actual vs Predicted
axes[0, 1].scatter(actuals, predictions, alpha=0.5, s=20)
min_val = min(actuals.min(), predictions.min())
max_val = max(actuals.max(), predictions.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Prices ($)')
axes[0, 1].set_ylabel('Predicted Prices ($)')
axes[0, 1].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Residual plot
residuals = actuals - predictions
axes[1, 0].scatter(predictions, residuals, alpha=0.5, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Prices ($)')
axes[1, 0].set_ylabel('Residuals ($)')
axes[1, 0].set_title('Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# Error distribution
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Error Distribution')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/results_plot.png', dpi=150, bbox_inches='tight')
print("✅ Results plot saved to: outputs/results_plot.png")

# ========== 13. FEATURE IMPORTANCE (Tabular only) ==========
print("\n[12] Analyzing feature importance...")

# Train a simple Random Forest for feature importance
from sklearn.ensemble import RandomForestRegressor

rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_train_tab, y_train_orig)

feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Most Important Tabular Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Save feature importance
feature_importance_df.to_csv('outputs/feature_importance.csv', index=False)
print("✅ Feature importance saved to: outputs/feature_importance.csv")

# ========== 14. SUMMARY ==========
print("\n" + "=" * 60)
print("✅ TASK 9 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Final Model Performance:")
print(f"   MAE: ${mae:,.2f}")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   R² Score: {r2:.4f}")
print(f"   MAPE: {mape:.2f}%")
print("\n🎯 Multimodal learning achieved!")
print("   ✓ Combined CNN (ResNet18) for images")
print("   ✓ Tabular data processing branch")
print("   ✓ Feature fusion architecture")
print("   ✓ Proper price normalization")
print("   ✓ Production-ready model saved")
print("\n📁 Output files:")
print("   • outputs/multimodal_model.pth - Full trained model")
print("   • outputs/tabular_scaler.pkl - Feature scaler")
print("   • outputs/price_scaler.pkl - Price normalizer")
print("   • outputs/results_plot.png - Visualization")
print("   • outputs/feature_importance.csv - Top features")