"""
Generate synthetic housing data with image features
"""

import pandas as pd
import numpy as np
from PIL import Image
import os

np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate tabular features
data = {
    'house_id': range(1, n_samples + 1),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.round(np.random.uniform(1, 4, n_samples), 1),
    'sqft_living': np.random.randint(500, 4000, n_samples),
    'sqft_lot': np.random.randint(1000, 10000, n_samples),
    'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
    'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'view': np.random.randint(0, 4, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(3, 13, n_samples),
    'sqft_above': np.random.randint(500, 3500, n_samples),
    'sqft_basement': np.random.randint(0, 1500, n_samples),
    'yr_built': np.random.randint(1900, 2024, n_samples),
    'yr_renovated': np.random.choice([0, np.random.randint(1980, 2024)], n_samples, p=[0.8, 0.2]),
    'zipcode': np.random.randint(98001, 98200, n_samples),
    'lat': np.random.uniform(47.3, 47.8, n_samples),
    'long': np.random.uniform(-122.5, -121.8, n_samples),
}

df = pd.DataFrame(data)

# Calculate price based on features (synthetic relationship)
price = (
    df['sqft_living'] * 200 +
    df['bedrooms'] * 5000 +
    df['bathrooms'] * 8000 +
    df['floors'] * 10000 +
    df['view'] * 15000 +
    df['condition'] * 8000 +
    df['grade'] * 15000 +
    df['waterfront'] * 100000 +
    np.random.normal(0, 30000, n_samples)
)

# Add image quality factor (simulating image features)
# Higher quality/renovated houses get better scores
image_quality = (
    (df['condition'] / 10) +
    (df['grade'] / 50) +
    (df['yr_renovated'] > 0) * 0.2 +
    np.random.normal(0, 0.1, n_samples)
)
price = price * (1 + image_quality)

df['price'] = price.astype(int)

# Ensure price is positive
df['price'] = np.abs(df['price'])

print(f"Dataset created: {len(df)} samples")
print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"Average price: ${df['price'].mean():,.0f}")

# Save CSV
df.to_csv('housing_data.csv', index=False)
print("\n✅ housing_data.csv saved!")

# Create synthetic images
print("\nGenerating synthetic images...")
os.makedirs('images', exist_ok=True)

for i in range(min(100, n_samples)):  # Generate 100 sample images
    # Create a simple colored image based on house features
    width, height = 224, 224
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Color based on house price (red = expensive, blue = cheap)
    price_norm = min(255, int(df.loc[i, 'price'] / 20000))
    color = (price_norm, 100, 255 - price_norm)
    
    # Create gradient pattern
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [color[0], color[1], color[2]]
    
    # Add some pattern based on bedrooms
    bedrooms = df.loc[i, 'bedrooms']
    for b in range(bedrooms):
        x = (b + 1) * 50
        y = height // 2
        img_array[y-10:y+10, x-10:x+10] = [255, 255, 255]
    
    # Save image
    img = Image.fromarray(img_array)
    img.save(f'images/house_{i+1}.jpg')

print(f"✅ Generated {len(os.listdir('images'))} synthetic images")

print("\n" + "="*50)
print("Dataset generation complete!")
print("="*50)