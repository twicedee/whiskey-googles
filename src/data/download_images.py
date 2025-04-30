'''This file downloads the images and updates the Data frame of the whiskey bottles'''

import os
import requests
import pandas as pd
from tqdm import tqdm  

# Load the dataset
df = pd.read_csv("dataset/raw/501 Bottle Dataset - Sheet1.csv")
os.makedirs("images", exist_ok=True)

def download_image(url, id):

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        
        # Generate filename: id.jpg
        filename = f"images/{int(id)}.jpg"
        
        # Save the image
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return False
    
    
'''
Changing the image URL to the image directory
'''
def update_image_url(df):
    df["image_url"] = df.apply(
        lambda row: f"images/{int(row['id'])}.jpg" 
        if pd.notna(row["id"]) else None,
        axis=1
    )
    return df


# Download images with progress tracking
success_count = 0
for index, row in tqdm(df.iterrows(), total=len(df)):
    if pd.notna(row['image_url']) and pd.notna(row['id']):
        if download_image(row['image_url'], row['id']):
            success_count += 1

print(f"\nSuccessfully downloaded {success_count}/{len(df)} images")
print(f"Images saved to: {os.path.abspath('images')}")


df = update_image_url(df)

#Save the dataset
df.to_csv("dataset/processed/clean_image_whisky_dataset.csv", index=False)