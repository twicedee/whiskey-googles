#!/bin/bash

# Exit on error
set -e

echo "🔍 Step 1: Updating .gitignore..."
cat <<EOL >> .gitignore

# Ignore large binary files and environments
backend/Lib/
__pycache__/
*.pyc
*.dll
*.pyd
*.pth
*.so
*.zip
*.tar.gz

# Ignore processed data
dataset/processed/ocr_models/
EOL

echo "🧹 Step 2: Untracking files..."
git rm --cached -r backend/Lib/site-packages/ || true
git rm --cached dataset/processed/ocr_models/craft_mlt_25k.pth || true
git rm --cached -r __pycache__/ || true

echo "📦 Step 3: Installing git-filter-repo..."
pip install git-filter-repo

echo "🧼 Step 4: Cleaning Git history..."
git filter-repo --path backend/Lib/site-packages/ --invert-paths
git filter-repo --path dataset/processed/ocr_models/craft_mlt_25k.pth --invert-paths

echo "📁 Step 5: Setting up Git LFS..."
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add dataset/processed/ocr_models/craft_mlt_25k.pth

echo "✅ Step 6: Final commit and force push..."
git add .gitignore
git commit -m "Cleaned large files and setup Git LFS"
git push origin --force

echo "🎉 Done! Your repo is now clean and ready."
