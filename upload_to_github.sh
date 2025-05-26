#!/bin/bash

# 🚀 Upload Spam Classifier to GitHub
# Script to upload the spam classifier project to GitHub

echo "🚀 Uploading Spam Classifier to GitHub"
echo "======================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
fi

# Add all files
echo "📁 Adding files to repository..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "🚀 Initial commit: GPT-based SMS Spam Classifier (97.49% accuracy)

- Production-ready spam classifier with 97.49% test accuracy
- CPU-optimized training (< 3 minutes on Mac)
- Complete documentation and testing scripts
- Modified GPT architecture for text classification
- Comprehensive loss tracking and stability features"

# Add remote origin (replace with your actual GitHub repo URL)
echo "🔗 Adding remote origin..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/code50/195839303.git

# Create main branch if it doesn't exist
git branch -M main

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
echo "Note: You may need to authenticate with GitHub"
git push -u origin main

echo ""
echo "✅ Upload complete!"
echo "🌐 Your repository should now be available at:"
echo "   https://github.com/code50/195839303"
echo ""
echo "📋 Next steps:"
echo "1. Visit your GitHub repository"
echo "2. Add a repository description"
echo "3. Consider adding topics/tags for discoverability"
echo "" 