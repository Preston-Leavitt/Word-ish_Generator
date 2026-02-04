#!/bin/bash
# Setup script for Word-ish Generator on Linux

echo "🎨 Setting up Word-ish Generator..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  No .env file found. Creating template..."
    cat > .env << 'EOL'
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here

# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production

# Optional: Price tracking (per 1K tokens)
OPENAI_INPUT_PRICE_PER_1K=0.00015
OPENAI_OUTPUT_PRICE_PER_1K=0.0006

# Optional: Media selection settings
MEDIA_SAMPLE_RATIO=1
MEDIA_LIST_CHAR_BUDGET=45000

# Optional: Word existence check via AI (1=yes, 0=no)
WORD_EXISTENCE_VIA_AI=1
EOL
    echo "✓ Created .env template. Please edit it with your API keys."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the server, run:"
echo "  python generator.py"
echo ""
echo "Don't forget to set your OPENAI_API_KEY in the .env file!"
