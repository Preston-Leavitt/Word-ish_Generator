# Word-ish Generator Setup Guide

## Quick Start (Linux/Mac)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Word-ish_Generator
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure your environment:**
   - Edit `.env` file with your OpenAI API key
   ```bash
   nano .env
   ```

4. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

5. **Run the server:**
   ```bash
   python generator.py
   ```

6. **Access the app:**
   - Open browser to http://127.0.0.1:5000

---

## Manual Setup (Linux/Mac)

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file (copy from template or create manually)
cp .env.example .env  # if you have one
nano .env  # Add your OPENAI_API_KEY

# Run the application
python generator.py
```

---

## Windows Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create/edit .env file with your API key
notepad .env

# Run the application
python generator.py
```

---

## Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key

Optional:
- `SECRET_KEY` - Flask secret key (change in production)
- `OPENAI_INPUT_PRICE_PER_1K` - Token price tracking
- `OPENAI_OUTPUT_PRICE_PER_1K` - Token price tracking
- `MEDIA_SAMPLE_RATIO` - Media selection ratio (0-1)
- `WORD_EXISTENCE_VIA_AI` - Use AI for word existence checks

---

## Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

---

## Updating Dependencies

If you install new packages:
```bash
pip freeze > requirements.txt
```

---

## Troubleshooting

**Virtual environment not activating?**
- Linux/Mac: Make sure you use `source .venv/bin/activate`
- Windows: Use `.venv\Scripts\activate`

**Module not found errors?**
- Make sure virtual environment is activated (you should see `(.venv)` in your prompt)
- Run `pip install -r requirements.txt` again

**Database errors?**
- The database will be created automatically on first run
- Check that you have write permissions in the project directory
