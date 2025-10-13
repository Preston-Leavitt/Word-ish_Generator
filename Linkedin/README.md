# LinkedIn Viral Post Generator

MIT License - A FastAPI service that generates viral LinkedIn posts using OpenAI's GPT models.

## Features

- Template-based post generation from PDF templates
- **DM Funnel Generation**: Complete 6-message DM sequences with intent tags
- **Video Generation**: AI-powered video creation with Sora integration
- Content safety checks (PII, profanity, defamation)
- Structured JSON output with posts, hooks, hashtags, and more
- **Multi-content Interface**: Switch between Posts, DM Funnels, and Videos
- Configurable tone, audience, and goals
- Fallback templates when PDF parsing fails

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*This installs FastAPI, OpenAI client, PDF parsing libraries, and video processing tools.*

### 2. Install FFmpeg (Required for Video Generation)
**Windows:**
```bash
# Download FFmpeg from https://ffmpeg.org/download.html
# Add FFmpeg to your PATH environment variable
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### 3. Set Up Environment
```bash
cp .env.example .env
```
*Creates your environment file from the template.*

Edit `.env` file and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```
*The API key is required to generate posts. Get one from OpenAI's website.*

### 4. Create Required Directories
```bash
mkdir -p app tests static
touch app/__init__.py
```
*Creates the Python package structure and placeholder directories.*

### 5. Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Starts the FastAPI development server with auto-reload on code changes.*

### 6. Test the API
Visit `http://localhost:8000` in your browser
*Shows a beautiful web interface for generating LinkedIn posts*

Visit `http://localhost:8000/docs` for interactive API documentation
*FastAPI automatically generates Swagger UI for testing endpoints.*

## Web Interface

The application now includes a user-friendly web interface at `http://localhost:8000` featuring:

- **Template Selection**: Choose from available post templates
- **Form Fields**: Easy input for tone, audience, goals, and content
- **Live Preview**: See your generated post with formatting
- **Metadata Display**: View hooks, hashtags, CTAs, and follow-up suggestions
- **Responsive Design**: Works on desktop and mobile devices

## What Each Command Does

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Installs all Python dependencies |
| `cp .env.example .env` | Creates environment configuration file |
| `mkdir -p app tests static` | Creates necessary directory structure |
| `touch app/__init__.py` | Makes `app` a Python package |
| `uvicorn app.main:app --reload` | Starts development server with hot reload and web UI |
| `curl -X GET http://localhost:8000/health` | Checks if API is running |
| `curl -X GET http://localhost:8000/templates` | Lists available post templates |
| `pytest tests/` | Runs the test suite |

## API Endpoints

### GET / 
*Modern web interface for generating posts, DM funnels, and videos*

### GET /health
*Health check - returns server status and template count*

### GET /templates
*List available post templates*

### POST /generate
*Generate LinkedIn post with complete DM funnel*
 
### POST /render_video
*Start video generation job with Sora AI*

### GET /video_status/{job_id}
*Check video generation progress*

## New DM Funnel Features

Each generated post now includes:

- **dm_cta**: Single token phrase for post CTA (e.g., "PLAYBOOK")
- **dm_flow**: Complete 6-message sequence:
  - Initial message (intent: lead)
  - 2 Follow-ups (intent: nurture)
  - Follow-up question (intent: question)  
  - Qualification question (intent: hire)
  - Meeting booking template (intent: hire)

## Video Generation Features

- **Sora AI Integration**: Generate videos from scripts
- **Real Video Creation**: Generates actual MP4 files using FFmpeg
- **Text Overlays**: Displays script content as styled text overlays
- **Multiple Formats**: Vertical (9:16), Landscape (16:9), Square (1:1)
- **Style Options**: Realistic backgrounds with professional text overlays
- **Duration Control**: 5-20 seconds with timed text segments
- **Download Support**: Direct download of generated videos
- **Auto-play Preview**: Videos play automatically in the browser

## Example Video Request

```bash
curl -X POST "http://localhost:8000/render_video" \
     -H "Content-Type: application/json" \
     -d '{
       "script": "Stop chasing viral content. Here's how to build a targeted audience that actually converts...",
       "hooks": ["Stop chasing virality", "Build targeted audience", "Convert your content"],
       "aspect": "vertical",
       "duration_sec": 15,
       "style": "realistic"
     }'
```

## Docker Setup (Optional)

### Build Image
```bash
docker build -t linkedin-generator .
```
*Creates a Docker image with your application*

### Run Container
```bash
docker run -p 8000:8000 --env-file .env linkedin-generator
```
*Runs the application in a container, accessible at localhost:8000*

## Development Workflow

### 1. Make Code Changes
*Edit files in the `app/` directory*

### 2. Server Auto-Reloads
*If using `--reload`, changes are picked up automatically*

### 3. Run Tests
```bash
pytest tests/ -v
```
*Runs all tests with verbose output to verify your changes*

### 4. Check Logs
*Monitor the console where uvicorn is running for errors and debug info*

## Troubleshooting

### "Directory does not exist" error
```bash
mkdir -p app tests static
touch app/__init__.py
```

### "OpenAI API key not found" error
*Check your `.env` file has `OPENAI_API_KEY=your_actual_key`*

### Import conflicts with installed packages
If you get import errors about conflicting package names:
```bash
# Option 1: Use virtual environment (recommended)
python -m venv linkedin-env
# On Windows:
linkedin-env\Scripts\activate
# On Mac/Linux:
source linkedin-env/bin/activate

pip install -r requirements.txt
```

### "Cannot import name" errors
Make sure all required files exist:
```bash
# Check all required files are present
ls app/
# Should show: __init__.py, main.py, schemas.py, templates.py, prompts.py, openai_client.py, safety.py, utils.py
```

### Port already in use
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
*Use a different port number*

### "Missing parentheses in call to 'print'" or Import conflicts
This happens when there's a conflicting package installed:
```bash
# Uninstall conflicting schemas package
pip uninstall schemas

# Or use virtual environment to isolate dependencies
python -m venv linkedin-env
source linkedin-env/bin/activate  # On Windows: linkedin-env\Scripts\activate
pip install -r requirements.txt
```
*Creates a clean environment without conflicting packages*

### "Schema violation for key" error
*Ensure your input JSON matches the expected schema for the generate endpoint.*

## Content Safety

The service automatically checks for:
- PII (emails, phone numbers)
- Profanity
- Potential defamation of named persons
- Content length limits
- Hook length limits
- **DM appropriateness**: Professional tone validation

## Post-Processing Integration Points

For video generation, integrate with:

- **FFmpeg**: Subtitle overlay, trimming, format conversion
- **Third-party APIs**: Brand bar overlays, watermarks
- **Cloud Storage**: Video hosting and CDN delivery
- **Webhook Support**: Real-time status updates

Example FFmpeg command for subtitle overlay:
```bash
ffmpeg -i input.mp4 -vf "subtitles=captions.srt" output_with_subs.mp4
```

## License

MIT License - feel free to use in your projects.
