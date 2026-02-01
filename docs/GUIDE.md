# Smart Notes - Complete Guide

## 📚 Project Overview

Smart Notes is a **multi-stage GenAI reasoning pipeline** for structured understanding of classroom content. It processes audio lectures, notes, equations, and context to produce comprehensive educational analysis through 7 independent reasoning stages.

### What It Does
- **Audio Processing**: Transcribes lectures using OpenAI Whisper
- **Image OCR**: Extracts text from handwritten notes, blackboard, and whiteboard photos
- **Text Processing**: Segments and normalizes content from multiple sources
- **Multi-Stage Reasoning**: 7 independent reasoning stages extract:
  1. Topic Identification
  2. Concept Extraction
  3. Equation Interpretation
  4. Misconception Detection
  5. FAQ Generation
  6. Real-World Connections
  7. Worked Example Analysis
- **Quality Evaluation**: 4-metric evaluation framework assesses output
- **Session Management**: Persists data and generates cumulative study guides
- **Web UI**: Streamlit interface for easy interaction

### Key Features
✅ Multi-stage reasoning (not single prompt)  
✅ Structured JSON output with Pydantic validation  
✅ Production-grade logging with rotation  
✅ Environment-based configuration  
✅ Session persistence and aggregation  
✅ Educational metrics evaluation  

---

## 🚀 Quick Start (2–3 minutes)

### 1. Create & Activate Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Add your key to .env:
```ini
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run Application
```bash
python -m streamlit run app.py
```

**Access at**: http://localhost:8501

---

## 📋 Requirements

- Python 3.8+
- OpenAI API key (free tier works)
- 2GB RAM minimum
- Internet connection

---

## ⚙️ Installation & Setup

### Step 1: Create & Activate Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

1. Get API key from https://platform.openai.com/api-keys
2. Edit `.env`:
    ```ini
    OPENAI_API_KEY=sk-your-key-here
    ```
3. Run:
    ```bash
    python -m streamlit run app.py
    ```

### Step 4: Verify Setup
```bash
python test_config.py
```

Expected output:
```
✓ Configuration loaded successfully!
    ENVIRONMENT: development
    DEBUG: True
```

---

## 🎮 How to Use

### Web Interface (Recommended)

```bash
streamlit run app.py
```

**Features:**
1. **Example Runner** - Try pre-built examples with one click
2. **Manual Input** - Upload audio/notes or paste content
3. **Structured Output** - 9 sections of analysis
4. **Session History** - View past sessions
5. **Study Guide** - Generate cumulative study materials

### CLI Usage

```python
from src.reasoning.pipeline import ReasoningPipeline

pipeline = ReasoningPipeline()
result = pipeline.process(
    audio_transcript="Lecture content here...",
    notes="Student notes here...",
    equations=["y = mx + b", "F = ma"],
    context="Additional context..."
)

print(result.to_dict())
```

### Example Script

```bash
python examples/demo_usage.py
```

---

## 📁 Project Structure

```
project/
├── app.py                          # Streamlit web interface
├── config.py                       # Configuration management
├── requirements.txt                # Dependencies
│
├── .env                            # Configuration (git-ignored)
├── .env.example                    # Configuration template
├── .gitignore                      # Git protection
│
├── src/
│   ├── audio/
│   │   └── transcription.py        # Audio to text (Whisper)
│   ├── preprocessing/
│   │   └── text_processing.py      # Text segmentation & normalization
│   ├── reasoning/
│   │   └── pipeline.py             # 7-stage reasoning pipeline
│   ├── schema/
│   │   └── output_schema.py        # Pydantic validation models
│   ├── evaluation/
│   │   └── metrics.py              # Quality evaluation (4 metrics)
│   ├── study_book/
│   │   └── session_manager.py      # Session persistence
│   └── logging_config.py           # Production logging
│
├── examples/
│   ├── demo_usage.py               # CLI example
│   ├── inputs/                     # Sample data
│   └── notes/                      # Sample notes
│
├── data/
│   ├── cache/                      # Cache files
│   └── audio_cache/                # Audio cache
│
├── logs/                           # Application logs
│
├── outputs/
│   └── sessions/                   # Session data
│
└── docs/
    └── GUIDE.md                    # This file
```

---

## ⚙️ Configuration

### Core Settings

Edit `.env` to customize:

```ini
# Environment
ENVIRONMENT=development              # development, staging, production
DEBUG=true                          # Enable debug mode

# API
OPENAI_API_KEY=sk-your-key          # Your API key
LLM_MODEL=gpt-4                     # Model: gpt-4 or gpt-3.5-turbo

# Features
LOG_LEVEL=DEBUG                     # DEBUG, INFO, WARNING, ERROR

# Reasoning Pipeline
MAX_TOPICS=10                        # Max topics to extract
MAX_FAQS=15                          # FAQ items to generate
MAX_WORKED_EXAMPLES=5               # Example problems
```

**See `.env.example` for 100+ configuration options**

---

## 🔧 Architecture

### 7-Stage Reasoning Pipeline

Each stage is independent and produces structured output:

```
Input (audio, notes, equations, context)
    ↓
1️⃣ Topic Identification
    Extracts major topics covered
    ↓
2️⃣ Concept Extraction
    Identifies key concepts per topic
    ↓
3️⃣ Equation Interpretation
    Converts equations to plain language
    ↓
4️⃣ Misconception Detection
    Identifies common student mistakes
    ↓
5️⃣ FAQ Generation
    Creates frequently asked questions
    ↓
6️⃣ Real-World Connections
    Links to practical applications
    ↓
7️⃣ Worked Example Analysis
    Provides example problem solutions
    ↓
Output (Structured JSON with all 7 stages)
```

### Processing Flow

```
Audio File
    ↓
Whisper (Audio → Text)
    ↓
Text Processor (Segmentation, Normalization)
    ↓
Reasoning Pipeline (7 Stages)
    ↓
Evaluator (4 Quality Metrics)
    ↓
Output Schema (Pydantic Validation)
    ↓
Session Manager (Persistence)
    ↓
JSON Output + Study Guide
```

### 4-Metric Evaluation

Output quality is evaluated on:

1. **Reasoning Correctness** (0-1)
   - Accuracy of concepts and analysis

2. **Structural Accuracy** (0-1)
   - Correct JSON schema and organization

3. **Hallucination Rate** (0-1)
   - Percentage of fabricated information

4. **Educational Usefulness** (1-5)
   - How helpful for learning

Thresholds set in `.env`:
```ini
MIN_REASONING_CORRECTNESS=0.7
MIN_STRUCTURAL_ACCURACY=0.8
MAX_HALLUCINATION_RATE=0.15
MIN_EDUCATIONAL_USEFULNESS=3.5
```

---

## 📊 Input & Output

### Input Format

**Audio + Text + Context:**
```python
{
    "audio_file": "lecture.wav",
    "notes": "Student notes...",
    "equations": ["E = mc²", "F = ma"],
    "context": "Additional context..."
}
```

**Image Upload (NEW!):**
```python
{
    "images": ["notes1.jpg", "whiteboard.png", "blackboard.jpg"],
    "audio_file": "lecture.wav",  # Optional
    "equations": ["..."],
    "context": "..."
}
```

### 📸 Image OCR Feature

Upload photos of handwritten notes, blackboard, or whiteboard and automatically extract text!

**Supported Formats:**
- JPG, JPEG, PNG, BMP
- Multiple images supported
- Handwritten notes
- Blackboard/whiteboard photos
- Printed documents

**How to Use:**
1. Open the Streamlit app
2. Select "Upload Images" option
3. Click "Browse files" and select your images
4. Preview thumbnails to verify upload
5. Click "Process Session" to extract text

**OCR Backend:**
- **Primary**: EasyOCR (deep learning-based, high accuracy)
- **Fallback**: Tesseract OCR (traditional OCR)
**Tips for Best Results:**
- Use good lighting
- Avoid shadows and glare
- Clear, legible handwriting
- High-resolution photos (recommended)
- Straight-on angle (not tilted)

**Image Preprocessing:**
- Automatic grayscale conversion
- Contrast enhancement
- Noise reduction
- Binarization for text detection

**Example:**
```bash
# 1. Start app
streamlit run app.py

# 2. Select "Upload Images"
# 3. Upload: lecture_notes.jpg, board_photo.png
# 4. Process → Text extracted automatically!
```

**Installation:**
```bash
pip install easyocr Pillow opencv-python-headless
```

### Output Format

**Structured JSON with 7 sections:**
```json
{
    "session_id": "2024-01-15-123456",
    "topics": [
        {
            "name": "Calculus",
            "concepts": ["Derivatives", "Limits"],
            "equations": [...]
        }
    ],
    "faqs": [...],
    "misconceptions": [...],
    "worked_examples": [...],
    "evaluation": {
        "reasoning_correctness": 0.92,
        "hallucination_rate": 0.08
    }
}
```

---

## 📝 Usage Examples

### Example 1: Run Pre-Built Demo
```bash
streamlit run app.py
```
Click "Example Runner" → Select example → See results

### Example 2: Upload Your Own
```bash
streamlit run app.py
```
Upload audio file + notes → Process → View results

### Example 3: CLI Processing
```python
from src.reasoning.pipeline import ReasoningPipeline

pipeline = ReasoningPipeline()
result = pipeline.process(
    audio_transcript="Learning about derivatives...",
    notes="Chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x)",
    equations=["d/dx[x²] = 2x"],
    context="Calculus lecture"
)

print(result.topics)
print(result.evaluation)
```

---

## 🚀 Running

### Production (Real API)
```bash
# 1. Add API key to .env
OPENAI_API_KEY=sk-your-real-key

# 2. Run
streamlit run app.py
```

---

## 📊 Logging

### Access Logs
```bash
# View latest logs
tail -f logs/smart_notes_*.log

# Search for errors
grep ERROR logs/smart_notes_*.log

# View today's logs
tail -100 logs/smart_notes_$(date +%Y-%m-%d).log
```

### Log Levels
- **DEBUG**: Detailed execution info (development)
- **INFO**: Important events (default)
- **WARNING**: Warning messages
- **ERROR**: Error messages only

Change log level:
```ini
# .env
LOG_LEVEL=DEBUG      # Verbose
LOG_LEVEL=INFO       # Normal
LOG_LEVEL=WARNING    # Errors only
```

---

## 🔍 Testing

### Verify Configuration
```bash
python test_config.py
```

### Run with Example Data
```bash
# Via web UI
streamlit run app.py
# Click "Example Runner"

# Via CLI
python examples/demo_usage.py
```

### Test Each Component
```python
# Test transcription
from src.audio.transcription import AudioTranscriber
transcriber = AudioTranscriber()
text = transcriber.transcribe("audio.wav")

# Test preprocessing
from src.preprocessing.text_processing import TextPreprocessor
processor = TextPreprocessor()
segments = processor.process("lecture text")

# Test reasoning
from src.reasoning.pipeline import ReasoningPipeline
pipeline = ReasoningPipeline()
result = pipeline.process(audio_transcript=segments)

# Test evaluation
from src.evaluation.metrics import ContentEvaluator
evaluator = ContentEvaluator()
score = evaluator.evaluate(result)
```

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# 1. Check config
python test_config.py

# 2. Check dependencies
pip install -r requirements.txt

# 3. Run with debug
DEBUG=true streamlit run app.py
```

### API Errors
```bash
# 1. Verify API key
echo $OPENAI_API_KEY

# 2. Check logs
tail -f logs/smart_notes_*.log
```

### Logs Not Creating
```bash
# 1. Check directory exists
ls -la logs/

# 2. Check permissions
chmod 755 logs/

# 3. Check LOG_OUTPUT setting
grep LOG_OUTPUT .env
```

### Out of Memory
```bash
# Reduce limits in .env
MAX_TOPICS=5
MAX_FAQS=8
WHISPER_MODEL_SIZE=tiny
```

---

## 📚 API Keys & Costs

### Getting OpenAI API Key

1. Visit https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy key (shown only once)
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

### Pricing (Jan 2024)

**Models:**
- GPT-4: $0.03/1K input, $0.06/1K output
- GPT-3.5-turbo: $0.0005/1K input, $0.0015/1K output

**Recommendation:**
- Testing: Use gpt-3.5-turbo ($0.001-0.01 per request)
- Production: Use gpt-4 (better quality)

---

## 🔐 Security

### Keep Safe
✅ Never share `.env` file  
✅ Never commit `.env` to git (already in .gitignore)  
✅ Rotate API keys regularly  
✅ Use different keys per environment  

### Protecting Your Key
```bash
# Check if .env is protected
git status
# Should show: nothing added to commit

# Verify .gitignore works
cat .gitignore | grep ".env"
```

---

## 🤝 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `config.py` | Configuration loader (reads .env) |
| `src/audio/transcription.py` | Audio → Text via Whisper |
| `src/preprocessing/text_processing.py` | Segment & normalize text |
| `src/reasoning/pipeline.py` | 7-stage reasoning (core logic) |
| `src/schema/output_schema.py` | Pydantic validation models |
| `src/evaluation/metrics.py` | Quality evaluation (4 metrics) |
| `src/study_book/session_manager.py` | Session persistence |
| `src/logging_config.py` | Production logging setup |
| `examples/demo_usage.py` | CLI example |
| `test_config.py` | Configuration validator |

---

## ✅ Common Tasks

### Try an Example
```bash
streamlit run app.py
# Click "Example Runner" → Select example
```

### Process Your Own Audio
```bash
streamlit run app.py
# Upload audio + notes → Process
```

### Change API Model
Edit `.env`:
```ini
# Fast & Cheap
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.3

# Smart & Expensive
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.2
```

### Enable Debug Logging
```bash
DEBUG=true LOG_LEVEL=DEBUG streamlit run app.py
```

### Generate Study Guide
```bash
streamlit run app.py
# Process sessions → Click "Generate Cumulative Study Guide"
```

---

## 📞 Support

### Check Status
```bash
python test_config.py
```

### View Logs
```bash
tail -f logs/smart_notes_*.log
```

### Test Components
```bash
python -c "from src.reasoning.pipeline import ReasoningPipeline; print('✓ Pipeline OK')"
```

### Reset to Defaults
```bash
rm .env
cp .env.example .env
```

---

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Test config: `python test_config.py`
3. ✅ Run app: `streamlit run app.py`
4. ✅ Try examples: Click "Example Runner"
5. ✅ Upload your content: Use file uploaders

---

## 📊 What You Can Do

- ✅ Extract topics from lectures
- ✅ Identify key concepts
- ✅ Explain equations in plain language
- ✅ Find common misconceptions
- ✅ Generate frequently asked questions
- ✅ Connect to real-world applications
- ✅ Provide worked example solutions
- ✅ Evaluate content quality
- ✅ Build cumulative study guides
- ✅ Manage multiple sessions

---

## 🚀 Quick Reference

```bash
# Setup
pip install -r requirements.txt
python test_config.py

# Run
streamlit run app.py

# View Logs
tail -f logs/smart_notes_*.log

# Test Config
python test_config.py

# CLI Example
python examples/demo_usage.py

# Add API Key
# Edit .env: OPENAI_API_KEY=sk-...
```

---

## 📖 Version Info

- **Created**: 2024
- **Language**: Python 3.8+
- **Framework**: Streamlit
- **LLM**: OpenAI (GPT-4 / GPT-3.5-turbo)
- **Audio**: OpenAI Whisper
- **Status**: ✅ Production Ready

