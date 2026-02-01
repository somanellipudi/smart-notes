# Example Data README

This directory contains sample data for demonstrating the GenAI Smart Study Book research prototype.

## Directory Structure

```
examples/
├── audio/                  # Audio lecture files (or placeholders)
│   ├── lecture1_placeholder.txt
│   └── lecture2_placeholder.txt
├── notes/                  # Handwritten notes text files
│   ├── notes1.txt         # Derivatives introduction
│   └── notes2.txt         # Integration basics
└── inputs/                # Complete input JSON files
    ├── example1.json      # Full example: derivatives
    └── example2.json      # Full example: integration
```

## Example Sessions

### Example 1: Introduction to Derivatives
- **Session ID**: `calculus101_derivatives_intro`
- **Topic**: Derivatives, limit definition, basic examples
- **Notes**: `notes/notes1.txt`
- **Equations**: Derivative definition, power rule
- **Concepts**: Rate of change, tangent lines, velocity

### Example 2: Integration Basics
- **Session ID**: `calculus101_integration_basics`
- **Topic**: Integration, antiderivatives, substitution
- **Notes**: `notes/notes2.txt`
- **Equations**: Integration formulas, power rule for integration
- **Concepts**: Antiderivatives, indefinite integrals, u-substitution

## Using Examples in the UI

### Method 1: Example Runner (Easiest)
1. Launch the Streamlit app: `streamlit run app.py`
2. In the sidebar, find **"Example Runner"**
3. Select an example from the dropdown
4. Click **"Run Example Session"**
5. Click **"Process Example"**

### Method 2: Manual Upload
1. Launch the app
2. Copy content from `notes/notes1.txt` into the notes text area
3. Copy equations from the JSON file
4. Add external context if desired
5. Click **"Generate Structured Study Output"**

## Adding New Examples

To add a new example session:

1. **Create notes file**: `examples/notes/notes_new.txt`
   - Write or paste classroom notes

2. **Create input JSON**: `examples/inputs/example_new.json`
   ```json
   {
     "session_id": "unique_session_id",
     "handwritten_notes": "Content from notes file...",
     "lecture_audio_path": "examples/audio/lecture_new_placeholder.txt",
     "equations": ["equation1", "equation2"],
     "external_context": "Textbook or reference material..."
   }
   ```

3. **Optional: Add audio**
   - Place actual audio file in `examples/audio/`
   - Update `lecture_audio_path` in JSON

4. **Test in UI**
   - Restart Streamlit app
   - New example should appear in dropdown

## Audio Files

Currently, audio files are **placeholders** (`.txt` files). Use real audio files to enable transcription.

To use real audio:
1. Record or obtain lecture audio (WAV, MP3, M4A)
2. Place in `examples/audio/` folder
3. Update JSON `lecture_audio_path` to point to the file
4. Ensure OpenAI API key is set (for Whisper transcription)

## Example Content Attribution

The example content covers basic calculus concepts (derivatives and integration) 
and is designed for educational demonstration purposes in a research context.
