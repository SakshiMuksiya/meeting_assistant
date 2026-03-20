# 🎙️ Context-Aware Agentic AI Meeting Assistant

A real-time AI meeting assistant that captures live audio, transcribes speech using OpenAI Whisper, and runs multiple specialized AI agents to continuously generate summaries, extract discussion topics, and detect action items — all displayed on a live local dashboard.

---

## ✨ Features

- 🎤 **Live audio capture** — records from your microphone in 7-second chunks
- 📝 **Real-time transcription** — powered by OpenAI Whisper running locally (no API cost)
- 🧠 **Context-aware memory** — sliding window keeps agents focused on recent discussion
- 🤖 **Three specialized agents** running every ~14 seconds:
  - **Summary Agent** — incrementally updates a concise meeting summary
  - **Topic Agent** — extracts key discussion themes as short phrases
  - **Action Item Agent** — detects tasks with person, task, and deadline
- 📊 **Live dashboard** — Streamlit UI that updates every 3 seconds
- 💾 **Auto-save** — meeting notes written to disk continuously
- 📥 **Export** — download as TXT, JSON, or CSV at any time
- 🆓 **Free LLM** — uses Google Gemini 2.0 Flash via AI Studio (no credit card needed)

---

## 🏗️ Architecture

```
Microphone
    │
    ▼
audio/recorder.py          Captures PCM audio chunks (7s each)
    │
    ▼
transcription/whisper_engine.py    Converts audio → text (local, offline)
    │
    ▼
memory/context_store.py    Maintains full transcript + sliding context window
    │
    ├──▶ agents/summary_agent.py    → str   (incremental summary)
    ├──▶ agents/topic_agent.py      → list  (discussion topics)
    └──▶ agents/action_agent.py     → list  (action items with person/task/deadline)
              │
              ▼
         llm/llm_client.py     Gemini 2.0 Flash via OpenAI-compatible endpoint
              │
              ▼
    storage/writer.py          Atomic JSON write → output/meeting_notes.json
              │
              ▼
    frontend/app.py            Streamlit dashboard (polls every 3s)
```

---

## 📁 Project Structure

```
meeting_assistant/
├── audio/
│   └── recorder.py              # Microphone capture with chunk queue
├── transcription/
│   └── whisper_engine.py        # Whisper model wrapper + silence filter
├── memory/
│   └── context_store.py         # Full transcript + sliding context window
├── agents/
│   ├── base_agent.py            # Abstract base class all agents inherit
│   ├── summary_agent.py         # Incremental meeting summarization
│   ├── topic_agent.py           # Discussion topic extraction
│   └── action_agent.py          # Structured action item detection
├── llm/
│   └── llm_client.py            # Gemini API wrapper with retry logic
├── pipeline/
│   └── runner.py                # Main orchestration loop
├── storage/
│   └── writer.py                # Atomic JSON persistence
├── frontend/
│   └── app.py                   # Streamlit live dashboard
├── output/                      # Auto-created — meeting notes saved here
├── config.py                    # All settings in one place
├── main.py                      # Entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/meeting-assistant.git
cd meeting-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Linux you may need `sudo apt install libportaudio2` for microphone support.
> Whisper will download the `base` model (~140 MB) automatically on first run.

### 3. Get a free Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **Get API key** → **Create API key**
4. Copy the key — no credit card required

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GEMINI_API_KEY=AIza-your-key-here
```

### 5. Run the pipeline

```bash
python main.py
```

### 6. Open the dashboard (separate terminal)

```bash
streamlit run frontend/app.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## 🖥️ Dashboard

The Streamlit dashboard updates every 3 seconds and displays:

| Section | Description |
|---|---|
| **Stats bar** | Words spoken, topics found, action items, last updated |
| **Summary** | Auto-updating meeting summary |
| **Topics** | Current discussion themes as color-coded badges |
| **Action Items** | Table with person, task, and deadline columns |
| **Transcript** | Full meeting transcript in a collapsible section |

### Saving your notes

1. Type the meeting name in the sidebar (e.g. `Q3 Planning Session`)
2. Click a download button:
   - 📄 **TXT** — human-readable formatted notes
   - 🗂️ **JSON** — structured data for programmatic use
   - 📊 **CSV** — action items table for sharing with the team

Files are named automatically: `Q3_Planning_Session_2024-10-15_14-32.txt`

---

## ⚙️ Configuration

All settings live in `config.py`. Common things to change:

| Setting | Location | Default | Description |
|---|---|---|---|
| Whisper model | `WhisperConfig.model_size` | `"base"` | Use `"small"` for better accuracy on GPU |
| Chunk duration | `AudioConfig.chunk_duration_seconds` | `7.0` | Seconds per audio chunk |
| Context window | `MemoryConfig.max_context_chars` | `3000` | Characters sent to agents |
| Agent interval | `PipelineConfig.agent_run_interval` | `2` | Run agents every N chunks |
| LLM model | `LLMConfig.summary_model` | `"gemini-2.0-flash"` | Gemini model to use |

### Whisper model sizes

| Model | Size | Speed | Accuracy | Use when |
|---|---|---|---|---|
| `tiny` | 75 MB | Fastest | Low | Testing only |
| `base` | 145 MB | Fast | Good | **Default — CPU real-time** |
| `small` | 465 MB | Medium | Better | GPU available |
| `medium` | 1.5 GB | Slow | High | GPU, accuracy matters |
| `large` | 3 GB | Slowest | Best | GPU, best results |

---

## 🎨 Customising the Dashboard Theme

Open `frontend/app.py` and edit the CSS variables at the top of the `<style>` block:

```css
:root {
    --primary:        #4F6CF6;   /* accent color — buttons, highlights */
    --primary-light:  #EEF1FF;   /* light accent backgrounds */
    --success:        #22C55E;   /* green — live indicator */
    --warning:        #F59E0B;   /* amber — topic badges */
    --text-primary:   #1E293B;   /* main text */
    --bg-card:        #FFFFFF;   /* card background */
    --bg-page:        #F8FAFC;   /* page background */
}
```

For a dark theme, also create `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#4F6CF6"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F1F5F9"
```

---

## 🧩 Optional — Vector Store (Semantic Memory)

Enable semantic retrieval of past meeting segments — useful for very long meetings where relevant context may have scrolled out of the sliding window.

1. Install the extra dependencies:
```bash
pip install sentence-transformers faiss-cpu
```

2. Enable in `config.py`:
```python
enable_vector_store: bool = True
```

Agents can now retrieve semantically relevant passages from earlier in the meeting, not just the most recent context.

---

## 📋 Requirements

- Python 3.10 or higher
- Microphone connected and accessible
- Internet connection (for Gemini API calls only — Whisper runs offline)
- ~500 MB disk space (for Whisper model cache)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition model
- [Google Gemini](https://aistudio.google.com) — LLM backend
- [Streamlit](https://streamlit.io) — dashboard framework
- [sounddevice](https://python-sounddevice.readthedocs.io) — audio capture
