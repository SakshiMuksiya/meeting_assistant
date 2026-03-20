"""
frontend/app.py — Streamlit live meeting dashboard.

Responsibility:
    Read meeting_notes.json every N seconds and render a live,
    intuitive dashboard showing summary, topics, action items,
    and full transcript. Provide download buttons for all formats.

Run with:
    streamlit run frontend/app.py

Requires the pipeline to be running in a separate terminal:
    python main.py
"""

import json
import csv
import io
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow importing from project root even when run from frontend/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MEETING_NOTES_PATH, settings


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Theme & Custom CSS ─────────────────────────────────────────────────────────
# Soft light theme with warm accents — easy on the eyes, professional.
# Edit the CSS variables at the top to customise colors and fonts.

st.markdown("""
<style>
/* ── CSS Variables — edit these to change the theme ── */
:root {
    --primary:        #4F6CF6;   /* accent color — buttons, badges, highlights */
    --primary-light:  #EEF1FF;   /* light accent — card backgrounds */
    --success:        #22C55E;   /* green — live indicator, action items */
    --success-light:  #F0FDF4;   /* light green background */
    --warning:        #F59E0B;   /* amber — topics */
    --warning-light:  #FFFBEB;   /* light amber background */
    --danger:         #EF4444;   /* red — errors */
    --text-primary:   #1E293B;   /* main text */
    --text-secondary: #64748B;   /* muted text */
    --bg-card:        #FFFFFF;   /* card background */
    --bg-page:        #F8FAFC;   /* page background */
    --border:         #E2E8F0;   /* card borders */
    --radius:         12px;      /* border radius */
    --shadow:         0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Page background ── */
.stApp { background-color: var(--bg-page); }

/* ── Hide default Streamlit elements ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Cards ── */
.ma-card {
    background:    var(--bg-card);
    border:        1px solid var(--border);
    border-radius: var(--radius);
    padding:       1.25rem 1.5rem;
    box-shadow:    var(--shadow);
    margin-bottom: 1rem;
}

/* ── Section headings ── */
.ma-heading {
    font-size:     0.75rem;
    font-weight:   600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color:         var(--text-secondary);
    margin-bottom: 0.5rem;
}

/* ── Live badge ── */
.ma-live {
    display:       inline-flex;
    align-items:   center;
    gap:           6px;
    background:    var(--success-light);
    color:         #16A34A;
    font-size:     0.75rem;
    font-weight:   600;
    padding:       4px 10px;
    border-radius: 99px;
    border:        1px solid #BBF7D0;
}
.ma-live-dot {
    width:            8px;
    height:           8px;
    border-radius:    50%;
    background:       #22C55E;
    animation:        pulse 1.5s ease-in-out infinite;
    display:          inline-block;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1);   }
    50%       { opacity: 0.5; transform: scale(0.85); }
}

/* ── Waiting badge ── */
.ma-waiting {
    display:       inline-flex;
    align-items:   center;
    gap:           6px;
    background:    #FEF3C7;
    color:         #92400E;
    font-size:     0.75rem;
    font-weight:   600;
    padding:       4px 10px;
    border-radius: 99px;
    border:        1px solid #FDE68A;
}

/* ── Topic badge ── */
.ma-topic {
    display:       inline-block;
    background:    var(--warning-light);
    color:         #92400E;
    border:        1px solid #FDE68A;
    border-radius: 99px;
    padding:       3px 10px;
    font-size:     0.8rem;
    font-weight:   500;
    margin:        3px 3px 3px 0;
}

/* ── Summary text ── */
.ma-summary {
    font-size:   1rem;
    line-height: 1.7;
    color:       var(--text-primary);
}

/* ── Stat card ── */
.ma-stat {
    text-align: center;
    padding:    0.75rem;
}
.ma-stat-value {
    font-size:   1.75rem;
    font-weight: 700;
    color:       var(--primary);
    line-height: 1;
}
.ma-stat-label {
    font-size:  0.75rem;
    color:      var(--text-secondary);
    margin-top: 4px;
}

/* ── Action item table ── */
.ma-table {
    width:           100%;
    border-collapse: separate;
    border-spacing:  0;
    font-size:       0.875rem;
}
.ma-table th {
    background:    var(--primary-light);
    color:         var(--primary);
    font-weight:   600;
    font-size:     0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding:       8px 12px;
    text-align:    left;
    border-bottom: 2px solid var(--primary);
}
.ma-table td {
    padding:       10px 12px;
    border-bottom: 1px solid var(--border);
    color:         var(--text-primary);
    vertical-align: top;
}
.ma-table tr:last-child td { border-bottom: none; }
.ma-table tr:hover td { background: var(--bg-page); }

/* ── Person badge inside table ── */
.ma-person {
    display:       inline-block;
    background:    var(--primary-light);
    color:         var(--primary);
    border-radius: 99px;
    padding:       2px 8px;
    font-weight:   600;
    font-size:     0.8rem;
}

/* ── Deadline badge ── */
.ma-deadline {
    display:       inline-block;
    background:    var(--success-light);
    color:         #16A34A;
    border-radius: 99px;
    padding:       2px 8px;
    font-size:     0.78rem;
}
.ma-deadline-none {
    display:       inline-block;
    background:    #F1F5F9;
    color:         var(--text-secondary);
    border-radius: 99px;
    padding:       2px 8px;
    font-size:     0.78rem;
}

/* ── Transcript ── */
.ma-transcript {
    font-family:  monospace;
    font-size:    0.82rem;
    line-height:  1.8;
    color:        var(--text-secondary);
    white-space:  pre-wrap;
    max-height:   320px;
    overflow-y:   auto;
    padding:      0.5rem;
    background:   var(--bg-page);
    border-radius: 8px;
    border:       1px solid var(--border);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}

/* ── Download buttons ── */
.stDownloadButton > button {
    width:         100%;
    border-radius: 8px;
    font-weight:   600;
    font-size:     0.85rem;
    margin-bottom: 0.25rem;
}

/* ── Empty state ── */
.ma-empty {
    text-align:   center;
    padding:      3rem 1rem;
    color:        var(--text-secondary);
}
.ma-empty-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.ma-empty-text { font-size: 1rem; font-weight: 500; }
.ma-empty-sub  { font-size: 0.85rem; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_notes() -> dict | None:
    """Read and parse meeting_notes.json. Returns None if not found."""
    if not MEETING_NOTES_PATH.exists():
        return None
    try:
        with open(MEETING_NOTES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def safe_filename(name: str) -> str:
    """Convert meeting name to a safe filename — replace spaces and specials."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")


def make_timestamp() -> str:
    """Return current time as a filename-safe string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def build_txt(data: dict, meeting_name: str) -> str:
    """Build a human-readable TXT export of all meeting data."""
    lines = []
    sep = "─" * 60

    lines.append("MEETING NOTES")
    lines.append(f"Meeting  : {meeting_name}")
    lines.append(f"Generated: {data.get('last_updated', 'N/A')}")
    lines.append(f"Words    : {data.get('word_count', 0):,}")
    lines.append("")
    lines.append(sep)
    lines.append("SUMMARY")
    lines.append(sep)
    lines.append(data.get("summary", "No summary yet."))
    lines.append("")
    lines.append(sep)
    lines.append("TOPICS")
    lines.append(sep)
    topics = data.get("topics", [])
    if topics:
        for t in topics:
            lines.append(f"  • {t}")
    else:
        lines.append("  No topics identified yet.")
    lines.append("")
    lines.append(sep)
    lines.append("ACTION ITEMS")
    lines.append(sep)
    actions = data.get("action_items", [])
    if actions:
        lines.append(f"  {'Person':<20} {'Task':<40} {'Deadline'}")
        lines.append(f"  {'─'*20} {'─'*40} {'─'*20}")
        for a in actions:
            lines.append(
                f"  {a.get('person',''):<20} "
                f"{a.get('task',''):<40} "
                f"{a.get('deadline','Not specified')}"
            )
    else:
        lines.append("  No action items identified yet.")
    lines.append("")
    lines.append(sep)
    lines.append("FULL TRANSCRIPT")
    lines.append(sep)
    lines.append(data.get("full_transcript", "No transcript yet."))
    lines.append("")

    return "\n".join(lines)


def build_csv(data: dict, meeting_name: str) -> str:
    """Build a CSV export — action items as primary table, metadata as headers."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Metadata rows
    writer.writerow(["Meeting Name", meeting_name])
    writer.writerow(["Generated",   data.get("last_updated", "N/A")])
    writer.writerow(["Word Count",  data.get("word_count", 0)])
    writer.writerow([])

    # Summary
    writer.writerow(["SUMMARY"])
    writer.writerow([data.get("summary", "")])
    writer.writerow([])

    # Topics
    writer.writerow(["TOPICS"])
    for t in data.get("topics", []):
        writer.writerow([t])
    writer.writerow([])

    # Action items table
    writer.writerow(["ACTION ITEMS"])
    writer.writerow(["Person", "Task", "Deadline"])
    for a in data.get("action_items", []):
        writer.writerow([
            a.get("person",   "Unassigned"),
            a.get("task",     ""),
            a.get("deadline", "Not specified"),
        ])
    writer.writerow([])

    # Full transcript
    writer.writerow(["FULL TRANSCRIPT"])
    writer.writerow([data.get("full_transcript", "")])

    return output.getvalue()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎙️ Meeting Assistant")
    st.markdown("---")

    # Meeting name input
    st.markdown("**Meeting Name**")
    meeting_name = st.text_input(
        label="meeting_name_input",
        label_visibility="collapsed",
        placeholder="e.g. Q3 Planning Session",
        value="Meeting",
    )

    st.markdown("---")

    # Auto-refresh toggle
    st.markdown("**Auto Refresh**")
    auto_refresh = st.toggle("Refresh every 3 seconds", value=True)

    st.markdown("---")

    # Download section
    st.markdown("**Export Notes**")

    data = load_notes()

    if data:
        ts   = make_timestamp()
        fn   = safe_filename(meeting_name)
        base = f"{fn}_{ts}"

        # TXT download
        txt_content = build_txt(data, meeting_name)
        st.download_button(
            label="📄 Download as TXT",
            data=txt_content,
            file_name=f"{base}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # JSON download
        json_content = json.dumps({**data, "meeting_name": meeting_name}, indent=2, ensure_ascii=False)
        st.download_button(
            label="🗂️ Download as JSON",
            data=json_content,
            file_name=f"{base}.json",
            mime="application/json",
            use_container_width=True,
        )

        # CSV download
        csv_content = build_csv(data, meeting_name)
        st.download_button(
            label="📊 Download as CSV",
            data=csv_content,
            file_name=f"{base}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.info("Start the pipeline to enable exports.", icon="ℹ️")

    st.markdown("---")
    st.markdown(
        "<small style='color:#94A3B8'>Auto-saved to<br>"
        f"<code>output/meeting_notes.json</code></small>",
        unsafe_allow_html=True,
    )


# ── Main dashboard ─────────────────────────────────────────────────────────────

# Header row
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        f"<h1 style='margin:0; font-size:1.75rem; color:#1E293B;'>"
        f"🎙️ {meeting_name}</h1>",
        unsafe_allow_html=True,
    )
with col_status:
    if data:
        st.markdown(
            "<div style='padding-top:0.6rem; text-align:right'>"
            "<span class='ma-live'>"
            "<span class='ma-live-dot'></span>LIVE"
            "</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='padding-top:0.6rem; text-align:right'>"
            "<span class='ma-waiting'>⏳ Waiting for pipeline</span>"
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# ── Empty state ────────────────────────────────────────────────────────────────
if not data:
    st.markdown("""
    <div class='ma-card'>
        <div class='ma-empty'>
            <div class='ma-empty-icon'>🎤</div>
            <div class='ma-empty-text'>Waiting for the meeting to start</div>
            <div class='ma-empty-sub'>Run <code>python main.py</code> in your terminal to begin</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Stats row ──────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)

    word_count   = data.get("word_count", 0)
    topic_count  = len(data.get("topics", []))
    action_count = len(data.get("action_items", []))
    last_updated = data.get("last_updated", "—")

    with s1:
        st.markdown(f"""
        <div class='ma-card ma-stat'>
            <div class='ma-stat-value'>{word_count:,}</div>
            <div class='ma-stat-label'>Words spoken</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class='ma-card ma-stat'>
            <div class='ma-stat-value'>{topic_count}</div>
            <div class='ma-stat-label'>Topics identified</div>
        </div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class='ma-card ma-stat'>
            <div class='ma-stat-value'>{action_count}</div>
            <div class='ma-stat-label'>Action items</div>
        </div>""", unsafe_allow_html=True)

    with s4:
        st.markdown(f"""
        <div class='ma-card ma-stat'>
            <div class='ma-stat-value' style='font-size:1rem; padding-top:0.35rem'>{last_updated}</div>
            <div class='ma-stat-label'>Last updated</div>
        </div>""", unsafe_allow_html=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = data.get("summary", "").strip()
    st.markdown("<div class='ma-heading'>📝 Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='ma-card'>
        <div class='ma-summary'>
            {summary if summary else "<span style='color:#94A3B8'>Summary will appear here once enough speech is detected...</span>"}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Topics + Action Items ──────────────────────────────────────────────────
    col_topics, col_actions = st.columns([1, 2])

    with col_topics:
        topics = data.get("topics", [])
        st.markdown("<div class='ma-heading'>🏷️ Topics</div>", unsafe_allow_html=True)

        if topics:
            badges = " ".join(f"<span class='ma-topic'>{t}</span>" for t in topics)
            st.markdown(
                f"<div class='ma-card'>{badges}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='ma-card'>"
                "<span style='color:#94A3B8; font-size:0.875rem'>"
                "Topics will appear as discussion progresses..."
                "</span></div>",
                unsafe_allow_html=True,
            )

    with col_actions:
        actions = data.get("action_items", [])
        st.markdown("<div class='ma-heading'>✅ Action Items</div>", unsafe_allow_html=True)

        if actions:
            rows = ""
            for a in actions:
                person   = a.get("person",   "Unassigned")
                task     = a.get("task",     "")
                deadline = a.get("deadline", "Not specified")

                deadline_html = (
                    f"<span class='ma-deadline'>{deadline}</span>"
                    if deadline.lower() != "not specified"
                    else f"<span class='ma-deadline-none'>{deadline}</span>"
                )

                rows += f"""
                <tr>
                    <td><span class='ma-person'>{person}</span></td>
                    <td>{task}</td>
                    <td>{deadline_html}</td>
                </tr>"""

            st.markdown(f"""
            <div class='ma-card' style='padding: 0; overflow:hidden'>
                <table class='ma-table'>
                    <thead>
                        <tr>
                            <th>Person</th>
                            <th>Task</th>
                            <th>Deadline</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='ma-card'>"
                "<span style='color:#94A3B8; font-size:0.875rem'>"
                "Action items will appear when tasks are assigned..."
                "</span></div>",
                unsafe_allow_html=True,
            )

    # ── Full transcript ────────────────────────────────────────────────────────
    transcript = data.get("full_transcript", "").strip()
    with st.expander("📄 Full Transcript", expanded=False):
        if transcript:
            st.markdown(
                f"<div class='ma-transcript'>{transcript}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#94A3B8'>Transcript will appear here...</span>",
                unsafe_allow_html=True,
            )

# ── Auto refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(settings.pipeline.frontend_poll_interval_seconds)
    st.rerun()
