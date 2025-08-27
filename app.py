import os
import io
from datetime import datetime, timedelta
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# LLM (Gemini) via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------------
# Setup: Secrets & LLM
# -------------------------
def get_llm():
    """
    Loads GOOGLE_API_KEY from .env or Streamlit secrets and returns a Gemini LLM.
    """
    load_dotenv(override=False)
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    if not api_key:
        st.stop()  # Stop rendering and show a clear error
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )


llm = get_llm()

st.set_page_config(page_title="Study bot", page_icon="🧭", layout="wide")
st.title("🧭 Study bot")
st.caption("AI tools for resumes, interviews, and daily productivity — powered by Gemini.")


# =========================
# Helpers
# =========================
def safe_split_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def to_docx_bytes(plain_text: str) -> bytes:
    """
    Creates a simple .docx file from plain text.
    (Keeps it dependency-light: uses python-docx if present, else falls back to .txt bytes)
    """
    try:
        from docx import Document
        doc = Document()
        for line in plain_text.splitlines():
            doc.add_paragraph(line)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        # Fallback: return text bytes (user can still download as .txt)
        return plain_text.encode("utf-8")


def parse_task_line(line: str) -> Tuple[str, int, str, str]:
    """
    Parse a task line like:
        Task name, 60, H, 15:30
    -> (name, duration_minutes, priority, deadlineHHMM or "")
    Priority: H/M/L
    Deadline is optional ('' if missing)
    """
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if parts else "Task"
    duration = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 60
    priority = parts[2].upper() if len(parts) > 2 else "M"
    deadline = parts[3] if len(parts) > 3 else ""
    return name, duration, priority, deadline


def schedule_tasks(
    tasks: List[Tuple[str, int, str, str]],
    work_start: str,
    work_end: str
):
    """
    Very simple scheduler:
    - Sort by (deadline present, deadline time), then priority H > M > L
    - Greedy place tasks in the day from start time until end
    """
    pri_order = {"H": 0, "M": 1, "L": 2}

    def deadline_key(dl: str):
        if not dl:
            return (1, "23:59")
        return (0, dl)

    tasks_sorted = sorted(
        tasks,
        key=lambda t: (deadline_key(t[3]), pri_order.get(t[2], 1))
    )

    today = datetime.now().date()
    start_dt = datetime.combine(today, datetime.strptime(work_start, "%H:%M").time())
    end_dt = datetime.combine(today, datetime.strptime(work_end, "%H:%M").time())
    current = start_dt

    schedule = []
    for name, dur, pri, dl in tasks_sorted:
        task_end = current + timedelta(minutes=dur)
        if task_end <= end_dt:
            schedule.append({
                "Task": name,
                "Priority": pri,
                "Start": current.strftime("%H:%M"),
                "End": task_end.strftime("%H:%M"),
                "Deadline": dl or "—"
            })
            current = task_end
        else:
            schedule.append({
                "Task": f"{name} (Overflow)",
                "Priority": pri,
                "Start": "—",
                "End": "—",
                "Deadline": dl or "—"
            })
    return schedule


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📄 Resume Builder", "🗣️ Interview Q&A", "🗓️ Daily Planner"])


# -------------------------
# 1) Resume Builder
# -------------------------
with tab1:
    st.subheader("📄 AI Resume Builder")

    colA, colB = st.columns(2)
    with colA:
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        location = st.text_input("Location (City, Country)")
        role = st.text_input("Target Role / Title", placeholder="e.g., Data Scientist")
        years_exp = st.number_input("Years of Experience", min_value=0.0, step=0.5, value=1.0)

    with colB:
        summary_pref = st.selectbox("Style", ["Concise", "Detailed", "Impact-focused"])
        skills = st.text_area("Skills (comma-separated)", placeholder="Python, SQL, Machine Learning, NLP")
        tools = st.text_area("Tools/Tech (comma-separated)", placeholder="TensorFlow, PyTorch, Pandas, Docker")
        industries = st.text_input("Industry focus (optional)", placeholder="FinTech, Healthcare")

    st.markdown("**Experience (optional)** — bullet points (one per line)")
    exp = st.text_area("Experience points", height=120, placeholder="Led X to achieve Y...\nImproved Z by 30%...")
    st.markdown("**Education (optional)** — bullet points (one per line)")
    edu = st.text_area("Education points", height=100, placeholder="B.Tech in CSE — XYZ University (2022)")

    if st.button("Generate Resume"):
        skill_list = safe_split_list(skills)
        tool_list = safe_split_list(tools)

        prompt = f"""
You are an expert resume writer. Create an ATS-friendly resume draft in clean Markdown.
Use crisp, achievement-oriented bullets with metrics where possible.
Style: {summary_pref}. Target role: {role}. Years of experience: {years_exp}.

Candidate:
- Name: {name}
- Email: {email}
- Phone: {phone}
- Location: {location}
- Skills: {", ".join(skill_list)}
- Tools/Tech: {", ".join(tool_list)}
- Industry focus: {industries or "General"}

Experience points (raw, optional):
{exp or "N/A"}

Education points (raw, optional):
{edu or "N/A"}

Output sections in this order:
1) Name & Contact (single header line)
2) Professional Summary (3–5 lines)
3) Skills & Tools (grouped)
4) Experience (bullet points)
5) Projects (add 1–2 plausible project bullets if none provided)
6) Education
7) Certifications (add suggests if none)

Keep it scannable. Avoid tables. Use bold for company/role. Limit to ~1 page.
"""
        result = llm.invoke(prompt)
        md = result.content.strip()

        st.markdown("### ✅ Resume Draft")
        st.markdown(md)

        # Downloads
        st.download_button("⬇️ Download as Markdown (.md)", md.encode("utf-8"),
                           file_name="resume_draft.md", mime="text/markdown")
        docx_bytes = to_docx_bytes(md)
        st.download_button("⬇️ Download as Word (.docx)", docx_bytes,
                           file_name="resume_draft.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# -------------------------
# 2) Interview Question Generator
# -------------------------
with tab2:
    st.subheader("🗣️ Interview Question Generator")

    role_type = st.selectbox("Role", ["Data Scientist", "Backend Engineer", "ML Engineer", "Product Manager", "DevOps"])
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    jd = st.text_area("Paste the Job Description (JD)", height=220)

    if st.button("Generate Questions & Model Answers"):
        prompt = f"""
You are a hiring manager preparing an interview.
Role: {role_type}
Difficulty: {difficulty}

Job Description:
{jd}

Generate:
- ~8 technical questions with model answers
- ~5 behavioral questions with STAR-structured model answers
- A short role-specific cheat sheet (bullet list of topics to revise)

Format clearly in Markdown with headings and numbered lists.
Keep answers concise but high quality.
"""
        result = llm.invoke(prompt)
        st.markdown("### ✅ Interview Pack")
        st.markdown(result.content)

        st.download_button("⬇️ Download Q&A (.md)",
                           result.content.encode("utf-8"),
                           file_name="interview_qa.md",
                           mime="text/markdown")


# -------------------------
# 3) Daily To-Do Planner
# -------------------------
with tab3:
    st.subheader("🗓️ AI Daily To-Do Planner")

    st.write("**How to enter tasks:** one per line, format → `Task name, duration_minutes, priority(H/M/L), deadline(HH:MM optional)`")
    st.caption("Example:  Code feature X, 90, H, 13:00")

    tasks_raw = st.text_area("Tasks", height=160, placeholder="Deep work on project, 120, H, 12:00\nTeam meeting, 30, M, 10:30\nGym, 60, L")
    col1, col2 = st.columns(2)
    with col1:
        work_start = st.text_input("Work start (24h)", value="09:00")
    with col2:
        work_end = st.text_input("Work end (24h)", value="18:00")

    if st.button("Generate Schedule"):
        lines = [l for l in tasks_raw.splitlines() if l.strip()]
        tasks = [parse_task_line(l) for l in lines] if lines else []
        if not tasks:
            st.warning("Please enter at least one task.")
        else:
            schedule = schedule_tasks(tasks, work_start, work_end)
            st.markdown("### ✅ Suggested Schedule")
            st.dataframe(schedule, use_container_width=True)

            # Also ask Gemini for short productivity tips based on tasks
            task_list_str = "\n".join([f"- {t[0]} ({t[1]}m, {t[2]}, deadline {t[3] or '—'})" for t in tasks])
            tip_prompt = f"""
You are a productivity coach. Given these tasks and schedule window {work_start}-{work_end},
suggest 5 short tips (bullet points) to improve focus, batching, and breaks.

Tasks:
{task_list_str}
"""
            tips = llm.invoke(tip_prompt).content
            st.markdown("#### 💡 Tips to Work Smarter")
            st.markdown(tips)
