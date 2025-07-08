
import sqlite3
from datetime import datetime
import os
import openai
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from transformers import pipeline
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import hmean, zscore
from sklearn.decomposition import PCA
st.sidebar.markdown("🏁 **Built for RAISE Hackathon 2025**")

st.sidebar.markdown("**Please give it 3-5 minutes for completely loading - VULTR GPU was denied**")


st.sidebar.markdown("""
1. Review the emotionally loaded message on the left.
2. Click 🔁 to get a RED-framed suggestion.
3. Optionally edit the suggestion and view GPI metrics.
4. Scores + rewrite are saved to local DB and downloadable.
""")

# Load secrets
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

st.set_page_config(page_title="EQenhance", layout="wide")

# -- Setup SQLite database
conn = sqlite3.connect("eqenhance_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_text TEXT,
        red_output TEXT,
        final_rewrite TEXT,
        confidence REAL,
        structure REAL,
        emotion REAL,
        gpi_score REAL
    )
""")
conn.commit()

# Classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
coping_labels = [
    "hedging", "self-deprecation", "compliance", "hopelessness", "numbness",
    "appeasing language", "resignation", "cynicism", "over-apologizing",
    "overcompensation", "minimizing", "deflecting", "vagueness", "rigid speech",
    "passive voice", "boasting", "negative self-talk", "confusion", "disengagement",
    "dismissiveness", "dismissal", "isolation narrative", "intellectualizing",
    "denial", "silencing self", "generalizing", "emotional fatigue", "hiding identity"
]

def compute_vector(text):
    result = classifier(text, candidate_labels=coping_labels, multi_label=True)
    return np.array(result["scores"])



def gpi_diff(vec_a, vec_b):
    z_a, z_b = zscore(vec_a), zscore(vec_b)
    cos_dist = cosine(z_a, z_b)

    pca = PCA(n_components=2)  # request 2 components
    pca.fit(np.vstack([vec_a, vec_b]))
    explained = pca.explained_variance_

    # Make sure we have at least 2 components
    if len(explained) >= 2:
        eig_shift = abs(explained[0] - explained[1]) / np.sqrt(len(vec_a))
    else:
        eig_shift = 0  # fallback if only 1 component

    eucl = euclidean(vec_a, vec_b)
    gpi_hm = hmean([cos_dist, eig_shift]) if cos_dist > 0 and eig_shift > 0 else 0
    return cos_dist, eig_shift, eucl, gpi_hm + eucl

def run_red_agent(user_text):
    red_prompt = f"""
You are a RED Coach (Recognize, Evaluate, Draw) for emotionally intelligent workplace communication.

Original message:
\"\"\"{user_text}\"\"\"

Walk through the RED model:
1. 🧠 Recognize Assumptions — what assumptions does the speaker seem to be making?
2. 📊 Evaluate Arguments — what's missing, what’s strong, any blind spots?
3. 🧭 Draw Conclusions — what's the emotionally intelligent rewrite?

Respond with a confident, values-aligned rewrite below.
"""
    try:
        response = openai.ChatCompletion.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": red_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from RED Agent: {str(e)}"

# UI
st.title("🧠 EQenhance: Rewrite with Confidence")

with st.expander("ℹ️ About"):
    st.markdown("""
        EQenhance helps you reframe emotionally-loaded workplace communication using:
        - GPI (Gaze Pressure Index) for confidence analysis
        - RED (Recognize, Evaluate, Draw) for decision clarity
        - GIVE / DEARMAN for interpersonal balance
    """)

# Default test message
default_msg = "Sorry if this isn’t the right time — I can totally wait if needed. I just thought it might help, but I understand if it’s too much."

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔎 Original Message")
    input_text = st.text_area("Paste the message you want to reframe:", value=default_msg, height=200)

with col2:
    st.subheader("🧠 RED Agent Suggestions")

    if "red_output" not in st.session_state:
        st.session_state.red_output = ""

    if st.button("🔁 Generate with RED Coach"):
        st.session_state.red_output = run_red_agent(input_text)

    if st.session_state.red_output:
        st.markdown("#### ✍️ Suggested Rewrite")
        st.success(st.session_state.red_output)

# Rewriting + GPI
if input_text:
    original_vec = compute_vector(input_text)

    st.subheader("✨ Rewrite Using GIVE / FAST / DEARMAN")

    rewritten_text = st.text_area(
        "Rewrite your message below:",
        value=st.session_state.red_output,
        height=200
    )


    if rewritten_text:
        rewritten_vec = compute_vector(rewritten_text)
        cos, eig, eucl, gpi = gpi_diff(original_vec, rewritten_vec)

        st.success("✅ Rewrite Analysis Complete")
        col3, col4, col5 = st.columns(3)
        col3.metric("Confidence Shift", f"{cos:.4f}")
        col4.metric("Structure Shift", f"{eig:.4f}")
        col5.metric("Emotional Load", f"{eucl:.4f}")
        st.metric("📈 Final GPI-Diff Score", f"{gpi:.4f}")

	# -- Emotional Feedback Summary
    if gpi < 0.15:
        st.success("✅ Rewrite is confident, clear, and emotionally regulated.")
    elif gpi < 0.4:
        st.info("⚠️ Rewrite is decent but might carry subtle hesitation or passivity.")
    else:
        st.warning("🚨 Rewrite still shows signs of stress or masking.")




    # -- Log to DB
        cursor.execute("""
            INSERT INTO logs (
                timestamp, input_text, red_output, final_rewrite,
                confidence, structure, emotion, gpi_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            input_text,
            st.session_state.red_output,
            rewritten_text,
            cos,
            eig,
            eucl,
            gpi
        ))
        conn.commit()

# -- Calming Visuals (replaces duck)
with st.expander("🧘‍♂️ Pause and Regroup"):
    st.image("https://media.tenor.com/I6kN-6X7nhAAAAAi/loading-thinking.gif", caption="Take a breath — you're safe to reflect.")

# -- Teams Integration Mock Section
with st.expander("📡 Microsoft Teams Integration (Planned)"):
    st.markdown("""
    EQenhance is designed to embed in Microsoft Teams, Slack, or Gmail using adaptive cards. Future versions will detect emotional friction in real-time and offer rewrite suggestions inside chat tools.
    """)
    st.image("https://learn.microsoft.com/en-us/graph/images/teams-adaptive-card.png", caption="Mock of future embedded UI")

# -- Judging Panel
with st.expander("📣 Judges: Why EQenhance Matters"):
    st.markdown("""
    - 🚀 Built for emotionally intelligent workplace collaboration
    - 🔁 Multi-agent design: RED coaching + GPI analytics
    - 💾 Gaze Pressure Index stored in SQLite for downstream HR use
    - 📡 Integration-ready for Teams, Slack, and Gmail
    """)

# -- Demo Instructions
with st.expander("📽️ Demo Flow"):
    st.markdown("""
    1. Review the emotionally loaded message on the left.
    2. Click 🔁 to get a RED-framed suggestion.
    3. Optionally edit the suggestion and view GPI metrics.
    4. Scores + rewrite are saved to local DB and downloadable.
    """)


st.markdown("---")
with st.expander("📁 Wellness Tools (Mocked)"):
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/57/Mountain_Range_Scenery.jpg", caption="🦆 Duck says breathe deeply.")

with st.expander("📈 My Growth (Mocked)"):
    st.metric("Gaze Confidence", "+27%")
    st.write("Final Score based on lowered emotional effort + improved structure.")
