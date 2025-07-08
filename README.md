# 🧠 EQenhance: Rewrite with Confidence

EQenhance is a gaze-aware, emotionally intelligent rewrite assistant designed to support emotionally loaded workplace communication. Built for the **RAISE Hackathon 2025**, it uses AI and human-centered design to reduce emotional labor and increase clarity, assertiveness, and confidence.

---

## 🚀 Features

- 🔁 **RED Agent (Recognize → Evaluate → Draw)**  
  Reframes your message using LLaMA3 via Groq API, guided by the RED decision model.

- 📊 **GPI (Gaze Pressure Index)**  
  Measures the emotional shift in your rewrite via zero-shot classification (BART) and cosine/PCA-based scoring.

- 📉 **Emotion Score Breakdown**
  - 🧠 Confidence Shift
  - 🔁 Structure Shift
  - 😣 Emotional Load

- 🧘‍♂️ **Wellness Panel**  
  Optional grounding visuals and calming elements for emotionally intense moments.

- 💾 **Score Logging** *(via SQLite)*  
  Logs original message, RED rewrite, final rewrite, and emotional metrics — future-ready for dashboards or HR feedback loops.

- 📡 **Integration-Ready**
  - ⚙️ Mock UI for Microsoft Teams and Slack embedding via adaptive cards
  - 📦 Local `.env` and `requirements.txt` for fast deployment

---

## 💡 Sample Use Case

> “Sorry if this isn’t the right time — I can totally wait. I just thought it might help, but I understand if it’s too much.”

✅ Reframed with RED:
> “Would it be helpful if I shared this now? I believe it could support us, but I’m happy to pause if needed.”

📈 Confidence Shift: +0.38  
📉 Emotional Load Reduced: -0.44

---

## 🧠 Architecture

- 🧬 **NLP Stack:** HuggingFace Transformers (BART), Groq OpenAI-Compatible API (LLaMA3)
- 🧪 **Scoring:** Cosine distance + PCA eigen shift
- 💾 **Data:** Local SQLite logging for messages and GPI breakdowns
- 🧱 **Framework:** Streamlit

---

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone git@github.com:Hattiomkar/EQ-enhance-v2.git
   cd EQ-enhance-v2
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Create a .env file:

ini
Copy
Edit
OPENAI_API_KEY=your-groq-api-key-here
Run the app:

bash
Copy
Edit
streamlit run main.py
📁 Files
File	Description
main.py	Streamlit app with RED agent + GPI system
.env	(not included) Secret key for Groq/OpenAI
.gitignore	Excludes large files, venvs, cache, etc.
requirements.txt	All dependencies
eqenhance_logs.db	SQLite DB (created at runtime)

🎯 Built For
🤝 Emotionally intelligent collaboration

💬 Email, Slack, Teams communication assistants

🧠 Healing emotional masking and performance scripts

🙌 Author
Hatti Omkar
💼 LinkedIn: www.linkedin.com/in/omkar-hatti-0880a816a
🌐 GitHub:https://github.com/Hattiomkar

🏁 Judge Notes
🧩 Emotional processing + productivity enhancement in 1 UI

🧠 Human-centered, trauma-informed design

🔗 Multi-agent pipeline (NLP + GPI + coaching)

📡 Extensible into any chat-based enterprise stack (Teams, Slack, Gmail)
