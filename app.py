import streamlit as st
import PyPDF2
import re
import difflib
from openai import OpenAI

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="PolicyExplainer+", layout="wide")

# ------------------ OPENAI CLIENT ------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------ HELPERS ------------------

def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return clean_text(text)

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    return [s.strip() for s in re.split(r"\.\s+", text) if len(s.strip()) > 25]

def get_risk_level(sentence):
    s = sentence.lower()
    if any(w in s for w in ["must", "required", "disciplinary", "penalty", "suspension"]):
        return "🔴 High Importance"
    if any(w in s for w in ["should", "recommended"]):
        return "🟡 Medium Importance"
    return "🟢 Informational"

def simplify_policy(text):
    sentences = split_sentences(text)
    simplified = []

    for s in sentences:
        simplified.append(f"{get_risk_level(s)} — {s}")

    return simplified[:12]

# ------------------ RULE-BASED RETRIEVAL ------------------

def answer_question(question, policy_text):
    question = question.lower().strip()
    sentences = split_sentences(policy_text)

    TIME_WORDS = ["when", "time", "start", "end", "begin"]
    TIME_PATTERN = r"\b\d{1,2}(:\d{2})?\s?(am|pm)?\b"

    scored = []

    for s in sentences:
        s_low = s.lower()
        score = 0
        reasons = []

        for word in question.split():
            if len(word) > 3 and word in s_low:
                score += 1
                reasons.append(f"keyword match: '{word}'")

        if any(w in question for w in TIME_WORDS) and re.search(TIME_PATTERN, s_low):
            score += 6
            reasons.append("contains specific time")

        if "purpose" in s_low or "aim" in s_low:
            score -= 3

        if score > 0:
            scored.append((score, s, reasons))

    if not scored:
        return None, 0, []

    scored.sort(reverse=True, key=lambda x: x[0])
    best = scored[0]

    confidence = min(best[0] / 10, 1.0)
    return best[1], confidence, best[2]

# ------------------ LLM FUNCTIONS ------------------

def llm_explain(sentence, question):
    prompt = f"""
Explain the policy rule below in simple language.

Question:
{question}

Rule:
{sentence}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def llm_answer_full(question, policy_text):
    prompt = f"""
Answer the question using ONLY the policy text below.

Policy:
{policy_text}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def highlight_sentence(text, sentence):
    return text.replace(sentence, f"**🟨 {sentence} 🟨**")

# ------------------ UI ------------------

st.title("📘 PolicyExplainer+")
st.caption("Hybrid AI system with explainable and trustworthy policy answers")

tabs = st.tabs([
    "📂 Upload Policy",
    "🧠 Simplifier",
    "💬 Chatbot (Explainable AI)",
    "🔎 Compare Policies"
])

# ------------------ TAB 1 ------------------
with tabs[0]:
    st.header("Upload or Paste Policy")

    pdf = st.file_uploader("Upload a policy PDF", type=["pdf"])
    pasted = st.text_area("Or paste policy text", height=200)

    if pdf:
        st.session_state["policy_text"] = extract_pdf(pdf)
        st.success("PDF processed successfully")

    elif pasted.strip():
        st.session_state["policy_text"] = clean_text(pasted)
        st.success("Text added successfully")

    if "policy_text" in st.session_state:
        st.text_area("Current Policy", st.session_state["policy_text"], height=250)

# ------------------ TAB 2 ------------------
with tabs[1]:
    st.header("Policy Simplifier with Risk Flags")

    policy = st.session_state.get("policy_text", "")
    if st.button("Simplify Policy"):
        if not policy:
            st.warning("Upload or paste a policy first.")
        else:
            for s in simplify_policy(policy):
                st.write(s)

# ------------------ TAB 3 ------------------
with tabs[2]:
    st.header("Ask the Policy")

    policy = st.session_state.get("policy_text", "")
    if not policy:
        st.info("Upload or paste a policy first.")
    else:
        question = st.text_input("Ask a question about the policy")

        if st.button("Ask"):
            sentence, confidence, reasons = answer_question(question, policy)

            if sentence:
                st.subheader("📌 Relevant Policy Rule")
                st.write(sentence)

                st.write(f"**Risk Level:** {get_risk_level(sentence)}")
                st.write(f"**Confidence:** {confidence:.2f}")

                with st.expander("Why this answer?"):
                    for r in reasons:
                        st.write("✔", r)

                st.subheader("🤖 LLM Explanation")
                st.write(llm_explain(sentence, question))

                st.subheader("📄 Highlighted in Policy")
                st.markdown(highlight_sentence(policy, sentence))
            else:
                st.subheader("🤖 LLM Answer")
                st.write(llm_answer_full(question, policy))

# ------------------ TAB 4 ------------------
with tabs[3]:
    st.header("Compare Two Policies")

    col1, col2 = st.columns(2)
    with col1:
        pdf_a = st.file_uploader("Policy A", type=["pdf"], key="a")
    with col2:
        pdf_b = st.file_uploader("Policy B", type=["pdf"], key="b")

    if st.button("Compare"):
        if not pdf_a or not pdf_b:
            st.warning("Upload both policies.")
        else:
            a = extract_pdf(pdf_a)
            b = extract_pdf(pdf_b)

            s_a = set(split_sentences(a))
            s_b = set(split_sentences(b))

            st.subheader("🟢 Added Rules")
            for s in list(s_b - s_a)[:5]:
                st.write("➕", s)

            st.subheader("🔴 Removed Rules")
            for s in list(s_a - s_b)[:5]:
                st.write("➖", s)
