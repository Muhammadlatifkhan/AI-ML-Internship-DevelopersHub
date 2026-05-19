"""
Task 11: Auto Tagging Support Tickets Using LLM
Zero-shot vs Few-shot learning for ticket categorization
"""

import streamlit as st
from groq import Groq
import os
import json

# ========== API KEY LOADING ==========
GROQ_API_KEY = None

# Try to load from .env file
try:
    with open('.env', 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key.strip() == 'GROQ_API_KEY':
                    GROQ_API_KEY = value.strip().strip('"').strip("'")
                    break
except:
    pass

if not GROQ_API_KEY:
    st.error("""
    ❌ GROQ_API_KEY not found!
    Please create a `.env` file with: GROQ_API_KEY=your_key_here
    """)
    st.stop()

# Page config
st.set_page_config(
    page_title="Support Ticket Tagger",
    page_icon="🏷️",
    layout="wide"
)

# Title
st.title("🏷️ Support Ticket Auto-Tagger")
st.markdown("Automatically categorize support tickets using **Groq Llama 3.3 70B**")

# Categories
categories = {
    "network_issues": "Internet connectivity, Wi-Fi problems, network outages",
    "billing": "Payment issues, refunds, incorrect charges, invoices",
    "account_access": "Login problems, password reset, account locked",
    "technical_bug": "App crashes, error messages, feature not working",
    "subscription": "Plan changes, cancellation, upgrades/downgrades",
    "performance": "Slow loading, lag, response time issues",
    "api_support": "API integration, webhooks, developer tools"
}

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("📋 Available Tags")
    for cat, desc in categories.items():
        st.markdown(f"**{cat}**: {desc}")
    
    st.divider()
    
    approach = st.radio(
        "Select Approach",
        ["Zero-Shot", "Few-Shot (3 examples)", "Few-Shot (5 examples)"],
        help="Zero-shot: No examples. Few-shot: Learn from examples"
    )
    
    st.divider()
    
    # Show API status
    masked = GROQ_API_KEY[:8] + "..." + GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 12 else "✓"
    st.success(f"✅ Groq API: Connected")
    st.caption(f"Key: {masked}")

# Initialize Groq client
@st.cache_resource
def init_groq():
    return Groq(api_key=GROQ_API_KEY)

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {"ticket": "My internet keeps disconnecting every 5 minutes. I've restarted my router.", "tag": "network_issues"},
    {"ticket": "I was charged twice for my subscription. Please refund.", "tag": "billing"},
    {"ticket": "Can't log in. It says invalid credentials even though password is correct.", "tag": "account_access"},
    {"ticket": "The app crashes when I try to upload a photo.", "tag": "technical_bug"},
    {"ticket": "How do I upgrade from Basic to Premium plan?", "tag": "subscription"}
]

def get_zero_shot_prompt(ticket_text, categories_dict):
    category_list = "\n".join([f"- {cat}: {desc}" for cat, desc in categories_dict.items()])
    return f"""You are a support ticket classifier. Classify the following ticket into ONE of these categories:

{category_list}

Ticket: "{ticket_text}"

Respond with ONLY the category name (e.g., "billing" or "network_issues"). Do not add any explanation.

Category:"""

def get_few_shot_prompt(ticket_text, categories_dict, num_examples=3):
    category_list = "\n".join([f"- {cat}: {desc}" for cat, desc in categories_dict.items()])
    examples = FEW_SHOT_EXAMPLES[:num_examples]
    examples_text = ""
    for ex in examples:
        examples_text += f'Ticket: "{ex["ticket"]}"\nCategory: {ex["tag"]}\n\n'
    return f"""You are a support ticket classifier. Here are examples:

{examples_text}
Categories:
{category_list}

Now classify this ticket:
Ticket: "{ticket_text}"

Respond with ONLY the category name.

Category:"""

def get_top_3_prompt(ticket_text, categories_dict):
    category_list = "\n".join([f"- {cat}: {desc}" for cat, desc in categories_dict.items()])
    return f"""Classify this ticket and return TOP 3 categories with confidence scores.

Categories:
{category_list}

Ticket: "{ticket_text}"

Respond ONLY valid JSON format:
{{"predictions": [
    {{"category": "category_name", "confidence": 0.95}},
    {{"category": "category_name", "confidence": 0.85}},
    {{"category": "category_name", "confidence": 0.70}}
]}}"""

def predict_tag(client, ticket_text, approach, categories_dict):
    if approach == "Zero-Shot":
        prompt = get_zero_shot_prompt(ticket_text, categories_dict)
    elif "3 examples" in approach:
        prompt = get_few_shot_prompt(ticket_text, categories_dict, 3)
    else:
        prompt = get_few_shot_prompt(ticket_text, categories_dict, 5)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def predict_top_3(client, ticket_text, categories_dict):
    prompt = get_top_3_prompt(ticket_text, categories_dict)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            predictions = json.loads(json_match.group())
            return predictions.get("predictions", [])
        return []
    except Exception as e:
        return []

# ========== MAIN INTERFACE ==========

# Initialize session state for ticket text
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Support Ticket")
    
    # Text area bound to session state
    ticket_input = st.text_area(
        "Enter or paste support ticket description:",
        height=150,
        value=st.session_state.ticket_text,
        placeholder="Example: My internet keeps disconnecting every 5 minutes. I've restarted my router twice but still having issues...",
        key="ticket_input"
    )
    
    # Update session state when text changes
    if ticket_input != st.session_state.ticket_text:
        st.session_state.ticket_text = ticket_input
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        predict_btn = st.button("🔮 Predict Tag", type="primary", use_container_width=True)
    with col_b:
        top3_btn = st.button("📊 Show Top 3 Tags", use_container_width=True)
    with col_c:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    result_container = st.container()

# Sample tickets section
with st.expander("📋 Sample Tickets (Click to try)"):
    sample_tickets = [
        "My internet keeps disconnecting every 5 minutes. I've restarted my router twice.",
        "I was charged twice for my monthly subscription. Please refund the extra charge.",
        "The login page shows 'invalid credentials' even though I'm using the correct password.",
        "The app crashes when I try to upload a profile picture.",
        "How do I upgrade my plan from Basic to Premium?",
        "Your service is too slow. Loading times are over 10 seconds.",
        "I didn't receive the welcome email after signing up.",
        "The payment gateway shows error code 404.",
        "My data isn't syncing across devices.",
        "I need help setting up API integration for my website."
    ]
    
    # Create rows of 2 buttons each
    for i in range(0, len(sample_tickets), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(sample_tickets):
                ticket = sample_tickets[i + j]
                if col.button(f"📌 {ticket[:55]}...", key=f"sample_{i+j}"):
                    st.session_state.ticket_text = ticket
                    st.rerun()

# Handle clear button
if clear_btn:
    st.session_state.ticket_text = ""
    st.rerun()

# Handle prediction
if predict_btn and st.session_state.ticket_text:
    with result_container:
        with st.spinner(f"Analyzing with {approach}..."):
            client = init_groq()
            prediction = predict_tag(client, st.session_state.ticket_text, approach, categories)
            st.success(f"**Predicted Tag:** `{prediction}`")
            if prediction in categories:
                st.info(f"📖 **Description:** {categories[prediction]}")

# Handle top 3 prediction
if top3_btn and st.session_state.ticket_text:
    with result_container:
        with st.spinner("Generating top 3 predictions..."):
            client = init_groq()
            predictions = predict_top_3(client, st.session_state.ticket_text, categories)
            
            if predictions:
                st.subheader("🏆 Top 3 Predictions")
                for i, pred in enumerate(predictions[:3], 1):
                    category = pred.get("category", "unknown")
                    confidence = pred.get("confidence", 0)
                    confidence_pct = confidence * 100 if isinstance(confidence, (int, float)) else 0
                    
                    st.markdown(f"**{i}. {category}**")
                    st.markdown(f"🎯 Confidence: `{confidence_pct:.1f}%`")
                    st.progress(min(confidence, 1.0))
                    
                    if category in categories:
                        st.caption(f"📖 {categories[category]}")
                    st.markdown("---")
            else:
                st.warning("Could not generate predictions. Please try again.")

# Footer
st.divider()
st.markdown("""
### 🎓 Learning Outcomes

| Technique | Description |
|-----------|-------------|
| **Zero-Shot** | LLM classifies without examples - good for general cases |
| **Few-Shot** | LLM learns from examples - better accuracy for specific domains |
| **Top-3 Predictions** | Shows multiple likely tags for ambiguous tickets |
""")