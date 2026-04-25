"""
Task 4: Health Chatbot - Groq API (Working Version)
"""

from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PUT YOUR ACTUAL GROQ API KEY HERE
# Get from: https://console.groq.com
# ============================================
GROQ_API_KEY = "USE YOUR OWN API KEY"  # REPLACE WITH YOUR REAL KEY!

# Initialize client
client = Groq(api_key=GROQ_API_KEY)

print("="*60)
print("TASK 4: HEALTH CHATBOT (Groq API - Llama 3.3 70B)")
print("="*60)

# Test connection with NEW model
try:
    test = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="llama-3.3-70b-versatile",  # ✅ UPDATED MODEL
        max_tokens=5
    )
    print("✅ Groq API connected! Using Llama 3.3 70B\n")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check your API key")
    exit(1)

# Safety keywords
unsafe_keywords = [
    'dose', 'dosage', 'prescription', 'medicine', 'drug', 'medication',
    'treatment', 'cure', 'diagnosis', 'treat', 'prescribe', 'antibiotic'
]

def is_safe_query(query):
    query_lower = query.lower()
    for keyword in unsafe_keywords:
        if keyword in query_lower:
            return False, keyword
    return True, None

def get_response(user_query):
    is_safe, unsafe_word = is_safe_query(user_query)
    
    if not is_safe:
        return f"""⚠️ SAFETY: I cannot provide advice about '{unsafe_word}'. Please consult a doctor."""
    
    system_prompt = """You are a helpful health information assistant. 
You provide ONLY general, educational health information.
You NEVER give medical advice, diagnoses, or treatment recommendations.
Keep responses concise and helpful."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ UPDATED MODEL
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content + "\n\n💡 General info only. See a doctor for medical concerns."
    except Exception as e:
        return f"Error: {e}"

# Simple chat
print("Health Chatbot - Type 'quit' to exit\n")
while True:
    user = input("You: ").strip()
    if user.lower() in ['quit', 'exit']:
        print("Goodbye! Stay healthy! 👋")
        break
    if user:
        print("\nBot: ", end="")
        print(get_response(user))
        print()