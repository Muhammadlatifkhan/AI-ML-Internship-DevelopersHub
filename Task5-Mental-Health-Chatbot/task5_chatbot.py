"""
Task 5: Mental Health Chatbot - Enhanced Inference
Uses fine-tuned LoRA model with post-processing for better responses
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import random
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TASK 5: MENTAL HEALTH CHATBOT")
print("Empathetic Support Assistant")
print("="*60)

# Load model
print("\n🤖 Loading fine-tuned model...")

base_model_name = "distilgpt2"
lora_path = "./mental_health_chatbot_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
)

model = PeftModel.from_pretrained(base_model, lora_path)
print("✅ Model loaded successfully!")

# Fallback responses (in case model produces bad output)
fallback_responses = {
    "anxious": [
        "Take a deep breath. You've got this! Anxiety is tough, but you're stronger.",
        "I hear your worry. Let's take it one step at a time. What's one small thing you can do right now?",
        "Your feelings are valid. Remember to be kind to yourself during stressful times."
    ],
    "sad": [
        "I'm sorry you're feeling this way. Would you like to talk more about it?",
        "It's okay to not be okay. You matter, and your feelings are important.",
        "Thank you for sharing with me. That takes courage."
    ],
    "lonely": [
        "You're not alone. I'm here with you. What's been on your mind?",
        "Feeling lonely is really hard. Sometimes reaching out to one person helps.",
        "I'm glad you reached out. Connection matters, and you deserve support."
    ],
    "stressed": [
        "That sounds overwhelming. Can we break it down into smaller pieces?",
        "Remember to take breaks. Your mental health comes first.",
        "Stress is tough. Let's try taking three deep breaths together."
    ],
    "joyful": [
        "That's wonderful! I'm so happy for you!",
        "What a great achievement! Thank you for sharing your joy with me.",
        "You deserve to celebrate this! Truly happy for you."
    ],
    "default": [
        "I'm here for you. Tell me more about how you're feeling.",
        "Thank you for sharing. Would you like to talk more?",
        "I appreciate you opening up. How can I support you today?"
    ]
}

def detect_emotion(text):
    """Detect emotion from user input"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["anxious", "nervous", "worried", "scared", "panic"]):
        return "anxious"
    elif any(word in text_lower for word in ["sad", "depressed", "awful", "terrible", "hurt", "cry"]):
        return "sad"
    elif any(word in text_lower for word in ["lonely", "alone", "isolated", "abandoned"]):
        return "lonely"
    elif any(word in text_lower for word in ["stressed", "overwhelmed", "tired", "exhausted"]):
        return "stressed"
    elif any(word in text_lower for word in ["happy", "proud", "excited", "joy", "celebrate"]):
        return "joyful"
    else:
        return "default"

def post_process_response(response, emotion):
    """Clean up repetitive or bad responses"""
    
    # Remove repeated phrases
    words = response.split()
    if len(words) > 5:
        # Check for repetition (same word 5+ times)
        from collections import Counter
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)[0]
        
        if most_common[1] > 5:
            # Too repetitive - use fallback
            return random.choice(fallback_responses.get(emotion, fallback_responses["default"]))
    
    # Limit response length
    if len(response) > 200:
        response = response[:200]
    
    # Remove strange characters
    response = response.replace('�', '')
    
    # If response is too short or nonsensical
    if len(response) < 10 or response.count('I') > 10:
        return random.choice(fallback_responses.get(emotion, fallback_responses["default"]))
    
    return response

def generate_response(user_input):
    """Generate empathetic response using fine-tuned model"""
    
    emotion = detect_emotion(user_input)
    
    # Format prompt
    prompt = f"<|emotion|>{emotion}\n<|user|>{user_input}\n<|supporter|>"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,  # Reduces repetition
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract supporter response
        if "<|supporter|>" in response:
            response = response.split("<|supporter|>")[-1].strip()
        
        # Clean up
        response = response.replace("<|endoftext|>", "").strip()
        
        # Post-process
        response = post_process_response(response, emotion)
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        response = random.choice(fallback_responses.get(emotion, fallback_responses["default"]))
    
    return response, emotion

# Interactive chat
print("\n" + "="*60)
print("💬 Let's Chat - Mental Health Support")
print("="*60)
print("\n💡 Share how you're feeling (anxious, sad, lonely, stressed, or happy)")
print("💡 I'm here to listen and support you")
print("💡 Type 'quit' to exit\n")

while True:
    user_input = input("🧑 You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q', 'goodbye']:
        print("\n🤗 Thank you for talking with me. Remember, you're not alone.")
        print("   Take care of yourself! 💙\n")
        break
    
    if not user_input:
        continue
    
    response, emotion = generate_response(user_input)
    print(f"🤗 Bot: {response}\n")