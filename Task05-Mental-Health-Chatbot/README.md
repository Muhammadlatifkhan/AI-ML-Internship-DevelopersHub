
# Task 5: Mental Health Chatbot (Fine-tuned)

## 🎯 Objective
Build a basic chatbot that provides supportive and empathetic responses for stress, anxiety, and emotional wellness using a fine-tuned Large Language Model.

## 🤖 Solution Overview
This chatbot uses **DistilGPT2** fine-tuned with **LoRA (Low-Rank Adaptation)** on empathetic conversation data. The model is trained to recognize emotional cues and respond with compassion and support.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **DistilGPT2** | Base language model (lightweight, 353M parameters) |
| **LoRA (PEFT)** | Parameter-efficient fine-tuning |
| **Hugging Face Transformers** | Model loading and inference |
| **PyTorch** | Deep learning framework |
| **Custom Dataset** | Empathetic conversation examples |

## 📁 Project Structure
```
Task5-Mental-Health-Chatbot/
├── mental_health_chatbot_lora/     # Fine-tuned LoRA adapters
│   ├── adapter_config.json         # LoRA configuration
│   ├── adapter_model.safetensors   # Fine-tuned weights
│   ├── tokenizer.json              # Tokenizer vocabulary
│   └── tokenizer_config.json       # Tokenizer settings
├── task5_chatbot.py                # Main chatbot script
├── README.md                       # Documentation
└── requirements.txt                # Python dependencies
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Chatbot
```bash
python task5_chatbot.py
```

## 💬 Example Interaction

```
🧑 You: I'm feeling really anxious about my interview tomorrow
🤗 Bot: Take a deep breath. You've got this! Anxiety is tough, but you're stronger than it.

🧑 You: I feel so alone today
🤗 Bot: You're not alone. I'm here with you. What's been on your mind?

🧑 You: I'm happy because I finished my project
🤗 Bot: That's wonderful! I'm so happy for you! You deserve this joy.

🧑 You: quit
🤗 Thank you for talking with me. Remember, you're not alone. Take care of yourself! 💙
```

## 🔧 Fine-tuning Details

| Parameter | Value |
|-----------|-------|
| Base Model | DistilGPT2 |
| Fine-tuning Method | LoRA |
| LoRA Rank (r) | 8 |
| LoRA Alpha | 32 |
| Trainable Parameters | 147,456 |
| Training Epochs | 10 |
| Learning Rate | 3e-4 |
| Batch Size | 8 |
| GPU Used | Tesla T4 (15.6 GB) |

## 📊 Emotional Detection

The chatbot detects and responds to these emotional states:

| Emotion | Keywords | Response Style |
|---------|----------|----------------|
| **Anxious** | nervous, worried, scared, panic | Calming, reassuring |
| **Sad** | depressed, hurt, awful, terrible | Compassionate, validating |
| **Lonely** | alone, isolated, abandoned | Supportive, connecting |
| **Stressed** | overwhelmed, exhausted, burnout | Practical, grounding |
| **Joyful** | happy, proud, excited, celebrate | Celebratory, affirming |
| **Default** | - | Listening, supportive |

## ✅ Task Completion Checklist

- [x] Fine-tune a small LLM using Hugging Face's Trainer API
- [x] Use LoRA for parameter-efficient fine-tuning
- [x] Train on empathetic conversation data
- [x] Ensure tone is gentle and emotionally supportive
- [x] Build command-line interface for testing
- [x] Implement emotion detection from user input
- [x] Add fallback responses for reliability

## 📝 Notes

- **First run:** Downloads base DistilGPT2 model (~353 MB) - subsequent runs use cache
- **Training data:** Custom curated empathetic conversations (6 examples, expanded to 24)
- **Inference:** Runs on CPU or GPU (automatically detected)
- **Fallback responses:** Built-in empathetic responses ensure the bot never leaves the user without support

## 🔄 Improvement Suggestions

For a production-ready chatbot:
- Train on full EmpatheticDialogues dataset (25,000+ conversations)
- Increase training epochs to 15-20
- Use larger model like GPT-2 Medium or Llama-2-7B
- Add streaming responses for better UX

## 🔗 Resources

- [DistilGPT2 Model](https://huggingface.co/distilgpt2)
- [LoRA Paper (Microsoft)](https://arxiv.org/abs/2106.09685)
- [PEFT Library (Hugging Face)](https://github.com/huggingface/peft)
- [EmpatheticDialogues Dataset](https://huggingface.co/datasets/empathetic_dialogues)

## 👨‍💻 Author
Muhammad Latif - AI/ML Engineering Intern

## 📅 Date
April 25, 2026
