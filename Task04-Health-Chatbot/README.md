# Task 4: General Health Query Chatbot

## 🎯 Objective
Build a chatbot that answers general health-related questions using a Large Language Model (LLM) via API, with prompt engineering and safety filters.

## 🤖 Solution Overview
This chatbot uses **Groq API** with **Llama 3.3 70B** - a state-of-the-art LLM that provides fast, intelligent responses. The implementation includes:
- Real LLM API integration (not hardcoded responses)
- Prompt engineering for medical safety
- Safety keyword filtering
- Interactive chat interface

## 🛠️ Technologies Used
| Technology | Purpose |
|------------|---------|
| **Groq API** | LLM provider (Llama 3.3 70B model) |
| **Python** | Core programming language |
| **python-dotenv** | API key management |

## 📁 Project Structure
Task4-Health-Chatbot/
├── task4_health_chatbot.py # Main chatbot script
├── README.md # Documentation
├── requirements.txt # Python dependencies
└── .env # API key (not committed to git)


## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install groq python-dotenv```


## 🔐 API Key Instructions

⚠️ **Important:** You need your own Groq API key to run this chatbot.

1. Sign up for free at [console.groq.com](https://console.groq.com)
2. Create an API key (starts with `gsk_`)
3. Create a `.env` file in this folder
4. Add: `GROQ_API_KEY=your_key_here`

The code will automatically load the key from the `.env` file.

📊 Example Interaction
text
You: i am sneezing

Bot: Sneezing can be caused by various factors such as allergies, colds, 
or irritants in the air. To help alleviate sneezing, you can try avoiding 
allergens, using a humidifier, or practicing good hygiene. If your 
symptoms persist, it's best to consult a healthcare professional.

💡 General info only. See a doctor for medical concerns.
🧠 Prompt Engineering
The system prompt is carefully crafted to ensure:

Only general health information is provided

No medical advice or diagnoses are given

Responses are helpful but safe

Professional consultation is encouraged when needed

✅ Task Completion Checklist
Send user queries to an LLM via API (Groq + Llama 3.3 70B)

Implement prompt engineering (system + user prompts)

Add safety filters (keyword blocking + LLM guidelines)

Build interactive conversational interface

Handle example health queries

Provide medical disclaimer

📝 Notes
Groq API provides blazing fast responses (280+ tokens/second)

Free tier available with generous rate limits

Model used: llama-3.3-70b-versatile

Runs entirely online via API (no local model download needed)
⚠️ **Note:** The API key in the code has been removed for security. 
To run this chatbot, replace `YOUR_GROQ_API_KEY_HERE` with your own 
Groq API key from [console.groq.com](https://console.groq.com).

👨‍💻 Author
AI/ML Engineering Intern - DevelopersHub Corporation

📅 Date
April 23, 2026