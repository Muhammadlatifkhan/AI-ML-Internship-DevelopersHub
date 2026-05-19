"""
Evaluate and compare Zero-Shot vs Few-Shot performance
"""

import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Categories
categories = {
    "network_issues": "Internet connectivity, Wi-Fi problems",
    "billing": "Payment issues, refunds, incorrect charges",
    "account_access": "Login problems, password reset",
    "technical_bug": "App crashes, error messages",
    "subscription": "Plan changes, cancellation",
    "performance": "Slow loading, lag",
    "api_support": "API integration"
}

# Test tickets
test_tickets = [
    ("My WiFi keeps disconnecting", "network_issues"),
    ("I was charged twice", "billing"),
    ("Can't log into my account", "account_access"),
    ("App crashes when I open it", "technical_bug"),
    ("How to cancel my plan", "subscription"),
]

def zero_shot_predict(ticket):
    prompt = f"""Classify this support ticket into one category: network_issues, billing, account_access, technical_bug, subscription, performance, api_support

Ticket: "{ticket}"

Category:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=50
    )
    return response.choices[0].message.content.strip().lower()

def few_shot_predict(ticket):
    examples = [
        'Ticket: "Internet keeps disconnecting"\nCategory: network_issues',
        'Ticket: "Wrong charge on my bill"\nCategory: billing',
        'Ticket: "Forgot my password"\nCategory: account_access',
    ]
    examples_text = "\n\n".join(examples)
    
    prompt = f"""Here are examples:
{examples_text}

Now classify: "{ticket}"

Category:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=50
    )
    return response.choices[0].message.content.strip().lower()

print("=" * 60)
print("TASK 11: ZERO-SHOT vs FEW-SHOT COMPARISON")
print("=" * 60)

results = []

for ticket, expected in test_tickets:
    print(f"\n📝 Ticket: {ticket}")
    print(f"   Expected: {expected}")
    
    # Zero-shot
    start = time.time()
    zs_pred = zero_shot_predict(ticket)
    zs_time = time.time() - start
    zs_correct = expected in zs_pred
    
    # Few-shot
    start = time.time()
    fs_pred = few_shot_predict(ticket)
    fs_time = time.time() - start
    fs_correct = expected in fs_pred
    
    results.append({
        "ticket": ticket,
        "expected": expected,
        "zero_shot_pred": zs_pred,
        "zero_shot_correct": zs_correct,
        "few_shot_pred": fs_pred,
        "few_shot_correct": fs_correct
    })
    
    print(f"   Zero-shot: {zs_pred} {'✅' if zs_correct else '❌'} ({zs_time:.2f}s)")
    print(f"   Few-shot:  {fs_pred} {'✅' if fs_correct else '❌'} ({fs_time:.2f}s)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

zs_accuracy = sum(1 for r in results if r["zero_shot_correct"]) / len(results) * 100
fs_accuracy = sum(1 for r in results if r["few_shot_correct"]) / len(results) * 100

print(f"\nZero-Shot Accuracy: {zs_accuracy:.1f}%")
print(f"Few-Shot Accuracy: {fs_accuracy:.1f}%")
print(f"Improvement: +{fs_accuracy - zs_accuracy:.1f}%")

print("\n✅ Evaluation complete!")