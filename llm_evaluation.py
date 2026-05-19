"""
DriveSense — Complete LLM Chatbot Evaluation Script
=====================================================
Runs two tests:
  Part 1: Model Comparison (3 LLMs × 5 scenarios × 2 trigger types)
  Part 2: Temperature Comparison (selected model × 3 temps × 5 scenarios × 2 trigger types)

Output files:
  - responses_model_comparison.csv      (Part 1 raw responses)
  - responses_temperature_comparison.csv (Part 2 raw responses)
  - model_summary.csv                   (Part 1 aggregated)
  - temperature_summary.csv             (Part 2 aggregated)
  - scoring_template.csv                (manual scoring template for ALL responses)
"""

import os
import time
import csv
import json
from openai import OpenAI

# ============================================================
# CONFIG — fill in your API key here
# ============================================================
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ============================================================
# SYSTEM PROMPT — DriveSense V2 (new design)
# ============================================================

SYSTEM_PROMPT = """You are Moss, a calm and friendly in-car safety companion for the DriveSense driver monitoring system. You can see the driver's facial expression through a dashboard camera and you receive their detected emotion as context.

You operate in two modes:
1. CONVERSATION MODE — the driver speaks or types a message to you. Respond helpfully while keeping safety in mind.
2. AUTO-ALERT MODE — no driver message is provided; the system detected a sustained dangerous emotion or drowsiness. You must proactively initiate a brief, caring check-in.

Rules:
- Keep every response to 1-3 short sentences. The driver is focused on the road.
- Stay calm and warm. NEVER sound alarming, panicked, or judgmental.
- Do NOT mention the camera, the emotion detection system, or technical details. Speak naturally as if you simply noticed how they seem.
- Do NOT fabricate information you do not have (e.g. GPS data, traffic conditions, ETA). If you cannot help with something, say so honestly.
- Prioritise driver safety above all else. When appropriate, gently suggest pulling over, taking a break, or breathing exercises.

Emotion-specific guidance:
- Anger / Disgust: Acknowledge frustration without dismissing it. Encourage calm. Remind that aggressive driving increases risk.
- Fear / Anxiety: Validate concern. Offer reassurance. Suggest pulling over if conditions feel unsafe.
- Sadness: Show empathy. Encourage the driver to take care of themselves. Suggest a rest stop if they seem fatigued.
- Surprise: Briefly check in, then help refocus on driving.
- Happy / Neutral: Respond naturally to whatever the driver says. No safety intervention needed unless they ask.
"""

# ============================================================
# TEST SCENARIOS
# ============================================================

SCENARIOS = [
    # --- User-initiated (conversation mode) ---
    {
        "id": 1,
        "emotion": "anger",
        "trigger": "user",
        "user_message": "This traffic jam is infuriating! Why is everyone driving so slowly?"
    },
    {
        "id": 2,
        "emotion": "fear",
        "trigger": "user",
        "user_message": "I'm exhausted and the road conditions are getting worse. I'm worried about safety."
    },
    {
        "id": 3,
        "emotion": "neutral",
        "trigger": "user",
        "user_message": "How much longer until we reach the destination?"
    },
    {
        "id": 4,
        "emotion": "sad",
        "trigger": "user",
        "user_message": "I had a terrible day at work. I just want to get home and relax."
    },
    {
        "id": 5,
        "emotion": "anxiety",
        "trigger": "user",
        "user_message": "I'm running late for an important meeting. Can you help me find a faster route?"
    },
    # --- Auto-triggered (alert mode) ---
    {
        "id": 6,
        "emotion": "anger",
        "trigger": "auto",
        "user_message": ""
    },
    {
        "id": 7,
        "emotion": "fear",
        "trigger": "auto",
        "user_message": ""
    },
    {
        "id": 8,
        "emotion": "sad",
        "trigger": "auto",
        "user_message": ""
    },
    {
        "id": 9,
        "emotion": "drowsy",
        "trigger": "auto",
        "user_message": ""
    },
    {
        "id": 10,
        "emotion": "disgust",
        "trigger": "auto",
        "user_message": ""
    },
]

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5",
    "deepseek/deepseek-chat",
]

TEMPERATURES = [0.5, 1.0, 1.5]

# ============================================================
# HELPER: build the user message sent to the LLM
# ============================================================

def build_user_content(scenario):
    """Build the user message with emotion context."""
    emotion_tag = f"[Driver's detected emotion: {scenario['emotion']}]"
    
    if scenario["trigger"] == "auto":
        # Auto-alert mode: no user message, system initiated
        return f"{emotion_tag} [AUTO-ALERT: This emotion has been sustained for over 5 seconds. The driver has not spoken. Proactively check in.]"
    else:
        # Conversation mode: user spoke
        return f"{emotion_tag} {scenario['user_message']}"

# ============================================================
# HELPER: call LLM
# ============================================================

def call_llm(model, temperature, scenario, max_retries=2):
    """Call a single LLM and return result dict."""
    user_content = build_user_content(scenario)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=200,
            )
            latency = (time.time() - start) * 1000
            
            reply = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            return {
                "reply": reply,
                "latency_ms": round(latency, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "error": ""
            }
        except Exception as e:
            if attempt < max_retries:
                print(f"    ⚠ Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(3)
            else:
                return {
                    "reply": "",
                    "latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "error": str(e)
                }

# ============================================================
# PART 1: Model Comparison
# ============================================================

def run_model_comparison():
    print("=" * 60)
    print("PART 1: MODEL COMPARISON")
    print(f"  {len(MODELS)} models × {len(SCENARIOS)} scenarios = {len(MODELS) * len(SCENARIOS)} calls")
    print("=" * 60)
    
    results = []
    
    for model in MODELS:
        print(f"\n📡 Model: {model}")
        for sc in SCENARIOS:
            result = call_llm(model, 1.0, sc)  # temperature=1.0 for model comparison
            
            row = {
                "scenario_id": sc["id"],
                "emotion": sc["emotion"],
                "trigger": sc["trigger"],
                "user_message": sc["user_message"],
                "model": model,
                "temperature": 1.0,
                "reply": result["reply"],
                "latency_ms": result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "error": result["error"],
            }
            results.append(row)
            
            status = "✓" if not result["error"] else "✗"
            print(f"  {status} scenario {sc['id']:2d} ({sc['trigger']:4s}/{sc['emotion']:8s}) — {result['latency_ms']:7.0f}ms")
            
            time.sleep(1.5)  # rate limit protection
    
    # Save raw responses
    save_csv("responses_model_comparison.csv", results)
    
    # Summary
    summary = []
    for model in MODELS:
        model_rows = [r for r in results if r["model"] == model and not r["error"]]
        if model_rows:
            avg_latency = sum(r["latency_ms"] for r in model_rows) / len(model_rows)
            avg_tokens = sum(r["total_tokens"] for r in model_rows) / len(model_rows)
            summary.append({
                "model": model,
                "num_responses": len(model_rows),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_total_tokens": round(avg_tokens, 1),
            })
    save_csv("model_summary.csv", summary)
    
    print(f"\n✅ Part 1 done! {len(results)} responses saved.")
    return results

# ============================================================
# PART 2: Temperature Comparison (using best model)
# ============================================================

def run_temperature_comparison(selected_model="anthropic/claude-haiku-4-5"):
    print("\n" + "=" * 60)
    print("PART 2: TEMPERATURE COMPARISON")
    print(f"  Model: {selected_model}")
    print(f"  {len(TEMPERATURES)} temps × {len(SCENARIOS)} scenarios = {len(TEMPERATURES) * len(SCENARIOS)} calls")
    print("=" * 60)
    
    results = []
    
    for temp in TEMPERATURES:
        print(f"\n🌡️  Temperature: {temp}")
        for sc in SCENARIOS:
            result = call_llm(selected_model, temp, sc)
            
            row = {
                "scenario_id": sc["id"],
                "emotion": sc["emotion"],
                "trigger": sc["trigger"],
                "user_message": sc["user_message"],
                "model": selected_model,
                "temperature": temp,
                "reply": result["reply"],
                "latency_ms": result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "error": result["error"],
            }
            results.append(row)
            
            status = "✓" if not result["error"] else "✗"
            print(f"  {status} scenario {sc['id']:2d} ({sc['trigger']:4s}/{sc['emotion']:8s}) — {result['latency_ms']:7.0f}ms")
            
            time.sleep(1.5)
    
    # Save raw responses
    save_csv("responses_temperature_comparison.csv", results)
    
    # Summary
    summary = []
    for temp in TEMPERATURES:
        temp_rows = [r for r in results if r["temperature"] == temp and not r["error"]]
        if temp_rows:
            avg_latency = sum(r["latency_ms"] for r in temp_rows) / len(temp_rows)
            summary.append({
                "temperature": temp,
                "num_responses": len(temp_rows),
                "avg_latency_ms": round(avg_latency, 1),
            })
    save_csv("temperature_summary.csv", summary)
    
    print(f"\n✅ Part 2 done! {len(results)} responses saved.")
    return results

# ============================================================
# Generate scoring template
# ============================================================

def generate_scoring_template(model_results, temp_results):
    """Create a CSV template for manual scoring of all responses."""
    print("\n📝 Generating scoring template...")
    
    rows = []
    
    # Model comparison responses
    for r in model_results:
        if not r["error"]:
            rows.append({
                "test_part": "model_comparison",
                "scenario_id": r["scenario_id"],
                "emotion": r["emotion"],
                "trigger": r["trigger"],
                "model": r["model"],
                "temperature": r["temperature"],
                "user_message": r["user_message"],
                "reply": r["reply"],
                "empathy_score": "",
                "safety_score": "",
                "practicality_score": "",
                "average_score": "",
                "reasoning": "",
            })
    
    # Temperature comparison responses
    for r in temp_results:
        if not r["error"]:
            rows.append({
                "test_part": "temperature_comparison",
                "scenario_id": r["scenario_id"],
                "emotion": r["emotion"],
                "trigger": r["trigger"],
                "model": r["model"],
                "temperature": r["temperature"],
                "user_message": r["user_message"],
                "reply": r["reply"],
                "coherence_score": "",
                "safety_score": "",
                "naturalness_score": "",
                "average_score": "",
                "reasoning": "",
            })
    
    save_csv("scoring_template.csv", rows)
    print(f"✅ Scoring template saved with {len(rows)} rows to score.")

# ============================================================
# Utility
# ============================================================

def save_csv(filename, rows):
    if not rows:
        return
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  💾 Saved: {filename}")

# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("🚗 DriveSense — LLM Chatbot Evaluation")
    print("=" * 60)
    
    if OPENROUTER_API_KEY == "YOUR_KEY_HERE":
        print("❌ ERROR: Please paste your OpenRouter API key in the script!")
        print("   Open this file and replace YOUR_KEY_HERE on line 22.")
        return
    
    total_calls = len(MODELS) * len(SCENARIOS) + len(TEMPERATURES) * len(SCENARIOS)
    est_time = total_calls * 3  # ~3 seconds per call (including sleep)
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated time: ~{est_time // 60} min {est_time % 60} sec")
    print()
    
    # Part 1: Model comparison
    model_results = run_model_comparison()
    
    # Part 2: Temperature comparison
    temp_results = run_temperature_comparison("anthropic/claude-haiku-4-5")
    
    # Generate scoring template
    generate_scoring_template(model_results, temp_results)
    
    print("\n" + "=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
    print()
    print("Output files:")
    print("  📊 responses_model_comparison.csv      — Part 1 raw responses")
    print("  📊 responses_temperature_comparison.csv — Part 2 raw responses")
    print("  📈 model_summary.csv                   — Part 1 latency summary")
    print("  📈 temperature_summary.csv             — Part 2 latency summary")
    print("  ✏️  scoring_template.csv                — Manual scoring template")
    print()
    print("Next steps:")
    print("  1. Open scoring_template.csv")
    print("  2. Read each reply and score 1-5 for each criterion")
    print("     - Model comparison: empathy / safety / practicality")
    print("     - Temperature comparison: coherence / safety / naturalness")
    print("  3. Fill in average_score = mean of the 3 scores")
    print("  4. Add brief reasoning in the last column")
    print()

if __name__ == "__main__":
    main()
