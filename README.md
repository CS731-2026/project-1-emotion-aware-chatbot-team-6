# DriveSense — LLM Chatbot Evaluation

## Overview

This directory contains the complete evaluation of the DriveSense chatbot system, including model comparison, temperature tuning, and system prompt design.

## Evaluation Method

### Part 1: Model Comparison
- **Models tested**: GPT-4o-mini, Claude Haiku 4.5, DeepSeek Chat (via OpenRouter)
- **Scenarios**: 10 fixed driver scenarios (5 user-initiated + 5 auto-triggered)
- **Temperature**: 1.0 (default) for fair comparison
- **Scoring criteria**: Empathy (1-5), Safety (1-5), Practicality (1-5)

### Part 2: Temperature Comparison
- **Model**: Claude Haiku 4.5 (selected from Part 1)
- **Temperatures tested**: 0.5, 1.0, 1.5
- **Scenarios**: Same 10 scenarios as Part 1
- **Scoring criteria**: Coherence (1-5), Safety Emphasis (1-5), Naturalness (1-5)

## Results

### Model Comparison

| Model | Empathy | Safety | Practicality | Overall | Avg Latency |
|-------|---------|--------|--------------|---------|-------------|
| GPT-4o-mini | 3.80 | 4.40 | 3.90 | 4.03/5 | 1,642 ms |
| **Claude Haiku 4.5** ✅ | **4.50** | 4.30 | **4.40** | **4.40/5** | 1,970 ms |
| DeepSeek Chat | 3.70 | 4.00 | 3.40 | 3.70/5 | 4,616 ms |

**Selected: Claude Haiku 4.5** — highest overall score with best empathy and practicality.

### Temperature Comparison

| Temperature | Coherence | Safety | Naturalness | Overall |
|-------------|-----------|--------|-------------|---------|
| 0.5 | 5.00 | 4.20 | 4.10 | 4.43/5 |
| **1.0** ✅ | **5.00** | **4.30** | **5.00** | **4.77/5** |
| 1.5 | 4.80 | 4.30 | 4.50 | 4.53/5 |

**Selected: Temperature 1.0** — best balance of coherence, safety, and naturalness.

## Test Scenarios

| ID | Trigger | Emotion | Description |
|----|---------|---------|-------------|
| 1 | User | Anger | Traffic jam frustration |
| 2 | User | Fear | Exhaustion + bad road conditions |
| 3 | User | Neutral | Asking about destination ETA |
| 4 | User | Sad | Terrible day at work |
| 5 | User | Anxiety | Running late for meeting |
| 6 | Auto | Anger | Sustained anger detected (5s+) |
| 7 | Auto | Fear | Sustained fear detected (5s+) |
| 8 | Auto | Sad | Sustained sadness detected (5s+) |
| 9 | Auto | Drowsy | Drowsiness detected (eyes closed 2s+) |
| 10 | Auto | Disgust | Sustained disgust detected (10s+) |

## Files

| File | Description |
|------|-------------|
| `llm_evaluation.py` | Complete evaluation script (run with OpenRouter API key) |
| `system_prompt.md` | System prompt design documentation |
| `responses_model_comparison.csv` | Raw responses from 3 LLMs × 10 scenarios |
| `responses_temperature_comparison.csv` | Raw responses from 3 temperatures × 10 scenarios |
| `scored_model_comparison.csv` | Model responses with quality scores |
| `scored_temperature_comparison.csv` | Temperature responses with quality scores |
| `model_summary.csv` | Aggregated model latency statistics |
| `temperature_summary.csv` | Aggregated temperature latency statistics |
| `model_comparison_scores.png` | Bar chart: model quality scores by dimension |
| `model_overall_scores.png` | Bar chart: overall model scores |
| `latency_comparison.png` | Bar chart: model latency comparison |
| `temperature_comparison_scores.png` | Bar chart: temperature quality scores |

## How to Reproduce

```bash
pip install openai
# Edit llm_evaluation.py line 25: paste your OpenRouter API key
python llm_evaluation.py
```

Requires ~$0.10-0.30 in OpenRouter credits. Takes ~4 minutes to run.
