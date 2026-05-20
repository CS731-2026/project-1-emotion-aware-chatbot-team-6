# Continuous Conversation Mode

## Overview

The driver assistant now supports **hands-free continuous conversation** after each LLM reply. You no longer need to repeat "hey moss" after every chatbot response—just keep talking naturally!

## How It Works

### 1. Initial Wake-Up
```
You:     "Hey moss"
System:  🎤 Wake-word detected → Auto-recording 5 seconds
         Speech → Transcribe → Send to LLM
```

### 2. LLM Reply
```
System:  LLM generates reply → Text-to-speech (TTS) speaks
```

### 3. Continuous Listening (NEW!)
```
System:  🎧 Auto-enter 10-second "continued listen" mode
         (No wake-word needed—ANY speech triggers recording)
You:     "Can you explain more?"
System:  Voice detected → Auto-recording 5 seconds
         Speech → Transcribe → Send to LLM
```

### 4. Loop Repeats
```
System:  LLM reply → TTS → Back to step 3 (another 10-second window)
You:     "Great, thanks!"
System:  Voice detected → Recording → LLM → TTS → Continued listening...
```

### 5. Timeout / Back to Wake-Word
```
You:     [10 seconds of silence]
System:  ⏱️ Continued window timed out → Back to "listening for 'hey moss'"
         🎤 Must say "hey moss" again to restart
```

## Key Features

✅ **No repeated wake-words** - After LLM replies, just keep speaking  
✅ **Natural conversation flow** - Back-and-forth dialogue without friction  
✅ **Auto-safety timeout** - Returns to "hey moss" mode after 10s silence  
✅ **Voice activity detection** - Only records when user actually speaks  
✅ **Seamless integration** - Works with all LLM models and emotions  

## Configuration

Edit `drivesense/frontend/gui.py` in `__init__` to adjust:

```python
self.continued_listener = ContinuedConversationListener(
    config=WakeWordConfig(...),
    on_voice_detected=self.on_continued_voice_detected,
    on_timeout=self.on_continued_timeout,
    timeout_seconds=10.0,  # Adjust this value (seconds)
)
```

**timeout_seconds** (default: 10.0)
- How long to listen for follow-up speech after LLM reply
- Set lower (5.0) for faster return to wake-word mode
- Set higher (15.0) for more time to think before speaking

## UI Indicators

| Status | Meaning |
|--------|---------|
| `listening for 'hey moss'` | Waiting for wake-word (initial state) |
| `awaiting follow-up input (10s window)...` | In continuous mode—say something! |
| `continued mode timeout, resumed wake-word listening` | 10s elapsed, now waiting for "hey moss" again |

## Implementation Details

- **WakeWordListener**: Detects "hey moss" / "hey" / "moss" keywords
- **ContinuedConversationListener**: New class that listens for ANY speech (no keyword matching)
- **SpeechWorker**: Transcribes user input via faster-whisper (tiny model, ~1-2 sec)
- **ChatWorker**: Sends transcription to OpenRouter LLM
- **Threading**: All listeners run as background daemon threads; safe to interrupt

## Troubleshooting

**Q: The bot keeps returning to "hey moss" mode too quickly**
- A: Increase `timeout_seconds` (e.g., 15.0 instead of 10.0)

**Q: The bot doesn't detect my follow-up speech**
- A: Speak clearly and pause between utterances (1+ second gap recommended)
- Whisper transcriber needs clear audio to recognize speech

**Q: Continuous mode sometimes skips without waiting 10 seconds**
- A: This is normal—it stops early if LLM is ready before timeout

**Q: Can I disable continuous mode?**
- A: Currently it's always on. To disable, comment out `handle_chat_response()` lines 862-868 in gui.py

## Future Enhancements

- [ ] Adaptive timeout (shorter if conversation is fast, longer if slow)
- [ ] Confidence threshold for voice detection (to avoid false positives)
- [ ] Visual countdown timer showing remaining seconds
- [ ] User preference toggle in GUI settings
- [ ] Multi-turn context awareness (remember previous replies)

## Code References

- `drivesense/backend/wake_word.py` - WakeWordListener & ContinuedConversationListener
- `drivesense/frontend/gui.py` - GUI integration (handle_chat_response, on_continued_voice_detected, on_continued_timeout)
- `drivesense/backend/speech.py` - WhisperTranscriber, TextToSpeech
- `drivesense/backend/chatbot.py` - DriverAssistantChatbot (LLM backend)
