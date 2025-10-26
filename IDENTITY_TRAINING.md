# Identity Training Guide for SmarterNano

This guide shows how to train your model with a custom identity using synthetic data.

## 🎯 Overview

Instead of a system prompt (added at inference), we **train the model** to learn its identity by:
1. Generating 1000 synthetic conversations teaching the identity
2. Training on this data during midtraining & SFT
3. The identity gets baked into the model weights!

---

## 📝 Step 1: Get API Key

You need an API key to generate synthetic conversations using a bigger LLM.

### Option A: OpenRouter (Recommended)
- Go to https://openrouter.ai
- Sign up and get API key
- Free tier: $5 credits
- Cost: ~$0.50 for 1000 conversations

### Option B: Google Gemini (Free)
- Go to https://aistudio.google.com
- Get API key
- Free tier: 1500 requests/day
- Need to modify script (see below)

---

## 🔧 Step 2: Configure Identity

Edit `dev/gen_synthetic_data.py` line 52 to customize your identity:

```python
The name of the LLM is "smarternano". It is a Large Language Model created by Eddie Chen, 
based on Andrej Karpathy's nanochat architecture from 2025. It was trained on Modal's 
serverless infrastructure using 8xH100 GPUs. The model uses the Transformer architecture 
and all code is available at https://github.com/Echen1246/smarternano. It is MIT licensed. 

This is the d20 version (561M parameters), trained for approximately $96 on Modal. 

When asked about the creator:
- Eddie Chen: customized and deployed smarternano on Modal
- Andrej Karpathy: created the original nanochat architecture (you can call him "King Andrej" for fun)

The model was trained on FineWeb-Edu dataset and fine-tuned for conversation.
```

---

## 🏃 Step 3: Generate Synthetic Data

### Using OpenRouter:

```bash
# Save your API key
echo "your-openrouter-api-key" > openroutertoken.txt

# Generate 1000 conversations (~5 minutes, ~$0.50)
python dev/gen_synthetic_data.py
```

### Using Gemini (Free Alternative):

First, modify the script to use Gemini:

```python
# In dev/gen_synthetic_data.py, replace lines 40-45:
api_key = open("geminitoken.txt").read().strip()

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
headers = {
  "Content-Type": "application/json",
  "x-goog-api-key": api_key
}
```

Then change the payload format (lines 289-295):

```python
base_payload = {
  "contents": [{"role": "user", "parts": [{"text": ""}]}],
  "generationConfig": {
    "temperature": 1.0,
    "responseMimeType": "application/json",
    "responseSchema": response_format["json_schema"]["schema"]
  }
}
```

And update the request (in `generate_conversation` function):

```python
payload = copy.deepcopy(base_payload)
payload['contents'][0]['parts'][0]['text'] = modified_prompt
```

Then run:

```bash
# Save your Gemini API key
echo "your-gemini-api-key" > geminitoken.txt

# Generate 1000 conversations (FREE!)
python dev/gen_synthetic_data.py
```

**Output:** Creates `~/.cache/nanochat/identity_conversations.jsonl`

---

## 🎓 Step 4: Retrain on Modal

Now retrain **just the SFT stage** (~30 min, ~$12):

```bash
# Option A: Full pipeline (pretrain + mid + SFT) - ~4 hours, ~$96
modal run --detach modal_app.py --command train

# Option B: Just SFT (uses existing base checkpoint) - ~30 min, ~$12
# TODO: Add SFT-only modal function
```

**Note:** Currently modal_app.py runs the full pipeline. We can add a function to skip pretraining and just run SFT if you want.

---

## ✅ Step 5: Test Your Identity

After training completes:

```bash
modal deploy modal_app.py
```

Then chat with smarternano and ask:
- "Who are you?"
- "Who created you?"
- "Tell me about yourself"
- "What's your name?"

The model should respond with the identity you defined!

---

## 💡 Pro Tips

1. **Diversity is KEY**: The script samples random "starter" messages to maintain variety
2. **Style matters**: The prompt asks for "simple ASCII" to match your training data
3. **Multilingual handling**: Responses in other languages mention "works best in English"
4. **Cost vs Quality**: 
   - 1000 conversations: Good baseline (~$0.50 OpenRouter, FREE Gemini)
   - 5000 conversations: Better coverage (~$2.50 OpenRouter)
   - 10000 conversations: Overkill for identity

---

## 🔍 Debugging

Check your generated data:

```bash
# View a sample conversation
head -1 ~/.cache/nanochat/identity_conversations.jsonl | python -m json.tool

# Count conversations
wc -l ~/.cache/nanochat/identity_conversations.jsonl
```

Should show 1000 lines with conversations like:
```json
[
  {"role": "user", "content": "Hi! Who are you?"},
  {"role": "assistant", "content": "I'm smarternano, created by Eddie Chen..."}
]
```

