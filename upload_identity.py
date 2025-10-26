"""
Upload local identity_conversations.jsonl to Modal volume.
"""
import modal
import os

# Get the volume
volume = modal.Volume.from_name("nanochat-data", create_if_missing=True)

# Local file path
local_path = os.path.expanduser("~/.cache/nanochat/identity_conversations.jsonl")

if not os.path.exists(local_path):
    print(f"❌ File not found: {local_path}")
    exit(1)

# Count conversations
with open(local_path, 'r') as f:
    num_conversations = sum(1 for _ in f)

print(f"📤 Uploading {num_conversations} conversations to Modal...")
print(f"   From: {local_path}")
print(f"   To: nanochat-data/identity_conversations.jsonl")

# Upload using Modal SDK
with volume.batch_upload() as batch:
    batch.put_file(local_path, "identity_conversations.jsonl")

print("✅ Upload complete!")

