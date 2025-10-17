"""
Modal App for running nanochat training on 8xH100 GPUs.

This runs the original Karpathy nanochat speedrun pipeline on Modal infrastructure.

Usage:
    modal run modal_app.py --command test
    modal run modal_app.py --command train
    modal deploy modal_app.py  # Deploy chat interface
"""

import modal
import os
import subprocess
import time

# -----------------------------------------------------------------------------
# Modal Configuration
# -----------------------------------------------------------------------------

# GPU configuration - 8xH100 for distributed training
GPU_CONFIG = "H100:8"

# Create persistent volumes for data and checkpoints
data_volume = modal.Volume.from_name("nanochat-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("nanochat-checkpoints", create_if_missing=True)

# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential")
    # Install Rust for tokenizer
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    # Install PyTorch with CUDA support (includes Flash Attention via F.scaled_dot_product_attention)
    # Using PyTorch 2.8+ as specified in pyproject.toml for proper torch.compile memory optimizations
    .pip_install("torch>=2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cu128")
    # Install other dependencies
    .pip_install(
        "tiktoken",
        "tokenizers",
        "numpy",
        "datasets",
        "tqdm",
        "requests",
        "wandb",
        "maturin",
        "pyarrow",
        "psutil",
        "fastapi",
        "uvicorn",
    )
    # Copy the nanochat code (this will be last to allow Docker layer caching)
    .add_local_dir(".", "/root/nanochat", copy=True)
    .workdir("/root/nanochat")
)

app = modal.App(name="nanochat")

# -----------------------------------------------------------------------------
# Modal Functions
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        "/root/.cache/nanochat": data_volume,
        "/root/checkpoints": checkpoint_volume,
    },
    timeout=60 * 10,  # 10 minute timeout for tests
)
def test_setup():
    """Test if the Modal environment is set up correctly."""
    import torch
    import sys
    
    print("=" * 80)
    print("🧪 Testing Modal Setup")
    print("=" * 80)
    
    # Basic GPU info
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
    
    # Test nanochat imports
    try:
        from nanochat.gpt import GPT, GPTConfig
        print("✅ nanochat.gpt imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        from nanochat.tokenizer import get_tokenizer
        print("✅ nanochat.tokenizer imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    return True


# -----------------------------------------------------------------------------
# Serving Function (Single A10G GPU for inference)
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",  # Single cheap GPU for inference ($1.10/hr vs $24/hr for 8xH100)
    volumes={
        "/root/.cache/nanochat": data_volume,
    },
    scaledown_window=300,  # Keep warm for 5 min after last request
    timeout=60 * 60,  # 1 hour max per request
)
@modal.concurrent(max_inputs=10)  # Handle up to 10 concurrent users
@modal.asgi_app()
def serve_chat():
    """
    Serve the chat web interface using the original scripts/chat_web.py.
    
    This loads the trained checkpoint and serves it on a single A10G GPU.
    Uses the EXACT same code as Karpathy's original nanochat deployment -
    we're just running it on Modal instead of Lambda.
    
    The model will auto-scale to 0 when idle, saving money!
    """
    import sys
    import os
    
    # Set environment variables for nanochat
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    
    # Configure sys.argv BEFORE importing chat_web
    # The chat_web module parses args at import time
    sys.argv = [
        "chat_web.py",
        "--source", "sft",         # Load from chatsft_checkpoints (supervised finetuned - best for chat!)
        "--num-gpus", "1",         # Single GPU for inference
        "--temperature", "0.8",    # Default sampling temperature
        "--top-k", "50",           # Default top-k sampling
        "--max-tokens", "512",     # Default max response tokens
    ]
    
    # Now import the chat_web module
    # This will parse the args and create the FastAPI app
    sys.path.insert(0, "/root/nanochat")
    from scripts import chat_web
    
    # Return the FastAPI app to Modal's ASGI wrapper
    return chat_web.app


# -----------------------------------------------------------------------------
# Local entrypoint for testing
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main(command: str = "test"):
    """
    Main entrypoint for the Modal app.
    
    Usage:
        modal run modal_app.py --command test       # Test GPU setup
        modal run modal_app.py --command train      # Full training pipeline (8xH100)
        modal deploy modal_app.py                   # Deploy chat interface (1xA10G)
    """
    if command == "test":
        print("🧪 Running setup test...")
        result = test_setup.remote()
        print(f"\n{'✅' if result else '❌'} Test completed")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: test")
        print("\nTo deploy the chat interface, use:")
        print("  modal deploy modal_app.py")

