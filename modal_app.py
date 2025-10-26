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

GPU_CONFIG = "H100:8"

data_volume = modal.Volume.from_name("nanochat-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("nanochat-checkpoints", create_if_missing=True)

# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "build-essential")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .pip_install("torch>=2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cu128")
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
    print("Testing Modal Setup")
    print("=" * 80)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    try:
        from nanochat.gpt import GPT, GPTConfig
        print("nanochat.gpt imported successfully")
    except Exception as e:
        print(f"Import failed: {e}")
        return False
    
    try:
        from nanochat.tokenizer import get_tokenizer
        print("nanochat.tokenizer imported successfully")
    except Exception as e:
        print(f"Import failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    return True


# -----------------------------------------------------------------------------
# Training Function (8xH100 GPUs for full speedrun)
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        "/root/.cache/nanochat": data_volume,
        "/root/checkpoints": checkpoint_volume,
    },
    timeout=60 * 60 * 6,  # 6 hour timeout for full training
)
def run_speedrun(wandb_run: str = "dummy"):
    """
    Run the full nanochat training pipeline (speedrun.sh equivalent).
    
    This includes:
    1. Build tokenizer (if needed)
    2. Download dataset
    3. Copy identity conversations from volume
    4. Pretrain base model (d20, 561M params)
    5. Evaluate base model
    6. Midtraining (with identity data)
    7. Supervised finetuning (with identity data)
    8. Final evaluation
    """
    import os
    import subprocess
    import glob
    import shutil
    
    start_time = time.time()
    
    # Set environment variables
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    os.environ["WANDB_MODE"] = "disabled" if wandb_run == "dummy" else "online"
    
    print("=" * 80)
    print("🚀 Starting nanochat Speedrun with Custom Identity")
    print("=" * 80)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏷️  Wandb run: {wandb_run}")
    
    # Copy identity conversations from volume to expected location
    print("\n" + "=" * 80)
    print("📝 Setting up identity conversations...")
    print("=" * 80)
    identity_source = "/root/.cache/nanochat/identity_conversations.jsonl"
    if os.path.exists(identity_source):
        with open(identity_source, 'r') as f:
            num_conversations = sum(1 for _ in f)
        print(f"✅ Found {num_conversations} custom identity conversations")
    else:
        print("⚠️  No custom identity file found, training will use defaults")
    
    # Check if tokenizer exists, build if needed
    tokenizer_path = f"{os.environ['NANOCHAT_BASE_DIR']}/tokenizer/tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        print("\n" + "=" * 80)
        print("🔤 Tokenizer not found - building it first...")
        print("=" * 80)
        
        # Build the rustbpe tokenizer
        print("\n📦 Building rustbpe with maturin...")
        subprocess.run(
            ["bash", "-c", "source $HOME/.cargo/env && maturin build --release --manifest-path rustbpe/Cargo.toml"],
            check=True
        )
        
        # Install the wheel
        print("📦 Installing rustbpe...")
        wheel_files = glob.glob("rustbpe/target/wheels/*.whl")
        if wheel_files:
            subprocess.run(["pip", "install", "--force-reinstall", wheel_files[0]], check=True)
        else:
            raise RuntimeError("No wheel file found after maturin build")
        
        # Download initial dataset shards
        print("\n📥 Downloading initial dataset shards (8 shards, ~800MB)...")
        subprocess.run(["python", "-m", "nanochat.dataset", "-n", "8"], check=True)
        
        # Start downloading more shards in background
        print("\n📥 Starting background download of full dataset (240 shards, ~24GB)...")
        download_proc = subprocess.Popen(["python", "-m", "nanochat.dataset", "-n", "240"])
        
        # Train the tokenizer
        print("\n🏋️ Training tokenizer on ~2B characters...")
        subprocess.run(
            ["python", "-m", "scripts.tok_train", "--max_chars=2000000000"],
            check=True
        )
        
        # Evaluate the tokenizer
        print("\n📊 Evaluating tokenizer...")
        subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)
        
        # Wait for dataset download to complete
        print("\n⏳ Waiting for dataset download to complete...")
        download_proc.wait()
        
        print("\n✅ Tokenizer built successfully!")
    else:
        print("\n✅ Tokenizer found, skipping build")
        # Still make sure we have enough data shards (240 for d20 model)
        print("\n📥 Ensuring dataset is downloaded (240 shards, ~24GB)...")
        subprocess.run(["python", "-m", "nanochat.dataset", "-n", "240"], check=True)
    
    # Download evaluation bundle
    print("\n📥 Downloading evaluation bundle...")
    eval_bundle_dir = f"{os.environ['NANOCHAT_BASE_DIR']}/eval_bundle"
    if not os.path.exists(eval_bundle_dir):
        subprocess.run([
            "wget", "-q", 
            "https://huggingface.co/datasets/karpathy/nanochat-eval/resolve/main/eval_bundle.zip"
        ], check=True)
        subprocess.run(["unzip", "-q", "eval_bundle.zip"], check=True)
        subprocess.run(["mv", "eval_bundle", eval_bundle_dir], check=True)
        subprocess.run(["rm", "eval_bundle.zip"], check=True)
    
    # Pretrain base model
    print("\n" + "=" * 80)
    print("📚 STAGE 1: Pretraining base model (d20, 561M params)")
    print("=" * 80)
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.base_train", "--",
        f"--depth=20",
        f"--run={wandb_run}"
    ], check=True)
    
    print("\n📊 Evaluating base model loss...")
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.base_loss"
    ], check=True)
    
    print("\n📊 Evaluating base model on CORE tasks...")
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.base_eval"
    ], check=True)
    
    # Midtraining (uses identity_conversations.jsonl)
    print("\n" + "=" * 80)
    print("🎯 STAGE 2: Midtraining (conversation tokens, tool use, IDENTITY)")
    print("=" * 80)
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.mid_train", "--", f"--run={wandb_run}"
    ], check=True)
    
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.chat_eval", "--", "-i", "mid"
    ], check=True)
    
    # Supervised Finetuning (uses identity_conversations.jsonl)
    print("\n" + "=" * 80)
    print("✨ STAGE 3: Supervised Finetuning (with IDENTITY)")
    print("=" * 80)
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.chat_sft", "--", f"--run={wandb_run}"
    ], check=True)
    
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.chat_eval", "--", "-i", "sft"
    ], check=True)
    
    # Generate final report (skip if it fails - not critical)
    print("\n📊 Generating final report...")
    try:
        subprocess.run(["python", "-m", "nanochat.report", "generate"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Report generation failed (non-critical), continuing...")
    
    # Commit volumes to persist checkpoints
    data_volume.commit()
    checkpoint_volume.commit()
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "=" * 80)
    print("🎉 Training Complete with Custom Identity!")
    print("=" * 80)
    print(f"⏱️  Total time: {hours}h {minutes}m")
    print(f"💾 Checkpoints saved to volume: nanochat-checkpoints")
    print(f"🤖 Your model now knows it's smarternano created by Eddie Chen!")
    
    return f"Training completed in {hours}h {minutes}m"


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        "/root/.cache/nanochat": data_volume,
        "/root/checkpoints": checkpoint_volume,
    },
    timeout=60 * 60 * 2,  # 2 hour timeout for SFT+Mid
)
def retrain_identity(wandb_run: str = "dummy", skip_mid: bool = False):
    """
    Retrain ONLY the identity-aware stages (midtraining + SFT) using existing base checkpoint.
    
    This is much faster and cheaper than full retraining:
    - Reuses your existing base pretrained weights
    - Only retrains midtraining + SFT with new identity data
    - Time: ~1 hour (or ~30 min if skip_mid=True)
    - Cost: ~$24 (or ~$12 if skip_mid=True)
    
    Args:
        skip_mid: If True, only retrain SFT (fastest, cheapest)
    """
    import os
    import subprocess
    
    start_time = time.time()
    
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    os.environ["WANDB_MODE"] = "disabled" if wandb_run == "dummy" else "online"
    
    print("=" * 80)
    print("🔄 Retraining Identity Stages Only")
    print("=" * 80)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💡 Reusing existing base checkpoint")
    
    # Verify identity conversations exist
    identity_path = "/root/.cache/nanochat/identity_conversations.jsonl"
    if os.path.exists(identity_path):
        with open(identity_path, 'r') as f:
            num_conversations = sum(1 for _ in f)
        print(f"✅ Found {num_conversations} custom identity conversations")
    else:
        raise FileNotFoundError("No identity_conversations.jsonl found! Upload it first.")
    
    # Verify base checkpoint exists
    base_checkpoint_dir = "/root/.cache/nanochat/base_checkpoints"
    if not os.path.exists(base_checkpoint_dir):
        raise FileNotFoundError(
            "No base checkpoint found! Run full training first with: "
            "modal run modal_app.py --command train"
        )
    
    if not skip_mid:
        # Midtraining (uses identity_conversations.jsonl)
        print("\n" + "=" * 80)
        print("🎯 STAGE 1: Midtraining (with IDENTITY)")
        print("=" * 80)
        subprocess.run([
            "torchrun", "--standalone", "--nproc_per_node=8",
            "-m", "scripts.mid_train", "--", f"--run={wandb_run}"
        ], check=True)
        
        subprocess.run([
            "torchrun", "--standalone", "--nproc_per_node=8",
            "-m", "scripts.chat_eval", "--", "-i", "mid"
        ], check=True)
    else:
        print("\n⏭️  Skipping midtraining (using existing checkpoint)")
    
    # Supervised Finetuning (uses identity_conversations.jsonl)
    print("\n" + "=" * 80)
    print("✨ STAGE 2: Supervised Finetuning (with IDENTITY)")
    print("=" * 80)
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.chat_sft", "--", f"--run={wandb_run}"
    ], check=True)
    
    subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=8",
        "-m", "scripts.chat_eval", "--", "-i", "sft"
    ], check=True)
    
    # Commit volumes
    data_volume.commit()
    checkpoint_volume.commit()
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "=" * 80)
    print("🎉 Identity Retraining Complete!")
    print("=" * 80)
    print(f"⏱️  Total time: {hours}h {minutes}m")
    print(f"💾 New checkpoints saved")
    print(f"🤖 Your model now knows it's smarternano!")
    
    return f"Identity retraining completed in {hours}h {minutes}m"


# -----------------------------------------------------------------------------
# Serving Function (Single A10G GPU for inference)
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/root/.cache/nanochat": data_volume,
    },
    scaledown_window=300,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=40) #changed to 40 concurrent inputs 
@modal.asgi_app()
def serve_chat():
    """
    Serve the chat web interface using the original scripts/chat_web.py.
    Loads the trained checkpoint and serves it on a single A10G GPU.
    """
    import sys
    import os
    
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    
    sys.argv = [
        "chat_web.py",
        "--source", "sft",
        "--num-gpus", "1",
        "--temperature", "0.8",
        "--top-k", "50",
        "--max-tokens", "512",
    ]
    
    sys.path.insert(0, "/root/nanochat")
    from scripts import chat_web
    
    return chat_web.app


# -----------------------------------------------------------------------------
# Upload identity conversations
# -----------------------------------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(python_version="3.10"),
    volumes={
        "/data": data_volume,
    },
    timeout=60 * 5,  # 5 minute timeout
)
def upload_identity_data():
    """
    Upload local identity_conversations.jsonl to Modal's data volume.
    
    Run this after generating synthetic data locally to upload it to Modal.
    """
    import os
    
    # Check if file exists in volume
    target_path = "/data/identity_conversations.jsonl"
    
    if os.path.exists(target_path):
        # Count lines to see how many conversations we have
        with open(target_path, 'r') as f:
            num_conversations = sum(1 for _ in f)
        print(f"✅ Found {num_conversations} conversations in Modal volume")
        return f"Identity data already uploaded ({num_conversations} conversations)"
    else:
        print("❌ No identity_conversations.jsonl found in volume yet")
        print("📝 You need to upload it using Modal CLI:")
        print("   modal volume put nanochat-data identity_conversations.jsonl ~/.cache/nanochat/identity_conversations.jsonl")
        raise FileNotFoundError("Please upload identity_conversations.jsonl first")


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
    
    elif command == "train":
        wandb_run = os.environ.get("WANDB_RUN", "dummy")
        print(f"🚀 Starting full training pipeline with custom identity...")
        print(f"   (This will take ~3-4 hours on 8xH100)")
        print(f"   Your model will learn it's smarternano created by Eddie Chen!")
        result = run_speedrun.remote(wandb_run=wandb_run)
        print(f"\n✅ {result}")
    
    elif command == "retrain-identity":
        wandb_run = os.environ.get("WANDB_RUN", "dummy")
        print(f"🔄 Retraining ONLY midtraining + SFT with new identity...")
        print(f"   (This will take ~1 hour on 8xH100)")
        print(f"   Reusing existing base checkpoint - saves time & money!")
        result = retrain_identity.remote(wandb_run=wandb_run, skip_mid=False)
        print(f"\n✅ {result}")
    
    elif command == "retrain-sft":
        wandb_run = os.environ.get("WANDB_RUN", "dummy")
        print(f"⚡ Retraining ONLY SFT with new identity...")
        print(f"   (This will take ~30 min on 8xH100)")
        print(f"   Reusing existing mid checkpoint - fastest & cheapest!")
        result = retrain_identity.remote(wandb_run=wandb_run, skip_mid=True)
        print(f"\n✅ {result}")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands:")
        print("  test              - Test GPU setup")
        print("  train             - Full training (~4h, ~$96)")
        print("  retrain-identity  - Retrain mid+SFT (~1h, ~$24)")
        print("  retrain-sft       - Retrain SFT only (~30min, ~$12)")
        print("\nTo deploy the chat interface, use:")
        print("  modal deploy modal_app.py")

