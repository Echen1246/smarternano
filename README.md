# smarternano

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy - now on Modal.

This is a fork of [Karpathy's nanochat](https://github.com/karpathy/nanochat) adapted to run on [Modal](https://modal.com) infrastructure. It's a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT.

**Key difference:** Instead of running on a dedicated Lambda/bare-metal GPU node, this uses Modal's serverless infrastructure with:
- **Training:** 8xH100 GPUs (on-demand, ~4 hours, ~$96)
- **Serving:** 1xA10G GPU (auto-scaling, $1.10/hr when active)
- **Storage:** Persistent Modal Volumes for checkpoints

## Quick start

### Prerequisites

1. **Install Modal CLI:**
```bash
pip install modal
modal setup  # This will authenticate you
```

2. **Clone this repo:**
```bash
git clone https://github.com/Echen1246/smarternano.git
cd smarternano
```

### Training on Modal (8xH100)

Run the full speedrun training pipeline (pretraining → midtraining → SFT):

```bash
# Run in detached mode so training continues even if you disconnect
modal run --detach modal_app.py --command train
```

This will:
- Spin up 8xH100 GPUs on Modal
- Download FineWeb-Edu dataset (~24GB)
- Build and train the tokenizer
- Run full training pipeline (~3-4 hours)
- Save checkpoints to Modal Volume
- Auto-stop when complete

**Cost:** ~$96 for the full speedrun (4 hours × 8xH100 @ $24/hr)

**Monitor progress:**
```bash
modal app logs nanochat  # View training logs
```

### Serving on Modal (1xA10G)

Deploy the chat interface (uses your trained checkpoint):

```bash
modal deploy modal_app.py
```

This will:
- Deploy the chat web UI on Modal
- Load your trained SFT checkpoint
- Serve on 1xA10G GPU (much cheaper for inference!)
- Auto-scale to 0 when idle (saves money)
- Give you a public URL

**Access your chat:** Visit the URL shown (e.g., `https://yourapp--serve-chat.modal.run`)

**Cost when serving:** 
- Idle: $0/hr (auto-scales to 0)
- Active: $1.10/hr (only when users are chatting)

### Modal Architecture

```
┌─────────────────────────────────────┐
│   Modal Volume: smarternano-data    │
│   (Persistent Cloud Storage)        │
│   - Checkpoints                     │
│   - Tokenizer                       │
│   - Dataset                         │
└─────────────────────────────────────┘
         ↑                    ↑
    (writes)            (reads from)
         │                    │
┌────────┴────────┐    ┌─────┴──────────┐
│  Training       │    │  Serving       │
│  8xH100 GPUs    │    │  1xA10G GPU    │
│  run_speedrun   │    │  serve_chat    │
│  (runs 4hrs)    │    │  (always on)   │
└─────────────────┘    └────────────────┘
```

Basically, all checkpoints are stored in Modal volumes. When we decide to serve, checkpoints are loaded, and we can use a cheaper GPU to chat with.Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :). Currently, it's not architecturally different from original nanochat, but I plan to make some changes to training or architecture to try and make it more efficient.

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

After training completes, you can view the performance metrics for your trained model. The training outputs a "report card" with evaluations and metrics. Here's an example from a successful run:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, there are two more scales of interest:
- **~$300 tier:** d26 model (depth=26, ~12 hours) - slightly outperforms GPT-2 CORE score
- **$1000 tier:** (~41.6 hours) - nice round number milestone

To train bigger models on Modal, you can modify the training configuration in `scripts/base_train.py` and `scripts/mid_train.py`:

```python
# In scripts/base_train.py, modify:
depth = 26  # Increase from 20 to 26 for GPT-2 size
device_batch_size = 16  # Halve from 32 to avoid OOM

# Make sure to download more data shards:
# Modal will handle this automatically, but you may need to update
# the dataset download in modal_app.py if training for longer
```

The biggest thing to pay attention to is managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

### About Modal GPU options:

- **8xH100:** Current setup, recommended for training
- **8xA100:** Works fine but ~30% slower, cheaper option
- **Single GPU:** You can modify `modal_app.py` to use `gpu="H100"` (single), but training will take 8x longer
- **Memory:** If you get OOM errors, reduce `device_batch_size` in the training scripts from 32 → 16 → 8 → 4

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## Contributing

nanochat is nowhere finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

## Acknowledgements

- **This fork builds on [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat)** - all credit for the original architecture, training pipeline, and brilliant simplicity goes to Karpathy.
- The name (nanochat) derives from Karpathy's earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard.
- Thank you to [HuggingFace](https://huggingface.co/) for FineWeb-Edu and Smoltalk datasets.
- Thank you to [Modal](https://modal.com) for making serverless GPU infrastructure accessible.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance (to Karpathy's original project).

### Modal Adaptation

This fork adapts nanochat to run on Modal's serverless infrastructure:
- Training uses Modal's on-demand 8xH100 GPUs
- Serving uses Modal's auto-scaling 1xA10G GPU
- Persistent storage via Modal Volumes
- All original nanochat functionality preserved

## Cite

If you find the original nanochat helpful in your research, cite Karpathy's work:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

If you're using this Modal adaptation:

```bibtex
@misc{smarternano,
  author = {Eddie Chen},
  title = {smarternano: nanochat on Modal serverless infrastructure},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Echen1246/smarternano},
  note = {Fork of Andrej Karpathy's nanochat}
}
```

## License

MIT
