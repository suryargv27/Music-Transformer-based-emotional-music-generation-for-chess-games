# EMOPIA Conditional Music Generation for Chess

Converts chess positions into emotion-conditioned piano music by combining chess engine evaluation, affective computing, and Transformer-based symbolic music generation.

## Overview

This project maps chess positions to emotional states using Stockfish evaluation, then generates piano music conditioned on that emotion using a Transformer model trained on the EMOPIA dataset.

```
Chess Position → Emotion Mapping → Emotion Token → Transformer Model → Generated MIDI Music
```

## How It Works

**1. Chess → Emotion**
A chess position (FEN) is evaluated using Stockfish. Evaluation score, tactical activity (captures, checks), and mobility are combined into a **valence** (positive/negative) and **arousal** (high/low energy) score based on Russell's Valence–Arousal Model, then mapped into one of four emotional quadrants:

| Quadrant | Emotion |
|----------|---------|
| Q1 | High Valence + High Arousal |
| Q2 | Low Valence + High Arousal |
| Q3 | Low Valence + Low Arousal |
| Q4 | High Valence + Low Arousal |

**2. Emotion → Music**
The emotion quadrant is used as a conditioning token for an autoregressive Transformer trained on the [EMOPIA dataset](https://annahung31.github.io/EMOPIA/) (1,087 piano clips across 387 songs), using REMI tokenization to represent pitch, duration, velocity, and timing as transformer-readable tokens.

## Dataset

- **EMOPIA**: multi-modal piano emotion dataset with clip-level emotion annotations
- 876 / 114 / 88 song split for train / val / test
- Tokenized with `miditok`'s REMI representation (beat-based symbolic encoding)

## Model

- Custom `MusicTransformer`: token + positional embeddings, causal self-attention encoder stack, linear output head
- ~26M parameters (d_model=512, 8 heads, 8 layers)
- Trained with AdamW, cosine annealing LR, gradient clipping, and early stopping
- Optimized via cross-entropy loss on next-token prediction

## Generation

Sampling uses temperature scaling, top-k, top-p (nucleus) sampling, and a repetition penalty to produce diverse, non-repetitive output conditioned on the target emotion quadrant.

## Setup

```bash
pip install miditok symusic pretty_midi python-chess stockfish cairosvg chess pygame fluidsynth
apt-get install -y stockfish fluidsynth
```

## Usage

```python
fen = "6k1/5ppp/8/8/3Q4/8/4KPPP/8 w - - 0 1"

result = generate_music_from_chess(
    fen,
    max_new_tokens=768,
    temperature=0.85,
    top_k=20,
    top_p=0.92,
    repetition_penalty=1.2,
    output_path="output.mid",
)
```

## Evaluation

- **Objective**: training/validation loss, perplexity, token diversity (`len(set(tokens)) / len(tokens)`)
- **Emotional accuracy**: valence–arousal scatter plots across a range of tactical, endgame, and positional test positions to confirm quadrant separation
- Final validation loss ~2.56 (perplexity ~12.9)

## Related Work

Builds on ideas from the [EMOPIA paper](https://annahung31.github.io/EMOPIA/) (Hung et al.), [Music Transformer](https://arxiv.org/abs/1809.04281) (Huang et al., Google Magenta), OpenAI's MuseNet, and the [REMI tokenization scheme](https://arxiv.org/abs/2002.00212) (Huang & Yang). This project's contribution is deriving emotion conditioning dynamically from chess positions rather than from predefined human-assigned labels.

## Limitations

- Small dataset relative to large-scale pretrained music models
- Heuristic (not learned) chess-to-emotion mapping
- No human listening evaluation
- Limited long-term musical structure

## Future Work

- Learned chess-position embeddings
- Continuous (rather than quadrant-based) emotion conditioning
- Larger pretrained music transformer backbones
- RLHF-based fine-tuning for musical quality
- Real-time interactive generation
