# Design Notes

PostTrain Garden is built around one small invariant: every model failure should
become useful training or evaluation signal.

## Loop

1. A bad response is stored as a regret.
2. A regret is converted into a preference pair.
3. A judge scores the pair.
4. Good pairs are distilled into chat-style SFT examples.
5. Training commands consume either SFT examples or preference pairs.
6. Evaluation and export summarize what changed and what remains risky.

## Dependency boundary

Core commands use only the standard library. Training commands import Hugging
Face libraries lazily so the repository remains inspectable and testable on
machines without GPU or ML packages.
