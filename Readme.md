# Inference-only implementation of LLaMA in plain NumPy

It uses NumPy, so it can run without a GPU. It also uses memory mapped files to load the weights, so you can run it with little memory.

Inspired by [picoGPT].

[picoGPT]: https://github.com/jaymody/picoGPT

Besides NumPy, it currently also has a dependency on Google's [SentencePiece] for tokenization.

[SentencePiece]: https://github.com/google/sentencepiece
