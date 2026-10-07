# Changelog

## 0.1.0 — October 7, 2026 (experimental)

- Native HunyuanImage 3.0 preset, checkpoint/component selection, Generate, API and sample saving.
- Isolated pinned backend with explicit dependency installation and no generation-time downloads.
- Cached model loading, idle/RAM-pressure release, tiled VAE and six VRAM profile policies.
- Quantization preflight and bounded adapter operations; experimental native-name GGUF support.
- 119 passing standalone tests; real W4A8 portraits and synthetic dense LoRA/LoKr generation.
- 8/10/12 GB budget simulations passed on RTX 5090; 16 GB currently fails a host-buffer transfer. 24/32 GB explicit profiles and physical smaller cards remain unverified. Speed/Automatic succeeded on the actual 32 GB card.
