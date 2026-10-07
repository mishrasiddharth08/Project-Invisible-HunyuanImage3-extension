# Validation — October 7, 2026

## Passed

- Standalone checks cover checkpoint ownership, wrapper chaining, API overrides, friendly preset label, native defaults and component refresh, six VRAM recipes, cache/RAM-pressure release, variant matching, safetensors truncation rejection, sampling contracts, adapters, mapped GGUF operations, JSON transport, sequential batch seeds, sample saving and simulated Stop paths. All 119 tests passed in the publication check.
- Real isolated worker starts using this Forge installation's Python, PyTorch 2.13.0+cu130, RTX 5090, comfy-kitchen 0.2.37 and comfy-aimdo 0.5.5. Dynamic VRAM initializes successfully. No model weights are loaded by this check.
- Python source compilation.
- Real Gradio controls and preset event bindings build successfully with simulated Forge interfaces. The sandbox blocked Gradio's event-loop socket; the same serverless check passed outside the sandbox. This is not browser acceptance.

## Real human portrait generation

October 7, 2026: three real Forge API requests passed using Instruct-Distil W4A8 and the fp16 VAE, Euler/Simple, 8 steps, CFG 1, distilled guidance 2.5, Spectrum and rewriting off. Native sample saving, returned images, seeds, dimensions and embedded model/settings metadata passed. Model files were already present; this acceptance run downloaded nothing.

| Image | Size | Seed | Request time | Visual inspection |
|---|---|---|---|---|
| Face | 1024 × 1024 | 710701 | 161.15 s | Coherent facial features, clear eyes, convincing skin texture; some smoothing |
| Face and hands | 1024 × 1536 | 710702 | 101.11 s | Five visible digits per hand, plausible pose; fingertips softer than face |
| Full body | 1024 × 1536 | 710703 | 96.23 s | Complete person and shoes, plausible proportions; smaller face and hands have less detail |

These are three subjective visual checks, not a general quality benchmark. The worker closes after every request; times include loading and saving and are not a persistent-worker speed comparison. 1024 × 1536 exceeds the model's nominal one-megapixel training table; the upstream encoder emitted its size warning. These particular outputs remained coherent.

Outputs and detailed metadata: `work/human-quality-2026-10-07/report.json` and the three PNGs in that folder. Images were rendered directly at the listed sizes; Forge Hires fix and upscaling were not used.

## Limits

Initial installation found no compatible weights. They were present for the portrait acceptance above. Base, full Instruct, int8/BF16, Spectrum and browser/gallery acceptance remain unverified. Later isolated-worker checks below cover partial editing, rewriting, reference slots, sampling cancellation and recovery; editing quality remains unresolved. Mock tests are wiring evidence only.

## Performance, memory and adapter acceptance

Existing Instruct-Distil W4A8/fp16 VAE, RTX 5090, 8 Euler/Simple steps, Spectrum off, same portrait prompt and seed 710707:

| Check | Size | Request time | Sampled total GPU peak | Worker RAM peak |
|---|---|---|---|---|
| Speed cold | 1024² | 98.803 s | 31.816 GiB | 57.716 GiB |
| Speed cached | 1024² | 21.060 s | 31.798 GiB | 57.843 GiB |
| Synthetic LoRA | 256² | 88.784 s | 31.508 GiB | 57.417 GiB |
| Synthetic LoKr | 256² | 72.222 s | 31.505 GiB | 57.439 GiB |
| Tiled VAE, Speed | 1024² | 76.377 s | 31.484 GiB | 57.457 GiB |
| Low memory + tiled VAE | 1024² | 91.024 s | 8.165 GiB | 55.458 GiB |

Cold and cached images were pixel-identical. Tiled images from Speed and Low memory were pixel-identical. Tiled versus untiled had a mean absolute channel error of 1.022/255 and remained visually coherent. LoRA/LoKr fixtures targeted one nonquantized dense layer in the quantized model; they prove loader/patch/sampler execution, not learned-adapter quality or every quantized adapter target. Adapter changes currently invalidate GPU residency and include that reload in their request times.

GPU figures are NVML total-device samples every 100 ms and include desktop/other GPU users. Initial GPU baseline was about 1.55 GiB, so the Low-memory request added about 6.61 GiB above that baseline. This is not an exact model-only allocation accounting. RAM is sampled worker RSS. Torch allocator peaks are separate because aimdo uses additional allocations. High-RAM use is still inherent to this 80B model.

Detailed results: `work/performance-2026-10-07/report.json`, `low-memory-report.json`, and `image-consistency.json`. No model weights were downloaded for these tests. Format/adapter CPU tests do not establish full-model GGUF, BF16, INT8, FP8, NVFP4 or learned LoCon/LoHa/DoRA/OFT/BOFT image quality.

The Reddit reference returned HTTP 403. The upstream GitHub source, comparison site, all three Hugging Face model cards, user's public repository listing and Forge Neo reference were accessible. Revisions are pinned in `vendor/revisions.json`.

Reproduce with Forge's Python: `tools/test.py`, `tools/check_worker.py`, `tools/check_ui.py`.

## VRAM profile acceptance

1024-square W4A8 generation on RTX 5090, software budgets only:

| Selected profile | Result | Request time | Total GPU peak |
|---|---|---|---|
| 8 GB | Passed | 92.607 s | 5.583 GiB |
| 10 GB | Passed | 125.739 s | 7.489 GiB |
| 12 GB | Passed | 138.408 s | 8.545 GiB |
| 16 GB | Failed | — | HostBuffer.read_file_slice failure |
| 24/32 GB explicit | Not completed | — | Unverified |

The final 16 GB retry failed with an idle GPU baseline of 1.55 GiB; contention alone does not explain this failure. Use Low memory or the tested 12 GB profile while this transfer failure is investigated. Speed/Automatic on the physical 32 GB card passed separately. No physical 8/10/12/16/24 GB card was tested. Smaller VRAM does not remove the roughly 55–58 GiB system RAM requirement.

## Later isolated-worker editing checks — October 7, 2026

On an RTX 5090 with Instruct-Distil W4A8, the matching SigLIP2 vision tower and fp16 VAE, a 1024 x 1024 red-cup reference was edited using seed 72026, eight Euler/Simple steps, CFG 1, guidance 2.5, and Spectrum and rewriting disabled. The requested blue colour appeared, but the image gained excessive contrast and texture. A second colour edit at guidance 1.0 and a keep-unchanged instruction at guidance 2.5 also showed this artifact. Lower guidance is therefore not a demonstrated remedy, and editing quality remains unresolved.

A VAE-only encode/decode roundtrip preserved the source closely (mean absolute pixel difference about 0.66/255), without the excessive texture. This narrows the problem to conditional generation; it does not establish that the model or quantization is responsible.

Nine bundled upstream files were compared semantically with `PedroMarinhoDev/ComfyUI-HunyuanImage3` revision `84ad3a3e2a54472e69e195269729f774b242d120`: nodes, latent format, pipeline, tokenizer, model, model base, loader, operations and VAE matched. The tested safetensors route uses that upstream loader and its supported editing settings. No reproducible implementation difference was identified in those paths. An INT8/BF16 comparison has not been run, so a format-related numerical limitation cannot yet be separated from shared upstream or model behavior.

The current CPU-only suite ran 119 tests: 110 passed and nine optional mapped-GGUF tests were skipped because their `gguf==0.19.0` dependency was unavailable in that test environment. This is not full-model GGUF validation. Functional isolated-worker checks also covered generation, rewriting, reference slots, sampling cancellation and recovery; the three-reference test duplicated one input and does not certify fusion of three distinct images. These checks do not establish full browser acceptance or intermediate latent previews.

The runtime fixes for singleton-frame VAE output and safe hook ownership have retained regressions. No new model downloads, dependency changes or runtime patches were made for the editing-quality diagnosis. The artifact remains an open limitation, not a passing quality claim.
