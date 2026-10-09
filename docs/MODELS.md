# Models and external dependencies

The source checkout includes the inverse-CDF LUT and SP1 source/lockfile.
It does not bundle pretrained diffusion models, metric models, or baseline
checkpoints. Each externally obtained model/codebase retains its own license.

| Purpose | Identifier or source |
|---|---|
| Default SD2.1 generation/recovery | `Manojb/stable-diffusion-2-1-base` on Hugging Face |
| SDXL generation/recovery | `stabilityai/stable-diffusion-xl-base-1.0` on Hugging Face |
| CLIP evaluation | `openai/clip-vit-base-patch32` on Hugging Face |
| KID features | torchvision `Inception_V3_Weights.DEFAULT` |
| BRISQUE | pyiqa `brisque` model |
| SP1 toolchain and GPU prover setup | [SP1 documentation](https://docs.succinct.xyz/) |

Diffusers/transformers load the declared model identifiers with their normal
`from_pretrained` download/cache mechanism. Complete any access requirements on
the model host, and use the same checkpoint and configuration for generation and
recovery. A local model directory can also be supplied through `--model-id` and
the corresponding verifier configuration.

Record the model revision and software environment with each evaluation run.

## Comparison methods

Generate comparison images using the respective official implementation with
the same prompt/seed pairs across methods. Supply their paths through the
manifest format in the [evaluation guide](REPRODUCE.md).

- [Tree-Ring](https://github.com/YuxinWenRick/tree-ring-watermark)
- [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading)
- [RingID](https://github.com/showlab/RingID)
- [Stable Signature / HiDDeN components](https://github.com/facebookresearch/stable_signature)
- [TrustMark](https://github.com/adobe/trustmark)

Use each baseline's own model setup and license. Images from additional methods
can be supplied through the same manifest interface.
