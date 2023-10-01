# `llm-serve`

An LLM server that responds to TCP clients.

## Build instructions

If building with cuBLAS, ensure CUDA is installed.
If building with CLBlast, ensure it is installed.

Then, invoke `cargo` as such:

```
  cargo build [--release] --features [cublas, clblast]
```

Building with both cuBLAS and CLBlast at once is not supported.

## Usage instructions

`llm-serve` can only read GGML models until GGUF support is [merged](https://github.com/rustformers/llm/issues/365).

2 files are required:
  - A GGML model, and
  - A prompt template file.

GGML models can be found on [HuggingFace](https://huggingface.co), for example [TheBloke's Wizard-Vicuna 7B model](https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML).

A prompt template file is a text file that is formatted similar to this:

```
  USER: {{prompt}}
  ASSISTANT: 
```

`{{prompt}}` is a special token used by `llm-serve` to format user prompts.
Refer to your model's model card for the prompt template.

Invoke `llm-serve` as such:

```
  llm-serve -m /path/to/model -a <model_arch> -T /path/to/template [-b <batch_size>] [-t <thread_count>] [-g <gpu_offload_layers>] [-p <listening_port>]
```

where `model_arch` is one of the lowercase names defined [here](https://github.com/rustformers/llm/blob/main/crates/llm/src/lib.rs#L174).
