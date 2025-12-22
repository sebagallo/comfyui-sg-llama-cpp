# comfyui-sg-llama-cpp

ComfyUI custom node that acts as a llama-cpp-python wrapper, with support for vision models. It
allows the user to generate text responses from prompts using llama.cpp.

![Screenshot](assets/node_preview.png)

## Features

- Load and use GGUF models (including vision models)
- Generate text prompts using llama.cpp
- Support for multi-modal inputs (images)
- Memory management options
- Integration with ComfyUI workflows

## Installation

1. Install the required dependency/wheel from:
   ```
   https://github.com/JamePeng/llama-cpp-python/releases
   ```

2. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/sebagallo/comfyui-sg-llama-cpp
   ```

3. Restart ComfyUI.

## Node Reference

### LlamaCPPModelLoader
Loads GGUF model files and prepares them for use.

**Inputs**
- **Required**:
  - `model_name`: Select the GGUF model file to load.
- **Optional**:
  - `chat_format`: Chat template to use (default: `llama-2`).
  - `mmproj_model_name`: Multi-modal projector model for vision (default: `None`).

**Outputs**
- `model`: The loaded Llama model object.

### LlamaCPPOptions
Configures advanced parameters for the model.

**Inputs**
- **Optional**:
  - `n_gpu_layers`: Number of layers to offload to GPU (default: `-1` for all).
  - `n_ctx`: Context window size (default: `2048`).
  - `n_threads`: CPU threads to use (default: `-1` for auto).
  - `n_threads_batch`: Threads for batch processing (default: `-1` for auto).
  - `n_batch`: Batch size (default: `512`).
  - `n_ubatch`: Micro-batch size (default: `512`).
  - `main_gpu`: Main GPU ID (default: `0`).
  - `offload_kqv`: Offload K/Q/V to GPU (default: `Enabled`).
  - `numa`: NUMA support (default: `Disabled`).
  - `use_mmap`: Memory mapping (default: `Enabled`).
  - `use_mlock`: Memory locking (default: `Disabled`).
  - `verbose`: Verbose logging (default: `Disabled`).

**Outputs**
- `options`: A configuration dictionary.

### LlamaCPPEngine
The main generation node.

**Inputs**
- **Required**:
  - `model`: The model from `LlamaCPPModelLoader`.
  - `prompt`: The text prompt.
- **Optional**:
  - `image`: Input image for vision models.
  - `options`: Options from `LlamaCPPOptions`.
  - `system_prompt`: System instruction (default: empty).
  - `memory_cleanup`: Strategy to clean memory after generation (default: `close`).
  - `response_format`: `text` or `json_object` (default: `text`).
  - `max_tokens`: Max new tokens (default: `512`).
  - `temperature`: Randomness (default: `0.2`).
  - `top_p`: Nucleus sampling (default: `0.95`).
  - `top_k`: Top-k sampling (default: `100`).
  - `repeat_penalty`: Penalty for repetition (default: `1.0`).
  - `seed`: Random seed (default: `-1`).

**Outputs**
- `response`: The generated text.

### LlamaCPPMemoryCleanup
Utility to manually free resources.

**Inputs**
- **Required**:
  - `memory_cleanup`: Cleanup mode (`close`, `backend_free`, `full_cleanup`, `persistent`).
- **Optional**:
  - `passthrough`: Any input to pass through (allows chaining).

**Outputs**
- `passthrough`: The input passed through unmodified.

## Custom Model Folders

By default, the node loads GGUF models from ComfyUI's `text_encoders` folder. You can optionally specify additional folders to load models from by creating a `config.json` file in the custom nodes directory.

### Configuration

1. Create a file named `config.json` in the same directory as this README
2. Add your custom model folders in the following JSON format:

```json
{
  "model_folders": [
    "C:\\Users\\YourUsername\\models",
    "D:\\AI\\LLM\\models",
    "/home/user/models"
  ]
}
```

### Notes

- The `config.json` file is **optional** - the node works without it
- Paths can be absolute or relative
- Both Windows (`C:\`) and Unix (`/`) style paths are supported
- Non-existent paths are automatically filtered out
- Models from all folders (ComfyUI's `text_encoders` + your custom folders) will appear in the model selection dropdown
- See `config.example.json` for additional examples

## Requirements

- llama-cpp-python (from https://github.com/JamePeng/llama-cpp-python)

## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

## Repository

https://github.com/sebagallo/comfyui-sg-llama-cpp
