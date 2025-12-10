# comfyui-sg-llama-cpp

ComfyUI custom node that acts as a llama-cpp-python wrapper, with support for vision models. It
allows the user to generate text responses from prompts using llama.cpp.

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

## Usage

This custom node provides four main components:

### LlamaCPPModelLoader
- Loads GGUF model files from the `text_encoders` folder
- Supports various chat formats
- Optional multi-modal projector support for vision models

### LlamaCPPOptions
- Configures model parameters like:
  - GPU layers
  - Context window size
  - Thread count
  - Batch sizes

### LlamaCPPEngine
- Generates text responses from prompts
- Supports vision inputs for compatible models
- Configurable generation parameters (temperature, top_p, etc.)
- Memory cleanup options

### LlamaCPPMemoryCleanup
- Manually manages memory usage
- Various cleanup modes available

## Requirements

- ComfyUI
- llama-cpp-python (from https://github.com/JamePeng/llama-cpp-python)

## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

## Repository

https://github.com/sebagallo/comfyui-sg-llama-cpp
