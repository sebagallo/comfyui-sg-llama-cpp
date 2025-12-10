import os
import gc
import torch
import base64
import io
from PIL import Image
from llama_cpp import Llama, llama_backend_free
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler,
    Llava16ChatHandler,
    ObsidianChatHandler,
    MoondreamChatHandler,
    NanoLlavaChatHandler,
    Llama3VisionAlphaChatHandler,
    MiniCPMv26ChatHandler,
    Gemma3ChatHandler,
    Qwen25VLChatHandler,
    Qwen3VLChatHandler,
)
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from typing import Dict, Any, List, Type

# Global LLM instance for persistence
_global_llm = None

def _convert_image_to_data_uri(image_tensor: torch.Tensor) -> str:
    """Convert a ComfyUI image tensor to a base64 data URI for vision models."""
    try:
        # ComfyUI images are typically [B, H, W, C] with values in [0, 1] range
        # Take the first image if batched
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Remove batch dimension

        # Ensure tensor is in CPU and convert to PIL
        image_tensor = image_tensor.cpu()

        # Convert from [0, 1] to [0, 255] and to uint8
        if image_tensor.dtype != torch.uint8:
            image_tensor = (image_tensor * 255).clamp(0, 255).to(torch.uint8)

        # Convert to PIL Image (assuming RGB)
        if image_tensor.shape[2] == 3:  # RGB
            pil_image = Image.fromarray(image_tensor.numpy(), mode='RGB')
        elif image_tensor.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(image_tensor.numpy(), mode='RGBA')
        else:
            raise ValueError(f"Unsupported image channels: {image_tensor.shape[2]}")

        # Convert to JPEG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # Encode to base64 and create data URI
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{base64_string}"

        # Clean up PIL Image and BytesIO buffer to prevent memory leaks
        pil_image.close()
        buffer.close()

        return data_uri

    except Exception as e:
        raise ValueError(f"Failed to convert image tensor to data URI: {str(e)}")

def _cleanup_global_llm(mode: str):
    """Helper function to cleanup the global LLM based on mode."""
    global _global_llm
    if mode == "persistent":
        return  # No cleanup

    # Common cleanup for all non-persistent modes
    if _global_llm is not None:
        # Clean up chat_handler if it exists (for vision models)
        if _global_llm.chat_handler is not None:
            try:
                _global_llm.chat_handler.close()
            except Exception:
                pass  # Ignore cleanup errors for chat_handler
            del _global_llm.chat_handler
            _global_llm.chat_handler = None
            gc.collect()
        try:
            _global_llm.close()
        except AttributeError:
            pass  # close() method may not be available in all versions
        del _global_llm
        _global_llm = None
        gc.collect()

    # Backend free cleanup for backend_free and full_cleanup modes
    if mode in ["backend_free", "full_cleanup"]:
        try:
            llama_backend_free()
            gc.collect()
        except (NameError, AttributeError):
            pass  # llama_backend_free may not be available

    # Full torch cleanup for full_cleanup mode
    if mode == "full_cleanup":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()


# Mapping of chat formats to vision chat handlers
VISION_HANDLERS: Dict[str, Type] = {
    "vision-llava15": Llava15ChatHandler,
    "vision-llava16": Llava16ChatHandler,
    "vision-obsidian": ObsidianChatHandler,
    "vision-moondream": MoondreamChatHandler,
    "vision-nanollava": NanoLlavaChatHandler,
    "vision-llama3visionalpha": Llama3VisionAlphaChatHandler,
    "vision-minicpmv26": MiniCPMv26ChatHandler,
    "vision-gemma3": Gemma3ChatHandler,
    "vision-qwen25vl": Qwen25VLChatHandler,
    "vision-qwen3vl": Qwen3VLChatHandler,
}


class LlamaCPPModelLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        # Manually scan text_encoders folders for GGUF files since folder_paths filters by extensions
        try:
            folders = folder_paths.get_folder_paths("text_encoders")
            model_list = []
            for folder in folders:
                if os.path.exists(folder):
                    files = os.listdir(folder)
                    model_list.extend([f for f in files if f.lower().endswith('.gguf')])
        except:
            model_list = []  # Fallback if folder doesn't exist

        # Filter models based on criteria (case-insensitive)
        model_name_list = [f for f in model_list if 'mmproj' not in f.lower() and 'draft' not in f.lower()]
        mmproj_list = [f for f in model_list if 'mmproj' in f.lower()]

        # Add "None" option to make mmproj truly optional
        if mmproj_list:
            mmproj_list.insert(0, "None")
        else:
            mmproj_list = ["None"]

        # Available chat formats from llama-cpp-python
        base_chat_formats = [
            "llama-2", "llama-3", "llama-4", "alpaca", "qwen", "vicuna", "oasst_llama", "baichuan-2", "baichuan",
            "openbuddy", "redpajama-incite", "snoozy", "phind", "intel", "open-orca", "mistrallite", "zephyr",
            "pygmalion", "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma", "functionary",
            "functionary-v2", "functionary-v1", "chatml-function-calling",
        ]
        vision_formats = list(VISION_HANDLERS.keys())
        chat_formats = sorted(base_chat_formats + vision_formats)

        return {
            "required": {
                "model_name": (model_name_list if model_name_list else ["No GGUF models found"], {"tooltip": "Select GGUF model file"}),
            },
            "optional": {
                "chat_format": (chat_formats, {"default": "llama-2", "tooltip": "Chat format template"}),
                "mmproj_model_name": (mmproj_list, {"default": "None", "tooltip": "Multi-modal projector model for vision (select 'None' to disable)"}),
            }
        }

    RETURN_TYPES = ("LLAMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LlamaCPP"

    def load_model(self, model_name: str, chat_format: str = "llama-2", mmproj_model_name: str = "None") -> tuple:
        try:
            # Get full path to model
            model_path = folder_paths.get_full_path("text_encoders", model_name)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if not model_name.lower().endswith('.gguf'):
                raise ValueError(f"Selected file is not a GGUF model: {model_name}")

            model_info = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Handle mmproj if provided and not "None"
            if mmproj_model_name and mmproj_model_name != "None":
                mmproj_model_path = folder_paths.get_full_path("text_encoders", mmproj_model_name)
                if not os.path.exists(mmproj_model_path):
                    raise FileNotFoundError(f"Multi-modal projector model not found: {mmproj_model_path}")
                model_info["mmproj_model_path"] = mmproj_model_path

            # Return model info dict
            return (model_info,)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


class LlamaCPPOptions(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "optional": {
                "n_gpu_layers": ("INT", {"default": 0, "min": -1, "max": 100, "tooltip": "Number of GPU layers to use"}),
                "n_ctx": ("INT", {"default": 2048, "min": 1, "max": 262144, "tooltip": "Context window size"}),
                "n_threads": ("INT", {"default": -1, "min": -1, "max": 256, "tooltip": "Number of threads (-1 for auto)"}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 16384, "tooltip": "Batch size"}),
                "n_ubatch": ("INT", {"default": 512, "min": 1, "max": 16384, "tooltip": "Micro batch size"}),
            }
        }

    RETURN_TYPES = ("LLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "get_options"
    CATEGORY = "LlamaCPP"

    def get_options(self, **kwargs) -> tuple:
        try:
            # Filter out None/empty values
            options = {k: v for k, v in kwargs.items() if v is not None and v != "" and v != -1}

            return (options,)

        except Exception as e:
            raise RuntimeError(f"Failed to process options: {str(e)}")


class LlamaCPPEngine(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": ("LLAMA_MODEL", {"tooltip": "Loaded Llama model from Model Loader"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "User prompt for chat completion"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input image for vision models"}),
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Options node"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "System prompt"}),
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "close", "tooltip": "Memory cleanup method after generation"}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 262144, "tooltip": "Maximum tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Sampling temperature"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Top-p sampling"}),
                "top_k": ("INT", {"default": 100, "min": 0, "max": 400, "tooltip": "Top-k sampling"}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Repeat penalty"}),
                "seed": ("INT", {"default": -1, "min": -1, "tooltip": "Random seed (-1 for random)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "LlamaCPP"

    def generate(self, model: Dict[str, Any], prompt: str, image: torch.Tensor = None, options: Dict[str, Any] = None, system_prompt: str = "", memory_cleanup: str = "close", max_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95, top_k: int = 40, repeat_penalty: float = 1.1, seed: int = -1) -> tuple:
        global _global_llm
        try:
            # Validate inputs
            if not isinstance(model, dict) or "model_path" not in model:
                raise ValueError("Invalid model data received from Model Loader")

            options = options or {}
            if not isinstance(options, dict):
                raise ValueError("Invalid options data received from Options node")

            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            # Extract model info
            model_path = model["model_path"]
            chat_format = model["chat_format"]

            # Determine if vision handler is enabled
            vision_enabled = "mmproj_model_path" in model and chat_format.startswith("vision-")

            # Create messages for chat completion
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})

            # Create user message content
            user_content = prompt.strip()

            # If vision is enabled and image is provided, convert image and create structured content
            if vision_enabled and image is not None:
                image_data_uri = _convert_image_to_data_uri(image)
                user_content = [
                    {"type": "image_url", "image_url": image_data_uri},
                    {"type": "text", "text": prompt.strip()}
                ]

            messages.append({"role": "user", "content": user_content})

            # Prepare Llama initialization parameters
            llama_kwargs = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Add options
            for k, v in options.items():
                llama_kwargs[k] = v

            # Handle vision models: use chat_handler based on chat_format
            if vision_enabled:
                handler_class = VISION_HANDLERS.get(chat_format, Llava15ChatHandler)
                chat_handler = handler_class(clip_model_path=model["mmproj_model_path"])
                llama_kwargs["chat_handler"] = chat_handler
                # Remove chat_format when using vision handler
                llama_kwargs.pop("chat_format", None)

            # Pre-generation LLM management
            if memory_cleanup == "persistent":
                if _global_llm is None:
                    _global_llm = Llama(**llama_kwargs)
            else:  # close or backend_free - always cleanup existing and create new
                _cleanup_global_llm(memory_cleanup)
                _global_llm = Llama(**llama_kwargs)

            # Generate response using global LLM
            response = _global_llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )

            # Extract the response text
            if not response or "choices" not in response or not response["choices"]:
                raise RuntimeError("No response generated by the model")

            response_text = response["choices"][0]["message"]["content"]

            # Post-generation memory cleanup
            _cleanup_global_llm(memory_cleanup)

        except Exception as e:
            response_text = f"Error generating response: {str(e)}"
            print(f"LlamaCPP Engine Error: {str(e)}")

        return (response_text,)


class LlamaCPPMemoryCleanup(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "close", "tooltip": "Memory cleanup method"}),
            },
            "optional": {
                "passthrough": ("*", {"tooltip": "Any input to pass through"}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "cleanup"
    CATEGORY = "LlamaCPP"

    def cleanup(self, memory_cleanup: str, passthrough=None) -> tuple:
        try:
            _cleanup_global_llm(memory_cleanup)
        except Exception as e:
            print(f"LlamaCPP Memory Cleanup Error: {str(e)}")

        return (passthrough,)
