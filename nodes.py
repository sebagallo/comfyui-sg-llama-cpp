import os
import gc
import inspect
import torch
import sys
import json
from llama_cpp import Llama, llama_backend_free
from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry
try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
except ImportError:
    # Support for older llama-cpp-python versions
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    MTMDChatHandler = Llava15ChatHandler
import folder_paths
from comfy_api.latest import io
from typing import Dict, Any, List
from .utils import image_to_data_uri

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config() -> Dict[str, Any]:
    """Load config from config.json, return empty dict if not found or invalid."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_user_model_folders() -> List[str]:
    """Get user-specified model folders from config."""
    config = load_config()
    return config.get('model_folders', [])

def get_merged_model_folders() -> List[str]:
    """Merge ComfyUI text_encoders folders with user folders."""
    try:
        comfy_folders = folder_paths.get_folder_paths("text_encoders")
    except:
        comfy_folders = []
    user_folders = get_user_model_folders()
    all_folders = comfy_folders + user_folders
    # Filter out non-existent paths
    return [f for f in all_folders if os.path.exists(f)]

def scan_gguf_models_in_folders() -> List[str]:
    """Scan merged folders for GGUF model files."""
    folders = get_merged_model_folders()
    model_list = []
    for folder in folders:
        try:
            files = os.listdir(folder)
            model_list.extend([f for f in files if f.lower().endswith('.gguf')])
        except:
            pass  # Skip inaccessible folders
    return model_list

def find_model_path(model_name: str) -> str:
    """Find full path to model in merged folders."""
    folders = get_merged_model_folders()
    for folder in folders:
        path = os.path.join(folder, model_name)
        if os.path.exists(path):
            return path
    return None

# Global LLM instance for persistence
_global_llm = None

def _cleanup_global_llm(mode: str):
    """Helper function to cleanup the global LLM based on mode."""
    global _global_llm
    if mode == "persistent":
        return  # No cleanup

    # Common cleanup for all non-persistent modes
    if _global_llm is not None:
        # Clean up chat_handler if it exists (for vision models)
        if hasattr(_global_llm, "chat_handler") and _global_llm.chat_handler is not None:
            try:
                chat_handler = _global_llm.chat_handler
                
                # Primary Fix: Close ExitStack if present (used in newer llama-cpp-python versions)
                # This triggers the callback to mtmd_free -> mtmd_cpp.mtmd_free
                if hasattr(chat_handler, "_exit_stack") and chat_handler._exit_stack is not None:
                    try:
                        chat_handler._exit_stack.close()
                    except Exception:
                        pass

                # Secondary Fix: Manual cleanup of attributes as fallback
                # Attempt to find and close the CLIP/mmproj model embedded in the handler
                clip_attrs = ["clip_model", "_clip_model", "mmproj", "_mmproj", "clf"]
                
                for attr_name in clip_attrs:
                    if hasattr(chat_handler, attr_name):
                        attr = getattr(chat_handler, attr_name)
                        if attr is not None:
                            # Try to close/free the model if a method exists
                            if hasattr(attr, "close"):
                                try:
                                    attr.close()
                                except Exception:
                                    pass
                            elif hasattr(attr, "__del__"):
                                try:
                                    attr.__del__()
                                except Exception:
                                    pass
                                    
                            # Explicitly remove the attribute to break reference cycles
                            setattr(chat_handler, attr_name, None)
                            
                _global_llm.chat_handler.close()
            except Exception:
                pass  # Ignore cleanup errors for chat_handler
            
            # Remove the handler callback/reference from the LLM if possible/accessible
            if hasattr(_global_llm, "chat_handler"):
                del _global_llm.chat_handler
            # _global_llm might not have chat_handler attribute in strict sense if it was just in kwargs,
            # but we are cleaning up the attribute we accessed.
            
            gc.collect()
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


# Mapping of chat formats to vision chat handlers (dynamically discovered)
import llama_cpp.llama_chat_format as lcf

VISION_HANDLERS = {}
for name, obj in inspect.getmembers(lcf):
    if inspect.isclass(obj) and issubclass(obj, MTMDChatHandler):
        vision_name = f"vision-{name.lower().replace('chathandler', '')}"
        VISION_HANDLERS[vision_name] = obj


class LlamaCPPModelLoader(io.ComfyNode):
    """Loads a GGUF model and optional mmproj for vision."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_list = scan_gguf_models_in_folders()

        # Filter models based on criteria (case-insensitive)
        model_name_list = [f for f in model_list if 'mmproj' not in f.lower() and 'draft' not in f.lower()]
        mmproj_list = [f for f in model_list if 'mmproj' in f.lower()]

        # Add "None" option to make mmproj truly optional
        if mmproj_list:
            mmproj_list.insert(0, "None")
        else:
            mmproj_list = ["None"]

        # Available chat formats from llama-cpp-python
        base_chat_formats = sorted(list(LlamaChatCompletionHandlerRegistry()._chat_handlers.keys()))
        vision_formats = list(VISION_HANDLERS.keys())
        chat_formats = sorted(base_chat_formats + vision_formats)

        return io.Schema(
            node_id="LlamaCPPModelLoader",
            display_name="Llama CPP Model Loader",
            category="LlamaCPP",
            inputs=[
                io.Combo.Input("model_name", options=model_name_list if model_name_list else ["No GGUF models found"], tooltip="Select GGUF model file"),
                io.Combo.Input("chat_format", options=chat_formats, default="llama-2", tooltip="Chat format template", optional=True),
                io.Combo.Input("mmproj_model_name", options=mmproj_list, default="None", tooltip="Multi-modal projector model for vision (select 'None' to disable)", optional=True),
            ],
            outputs=[
                io.Custom("LLAMA_MODEL").Output(display_name="MODEL")
            ]
        )

    @classmethod
    def execute(cls, model_name: str, chat_format: str = "llama-2", mmproj_model_name: str = "None") -> io.NodeOutput:
        try:
            model_path = find_model_path(model_name)

            if model_path is None:
                raise FileNotFoundError(f"Model file not found: {model_name}")

            if not model_name.lower().endswith('.gguf'):
                raise ValueError(f"Selected file is not a GGUF model: {model_name}")

            model_info = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Handle mmproj if provided and not "None"
            if mmproj_model_name and mmproj_model_name != "None":
                mmproj_model_path = find_model_path(mmproj_model_name)
                if mmproj_model_path is None:
                    raise FileNotFoundError(f"Multi-modal projector model not found: {mmproj_model_name}")
                model_info["mmproj_model_path"] = mmproj_model_path

            # Return model info dict
            return io.NodeOutput(model_info)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


class LlamaCPPOptions(io.ComfyNode):
    """Configures Llama-cpp-python initialization parameters."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCPPOptions",
            display_name="Llama CPP Options",
            category="LlamaCPP",
            inputs=[
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=100, tooltip="Number of GPU layers to use", optional=True),
                io.Int.Input("n_ctx", default=2048, min=-1, max=262144, tooltip="Context window size (0 for max, -1 for default)", optional=True),
                io.Int.Input("n_threads", default=-1, min=-1, max=256, tooltip="Number of threads (-1 for auto)", optional=True),
                io.Int.Input("n_threads_batch", default=-1, min=-1, max=256, tooltip="Number of threads per batch (-1 for auto)", optional=True),
                io.Int.Input("n_batch", default=2048, min=-1, max=16384, tooltip="Batch size (-1 for default)", optional=True),
                io.Int.Input("n_ubatch", default=512, min=-1, max=16384, tooltip="Micro batch size (-1 for default)", optional=True),
                io.Int.Input("main_gpu", default=0, min=-1, max=100, tooltip="GPU ID for main device (-1 for default)", optional=True),
                io.Boolean.Input("offload_kqv", default=True, tooltip="Enable offloading of K/Q/V tensors to GPU", optional=True),
                io.Boolean.Input("numa", default=False, tooltip="Enable NUMA affinity", optional=True),
                io.Boolean.Input("use_mmap", default=True, tooltip="Enable memory-mapped files", optional=True),
                io.Boolean.Input("use_mlock", default=False, tooltip="Enable lock for memory-mapped files", optional=True),
                io.Boolean.Input("use_direct_io", default=False, tooltip="Enable direct I/O for library (Linux only)", optional=True),
                io.Boolean.Input("verbose", default=False, tooltip="Enable verbose logging", optional=True),
                io.Boolean.Input("vision_use_gpu", default=True, tooltip="Vision: Enable GPU for vision handler", optional=True),
                io.Int.Input("vision_image_min_tokens", default=-1, min=-1, max=16384, tooltip="Vision: Minimum image tokens (-1 for default)", optional=True),
                io.Int.Input("vision_image_max_tokens", default=-1, min=-1, max=16384, tooltip="Vision: Maximum image tokens (-1 for default)", optional=True),
                io.Boolean.Input("vision_enable_thinking", default=False, tooltip="Vision: Enable thinking", optional=True),
                io.Boolean.Input("vision_force_reasoning", default=False, tooltip="Vision: Force reasoning", optional=True),
                io.Boolean.Input("vision_add_vision_id", default=True, tooltip="Vision: Add vision ID", optional=True),
            ],
            outputs=[
                io.Custom("LLAMA_OPTIONS").Output(display_name="OPTIONS")
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        try:
            # Filter out None/empty values
            options = {k: v for k, v in kwargs.items() if v is not None and v != ""}

            return io.NodeOutput(options)

        except Exception as e:
            raise RuntimeError(f"Failed to process options: {str(e)}")


class LlamaCPPEngine(io.ComfyNode):
    """Executes chat completion using the loaded model."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCPPEngine",
            display_name="Llama CPP Engine",
            category="LlamaCPP",
            inputs=[
                io.Custom("LLAMA_MODEL").Input("model", tooltip="Loaded Llama model from Model Loader"),
                io.String.Input("prompt", multiline=True, tooltip="User prompt for chat completion"),
                io.Image.Input("images", tooltip="Input image(s) for vision models (supports batches)", optional=True),
                io.Custom("LLAMA_OPTIONS").Input("options", tooltip="Model options from Options node", optional=True),
                io.String.Input("system_prompt", multiline=True, default="", tooltip="System prompt", optional=True),
                io.Combo.Input("memory_cleanup", options=["close", "backend_free", "full_cleanup", "persistent"], default="close", tooltip="Memory cleanup method after generation", optional=True),
                io.DynamicCombo.Input("response_format", options=[
                    io.DynamicCombo.Option("text", inputs=[]),
                    io.DynamicCombo.Option("json_object", inputs=[
                        io.String.Input("json_schema", multiline=True, default="{}", tooltip="JSON Schema to enforce if json_object is selected", optional=True),
                    ])
                ], tooltip="Output format (json_object forces valid JSON)", optional=True),
                io.Int.Input("max_tokens", default=512, min=1, max=262144, tooltip="Maximum tokens to generate", optional=True),
                io.Float.Input("temperature", default=0.2, min=0.0, max=10.0, step=0.01, tooltip="Sampling temperature", optional=True),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01, tooltip="Top-p sampling", optional=True),
                io.Int.Input("top_k", default=40, min=0, max=400, tooltip="Top-k sampling", optional=True),
                io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Min-p sampling", optional=True),
                io.Float.Input("repeat_penalty", default=1.1, min=1.0, max=5.0, step=0.01, tooltip="Repeat penalty", optional=True),
                io.Float.Input("present_penalty", default=0.0, min=0.0, max=5.0, step=0.01, tooltip="Present penalty", optional=True),
                io.Float.Input("frequency_penalty", default=0.0, min=0.0, max=5.0, step=0.01, tooltip="Frequency penalty", optional=True),
                io.Int.Input("seed", default=1, min=-sys.maxsize, max=sys.maxsize, control_after_generate=True, tooltip="Random seed", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="RESPONSE")
            ]
        )

    @classmethod
    def execute(cls, model: Dict[str, Any], prompt: str, images: torch.Tensor = None, options: Dict[str, Any] = None, system_prompt: str = "", memory_cleanup: str = "close", response_format: Dict[str, Any] = {"type": "text"}, max_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.95, top_k: int = 40, min_p: float = 0.05, repeat_penalty: float = 1.1, present_penalty: float = 0.0, frequency_penalty: float = 0.0, seed: int = -1) -> io.NodeOutput:
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
            user_content = [{"type": "text", "text": prompt.strip()}]

            # If vision is enabled and images are provided, convert images and create structured content
            if vision_enabled and images is not None:
                # images is (B, H, W, C)
                B = images.shape[0]
                for i in range(B):
                    img_tensor = images[i]  # (H, W, C)
                    data_uri = image_to_data_uri(img_tensor)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })

            messages.append({"role": "user", "content": user_content})

            # Prepare Llama initialization parameters
            llama_kwargs = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Add options
            for k, v in options.items():
                if not k.startswith("vision_"):
                    # Whitelist for -1 values: only n_gpu_layers is allowed to pass -1
                    if v == -1 and k != "n_gpu_layers":
                        continue
                    llama_kwargs[k] = v

            # Handle vision models: use chat_handler based on chat_format
            if vision_enabled:
                handler_class = VISION_HANDLERS.get(chat_format, MTMDChatHandler)
                
                # Dynamically get parameters from the base class and actual class (MTMDChatHandler)
                base_sig = inspect.signature(MTMDChatHandler)
                handler_params = set(base_sig.parameters.keys())
                handler_sig = inspect.signature(handler_class)
                handler_params.update(handler_sig.parameters.keys())
                
                handler_kwargs = {
                    "clip_model_path": model["mmproj_model_path"],
                    "verbose": options.get("verbose", False),
                }

                # Process vision_ prefixed options
                for k, v in options.items():
                    if k.startswith("vision_"):
                        param_name = k.replace("vision_", "")
                        if param_name in handler_params:
                            # Skip -1 values for vision handlers
                            if v == -1:
                                continue
                            handler_kwargs[param_name] = v

                chat_handler = handler_class(**handler_kwargs)
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

            # Prepare response_format parameter
            response_format_param = None
            if isinstance(response_format, dict):
                fmt_type = response_format.get("response_format", "text")
                if fmt_type == "json_object":
                    response_format_param = {"type": "json_object"}
                    json_schema = response_format.get("json_schema", "{}").strip()
                    if json_schema:
                        try:
                            response_format_param["schema"] = json.loads(json_schema)
                        except json.JSONDecodeError:
                            response_format_param["schema"] = json_schema
                else:
                    response_format_param = {"type": fmt_type}
            else:
                # Fallback for simple string if it happens
                response_format_param = {"type": str(response_format)}

            # Generate response using global LLM
            response = _global_llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                present_penalty=present_penalty,
                frequency_penalty=frequency_penalty,
                min_p=min_p,
                seed=seed,
                response_format=response_format_param,
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

        return io.NodeOutput(response_text)


class LlamaCPPMemoryCleanup(io.ComfyNode):
    """Manually triggers memory cleanup."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("passthrough")
        return io.Schema(
            node_id="LlamaCPPMemoryCleanup",
            display_name="Llama CPP Memory Cleanup",
            category="LlamaCPP",
            inputs=[
                io.Combo.Input("memory_cleanup", options=["close", "backend_free", "full_cleanup", "persistent"], default="close", tooltip="Memory cleanup method"),
                io.MatchType.Input("passthrough", template=template, tooltip="Any input to pass through", optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="PASSTHROUGH")
            ]
        )

    @classmethod
    def execute(cls, memory_cleanup: str, passthrough=None) -> io.NodeOutput:
        try:
            _cleanup_global_llm(memory_cleanup)
        except Exception as e:
            print(f"LlamaCPP Memory Cleanup Error: {str(e)}")

        return io.NodeOutput(passthrough)
