from .nodes import LlamaCPPModelLoader, LlamaCPPOptions, LlamaCPPEngine, LlamaCPPMemoryCleanup
from comfy_api.latest import ComfyExtension, io

class SGLlamaExtension(ComfyExtension):
    """Extension for Llama-cpp nodes."""
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LlamaCPPModelLoader,
            LlamaCPPOptions,
            LlamaCPPEngine,
            LlamaCPPMemoryCleanup,
        ]

async def comfy_entrypoint() -> SGLlamaExtension:
    return SGLlamaExtension()
