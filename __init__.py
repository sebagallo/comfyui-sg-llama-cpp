from .nodes import LlamaCPPModelLoader, LlamaCPPOptions, LlamaCPPEngine, LlamaCPPMemoryCleanup

NODE_CLASS_MAPPINGS = {
    "LlamaCPPModelLoader": LlamaCPPModelLoader,
    "LlamaCPPOptions": LlamaCPPOptions,
    "LlamaCPPEngine": LlamaCPPEngine,
    "LlamaCPPMemoryCleanup": LlamaCPPMemoryCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPPModelLoader": "Llama CPP Model Loader",
    "LlamaCPPOptions": "Llama CPP Options",
    "LlamaCPPEngine": "Llama CPP Engine",
    "LlamaCPPMemoryCleanup": "Llama CPP Memory Cleanup",
}
