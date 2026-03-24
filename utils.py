import torch
import torchvision.transforms as T
import base64
import io

def image_to_data_uri(img_tensor):
    """
    Convert a single ComfyUI image tensor to data URI string.

    Args:
        img_tensor: Tensor of shape (H, W, C) - single ComfyUI image

    Returns:
        Data URI string (data:image/png;base64,<base64>)
    """
    to_pil = T.ToPILImage()
    # Assume RGB, permute to (C, H, W) for ToPILImage
    pil_img = to_pil(img_tensor.permute(2, 0, 1))
    # Convert to base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    data_uri = f"data:image/png;base64,{img_base64}"
    # Clean up memory
    buffered.close()
    del pil_img
    return data_uri
