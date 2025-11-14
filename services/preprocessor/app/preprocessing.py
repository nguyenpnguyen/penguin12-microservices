import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> list:
    """
    Load and preprocess image bytes for inference using Pillow and Numpy.
    This replicates:
    - transforms.Resize((128, 128))
    - transforms.ToTensor()
    - transforms.Normalize(mean=mean, std=std)
    - .unsqueeze(0)
    """
    global mean_np, std_np
    if mean_np is None or std_np is None:
        raise RuntimeError("Mean/Std arrays are not loaded")
        
    # 1. Load image from in-memory bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Resize
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    
    # 3. Replicate transforms.ToTensor()
    # Convert to numpy array
    image_np = np.array(image, dtype=np.float32)
    # Scale from [0, 255] to [0.0, 1.0]
    image_np = image_np / 255.0
    # Change from (H, W, C) to (C, H, W)
    image_np = np.transpose(image_np, (2, 0, 1))

    # 4. Replicate transforms.Normalize()
    image_np = (image_np - mean_np) / std_np
    
    # 5. Replicate .unsqueeze(0) to add batch dimension (B, C, H, W)
    image_np = np.expand_dims(image_np, axis=0)
    
    # 6. Convert to list for JSON serialization
    return image_np.tolist()
