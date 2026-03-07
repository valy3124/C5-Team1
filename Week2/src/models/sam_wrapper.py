import torch
from PIL import Image
from typing import Dict, Any, Tuple, List
from transformers import SamModel, SamProcessor

class SamWrapper:
    """
    Wrapper for the Hugging Face Segment Anything Model (SAM).
    Handles model initialization, device management, and inference.
    """
    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str = None):
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
        print(f"Loading SAM ({model_id}) on device: {self.device}...")
        
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")

    def predict(self, image: Image.Image, prompt_dict: Dict[str, Any]) -> Tuple[List[torch.Tensor], torch.Tensor, float]:
        """
        Runs SAM inference using an image and a prompt dictionary.
        """
        import time
        start_time = time.time()
        
        prompt_type = prompt_dict.get("type")
        if prompt_type == "point":
            input_points = [[prompt_dict["points"].tolist()]]
            input_labels = [[prompt_dict["point_labels"].tolist()]]
            
            inputs = self.processor(
                image, 
                input_points=input_points, 
                input_labels=input_labels, 
                return_tensors="pt"
            ).to(self.device)
            
        elif prompt_type == "box":
            input_boxes = [[[prompt_dict["boxes"].tolist()]]]
            inputs = self.processor(
                image, 
                input_boxes=input_boxes, 
                return_tensors="pt"
            ).to(self.device)
            
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Run inference without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)

        # SAM generates masks in a fixed internal resolution, this scales them back
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get the confidence scores (IoU predictions) for the generated masks
        scores = outputs.iou_scores.cpu()
        
        inference_time = time.time() - start_time

        return masks, scores, inference_time