import numpy as np
from typing import Any, Dict, List
import PIL.Image as Image


class YoloDetPromptStrategy:
    """
    Dummy prompt strategy for use with YoloSamWrapper.

    YoloSamWrapper runs its own YOLO detector internally and ignores the
    prompt_dict produced here. This strategy exists solely to:
      1. Keep the prompt-strategy interface consistent.
      2. Return a non-empty count (1) so the inference loop never skips a frame.

    The visualisation for this prompt is handled separately via the
    `save_prompt_boxes` flag, which draws the actual YOLO bounding boxes
    returned by YoloSamWrapper.predict().
    """

    def generate_prompt(self, image: Image.Image, annotations: List[Any], **kwargs) -> Dict[str, Any]:
        # Return a sentinel dict; count=1 keeps the inference loop running.
        return {
            "type": "yolo",      # handled in get_prompt_count / draw_prompts
            "count": 1,
        }
