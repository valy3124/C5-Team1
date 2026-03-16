from typing import Any, Dict, List
import PIL.Image as Image


class TextPromptStrategy:
    """
    Text prompt strategy for Grounded SAM.
    """

    def __init__(self, text_labels: str = "person. car."):
        self.text_labels = text_labels

    def generate_prompt(self, image: Image.Image, annotations: List[Any], **kwargs) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": self.text_labels,
        }
