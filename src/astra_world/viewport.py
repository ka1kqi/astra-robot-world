"""Bounded browser simulation images accompanying a single conversation turn."""
import base64
from io import BytesIO
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ViewportInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image: str = Field(max_length=1_400_000)
    view: Literal['live', 'experiment', 'action']
    frame_age_ms: int = Field(ge=0, le=86_400_000)

    @field_validator('image')
    @classmethod
    def valid_jpeg(cls, value):
        prefix = 'data:image/jpeg;base64,'
        if not value.startswith(prefix):
            raise ValueError('Viewport must be an inline JPEG.')
        try:
            raw = base64.b64decode(value[len(prefix):], validate=True)
            with Image.open(BytesIO(raw)) as image:
                if image.format != 'JPEG' or not (1 <= image.width <= 1600 and 1 <= image.height <= 1200):
                    raise ValueError('Invalid viewport dimensions or format.')
                image.load()
        except Exception:
            raise ValueError('Viewport must be a valid JPEG up to 1600 by 1200 pixels.') from None
        return value

    def message(self, text, *, responses):
        context = (
            f'\n\nAttached browser simulation viewport: {self.view}; frame age at send: {self.frame_age_ms} ms. '
            'This is a snapshot at message submission, not a continuous visual feed. '
            + ('It shows an experiment copy or recording, not the current live state. ' if self.view != 'live' else '')
            + 'Use observe_world for current exact positions before moving. Screen directions refer to this image; '
            'do not assume screen-right is world +X. The desktop viewer may use a different camera.'
        )
        if responses:
            content = [{'type': 'input_text', 'text': text + context},
                       {'type': 'input_image', 'image_url': self.image, 'detail': 'high'}]
        else:
            content = [{'type': 'text', 'text': text + context},
                       {'type': 'image_url', 'image_url': {'url': self.image, 'detail': 'high'}}]
        return {'role': 'user', 'content': content}
