from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from trame_rca.encoders.img import rgbx_view

try:
    from trame_rca.encoders import turbo_jpeg
except (ModuleNotFoundError, RuntimeError):  # PyTurboJPEG or libjpeg-turbo >= 3.0
    turbo_jpeg = None


@pytest.mark.skipif(
    turbo_jpeg is None, reason="needs PyTurboJPEG and libjpeg-turbo>=3.0"
)
@pytest.mark.parametrize("rgbx", [False, True], ids=["rgb", "rgbx-view"])
def test_turbo_jpeg_uses_420_chroma_subsampling(rgbx):
    rgba = np.random.default_rng(0).integers(0, 256, (48, 64, 4), dtype=np.uint8)
    image = rgba[..., :3] if rgbx else np.ascontiguousarray(rgba[..., :3])
    assert (rgbx_view(image) is not None) == rgbx

    encoded = turbo_jpeg.encode_np_img_to_bytes(image, 48, 64, 90)

    # (horizontal, vertical) sampling factors of Y, Cb, Cr: 4:2:0
    jpeg = Image.open(BytesIO(encoded))
    assert [layer[1:3] for layer in jpeg.layer] == [(2, 2), (1, 1), (1, 1)]
