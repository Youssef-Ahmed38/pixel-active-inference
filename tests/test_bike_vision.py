import builtins
import numpy as np
import pytest
from pai.bike.vision import ImageOrientation, image_features


def test_orientation_fits_pixels_and_reloads_without_simulator(tmp_path, monkeypatch):
    rng = np.random.default_rng(38)
    colors = rng.integers(0, 256, (80, 3), dtype=np.uint8)
    images = np.broadcast_to(colors[:, None, None, :], (80, 8, 8, 3)).copy()
    labels = colors / 255.0 * [.4, .2, .3] - [.2, .1, .15]
    model = ImageOrientation.fit(images[:60], labels[:60], ridge=1e-6)
    path = tmp_path / "model.json"
    model.save(path)
    original = builtins.__import__
    def reject_simulator(name, *args, **kwargs):
        if name == "mujoco" or name.endswith("bike.env"):
            raise AssertionError("image inference accessed simulation")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_simulator)
    restored = ImageOrientation.load(path)
    for image, expected in zip(images[60:], labels[60:]):
        np.testing.assert_allclose(restored.predict(image), expected, atol=1e-5)


def test_spatial_pooling_preserves_pixel_location():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:2, 2:4, 0] = 255
    features = image_features(image)
    assert features.shape == (193,)
    assert features[3] == 1 and features[-1] == 1
    assert np.count_nonzero(features) == 2


def test_finer_grid_checkpoint_and_legacy_grid(tmp_path):
    import json
    model = ImageOrientation(np.zeros((769, 3)), grid_size=16)
    path = tmp_path / "model.json"
    model.save(path)
    restored = ImageOrientation.load(path)
    assert restored.grid_size == 16
    np.testing.assert_array_equal(restored.predict(np.zeros((96, 96, 3), dtype=np.uint8)), np.zeros(3))
    legacy = {"schema": 1, "weights": np.zeros((193, 3)).tolist(), "summary": {}}
    path.write_text(json.dumps(legacy))
    assert ImageOrientation.load(path).grid_size == 8


@pytest.mark.parametrize("image", [np.zeros((0, 8, 3), dtype=np.uint8), np.zeros((8, 8)),
    np.zeros((9, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=float)])
def test_invalid_images_rejected(image):
    with pytest.raises(ValueError):
        image_features(image)


def test_invalid_fitting_inputs_rejected():
    with pytest.raises(ValueError):
        ImageOrientation.fit([], [])
    with pytest.raises(ValueError):
        ImageOrientation.fit([np.zeros((8, 8, 3), dtype=np.uint8)], [[np.nan, 0, 0]])
    with pytest.raises(ValueError):
        ImageOrientation(np.zeros((2, 3)))
