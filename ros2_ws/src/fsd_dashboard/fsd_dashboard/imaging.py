"""Image encoding for the dashboard's camera panel: JPEG via OpenCV or Pillow, else PNG (stdlib)."""

from __future__ import annotations

import struct
import zlib

import numpy as np

try:  # OpenCV ships with most ROS desktop installs
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None
try:
    from PIL import Image as _PIL  # type: ignore
except Exception:  # noqa: BLE001
    _PIL = None

MAX_WIDTH = 640

# Sequential single-hue ramp for depth (near = light, far = dark), from the dashboard's blue scale.
_DEPTH_STOPS = np.array([
    [0xcd, 0xe2, 0xfb], [0x9e, 0xc5, 0xf4], [0x6d, 0xa7, 0xec], [0x39, 0x87, 0xe5],
    [0x25, 0x6a, 0xbf], [0x18, 0x4f, 0x95], [0x0d, 0x36, 0x6b], [0x08, 0x1f, 0x40],
], dtype=np.float32)
_DEPTH_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_DEPTH_STOPS)), _DEPTH_STOPS[:, c])
                       for c in range(3)], axis=1).astype(np.uint8)
NO_DEPTH = np.array([0x2a, 0x2e, 0x33], np.uint8)


def encoder_name() -> str:
    return "opencv-jpeg" if cv2 is not None else "pillow-jpeg" if _PIL is not None else "png"


def downscale(img: np.ndarray, max_width: int = MAX_WIDTH) -> np.ndarray:
    step = int(np.ceil(img.shape[1] / max_width)) if img.shape[1] > max_width else 1
    return img[::step, ::step] if step > 1 else img


def colorize_depth(depth: np.ndarray, max_range: float = 40.0) -> np.ndarray:
    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 0)
    idx = np.clip(np.where(valid, d, 0) / max_range * 255.0, 0, 255).astype(np.uint8)
    rgb = _DEPTH_LUT[idx]
    rgb[~valid] = NO_DEPTH
    return rgb


def encode_rgb(rgb: np.ndarray, quality: int = 72) -> tuple[bytes, str]:
    """RGB uint8 (H x W x 3) -> (bytes, mime)."""
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return buf.tobytes(), "image/jpeg"
    if _PIL is not None:
        import io

        out = io.BytesIO()
        _PIL.fromarray(rgb).save(out, format="JPEG", quality=quality)
        return out.getvalue(), "image/jpeg"
    return encode_png(rgb), "image/png"


def encode_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def from_ros_image(encoding: str, height: int, width: int, step: int, data: bytes) -> tuple[str, np.ndarray] | None:
    """sensor_msgs/Image fields -> ("rgb", HxWx3 uint8) or ("depth", HxW float32 metres)."""
    enc = encoding.lower()
    buf = np.frombuffer(data, np.uint8)
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if "a" in enc[3:] else 3
        img = buf.reshape(height, step)[:, : width * ch].reshape(height, width, ch)[..., :3]
        return "rgb", (img[..., ::-1] if enc.startswith("bgr") else img)
    if enc == "32fc1":
        return "depth", buf.view(np.float32).reshape(height, step // 4)[:, :width]
    if enc == "16uc1":
        return "depth", buf.view(np.uint16).reshape(height, step // 2)[:, :width].astype(np.float32) / 1000.0
    if enc in ("mono8",):
        g = buf.reshape(height, step)[:, :width]
        return "rgb", np.repeat(g[..., None], 3, axis=2)
    return None
