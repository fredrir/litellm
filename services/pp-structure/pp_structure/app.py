from __future__ import annotations

import argparse
import base64
import binascii
import io
import os
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from pp_structure import SERVICE_VERSION

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MAX_PIXELS = 30_000_000
DISABLED_STAGES = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "use_table_recognition": False,
    "use_formula_recognition": False,
    "use_seal_recognition": False,
    "use_chart_recognition": False,
    "use_region_detection": True,
}


class LayoutRequest(BaseModel):
    png: str = Field(min_length=1)


class TextLine(BaseModel):
    bbox: list[float]
    text: str
    score: float


class LayoutBlock(BaseModel):
    label: str
    bbox: list[float]
    score: float
    order: int | None = None
    text: str = ""


class LayoutResponse(BaseModel):
    width: int
    height: int
    blocks: list[LayoutBlock]
    lines: list[TextLine]
    duration_ms: int


def default_device() -> str:
    import paddle

    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu:0"
    return "cpu"


def _box(value: Any) -> list[float]:
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    if len(values) != 4:
        raise ValueError("invalid box")
    return [float(v) for v in values]


def _iou(a: list[float], b: list[float]) -> float:
    shared = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union > 0 else 0.0


class Analyzer:
    def __init__(self, device: str) -> None:
        from paddleocr import PPStructureV3

        self.device = device
        # Paddle 3.3.1 oneDNN cannot run PP-StructureV3's PIR graphs on CPU.
        cpu = {"enable_mkldnn": False} if device.startswith("cpu") else {}
        self.pipeline = PPStructureV3(device=device, **cpu, **DISABLED_STAGES)
        self.lock = threading.Lock()

    def analyze(self, image: np.ndarray) -> LayoutResponse:
        started = time.monotonic()
        with self.lock:
            result = next(iter(self.pipeline.predict(image)))
        payload = result.json["res"]
        detected = [
            LayoutBlock(label=str(box["label"]), bbox=_box(box["coordinate"]), score=float(box["score"]))
            for box in payload.get("layout_det_res", {}).get("boxes", [])
        ]
        blocks: list[LayoutBlock] = []
        for item in payload.get("parsing_res_list", []):
            bbox = _box(item["block_bbox"])
            match = max(detected, key=lambda box: _iou(box.bbox, bbox), default=None)
            score = match.score if match is not None and _iou(match.bbox, bbox) > 0.5 else 1.0
            order = item.get("block_order")
            blocks.append(
                LayoutBlock(
                    label=str(item["block_label"]),
                    bbox=bbox,
                    score=score,
                    order=int(order) if isinstance(order, int) and not isinstance(order, bool) else None,
                    text=str(item.get("block_content") or ""),
                )
            )
        for box in detected:
            if all(_iou(box.bbox, block.bbox) <= 0.5 for block in blocks):
                blocks.append(box)
        ocr = payload.get("overall_ocr_res", {})
        lines = [
            TextLine(bbox=_box(box), text=str(text), score=float(score))
            for text, box, score in zip(ocr.get("rec_texts", []), ocr.get("rec_boxes", []), ocr.get("rec_scores", []))
            if str(text).strip()
        ]
        return LayoutResponse(
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            blocks=blocks,
            lines=lines,
            duration_ms=round((time.monotonic() - started) * 1000),
        )

    def warmup(self) -> None:
        self.analyze(np.full((1684, 1190, 3), 255, dtype=np.uint8))


def decode_png(encoded: str) -> np.ndarray:
    try:
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > MAX_PIXELS:
                raise ValueError("invalid or oversized image")
            return np.asarray(image.convert("RGB"))
    except (OSError, binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid png: {exc}") from exc


def create_app(device: str | None = None) -> FastAPI:
    import paddle
    import paddleocr

    analyzer = Analyzer(device or default_device())
    analyzer.warmup()
    app = FastAPI(title="pp-structure")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ready": True,
            "service": SERVICE_VERSION,
            "device": analyzer.device,
            "paddleocr": paddleocr.__version__,
            "paddle": paddle.__version__,
        }

    @app.post("/v1/layout", response_model=LayoutResponse)
    def layout(request: LayoutRequest) -> LayoutResponse:
        return analyzer.analyze(decode_png(request.png))

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="pp-structure")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--device", default=os.environ.get("PP_STRUCTURE_DEVICE") or None)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.device), host=args.host, port=args.port, log_level="warning")
    return 0
