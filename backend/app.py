# backend/app.py
import uuid
import subprocess
import os
from pathlib import Path
from enum import Enum
from typing import Optional, Annotated

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------
# 路径 & 目录配置
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "saved_models"

for p in [
    UPLOAD_DIR / "images",
    UPLOAD_DIR / "videos",
    OUTPUT_DIR / "images",
    OUTPUT_DIR / "videos",
]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 风格类型 & 固定风格枚举
# -----------------------------
class StyleType(str, Enum):
    fixed = "fixed"
    arbitrary = "arbitrary"


class FixedStyle(str, Enum):
    candy = "candy"
    mosaic = "mosaic"
    rain_princess = "rain_princess"
    udnie = "udnie"


STYLE_MODELS = {
    "candy": MODEL_DIR / "candy.pth",
    "mosaic": MODEL_DIR / "mosaic.pth",
    "rain_princess": MODEL_DIR / "rain_princess.pth",
    "udnie": MODEL_DIR / "udnie.pth",
}

app = FastAPI(
    title="AI Style Transfer Backend",
    description="图像 & 视频风格迁移后端服务（固定风格 + 任意风格）",
    version="1.0.0",
)

# -----------------------------
# 辅助函数：固定风格 - 图像
# -----------------------------
def run_fixed_style_image(content_path: Path, style_name: str) -> Path:
    if style_name not in STYLE_MODELS:
        raise ValueError(f"Unknown style: {style_name}")

    model_path = STYLE_MODELS[style_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_path = OUTPUT_DIR / "images" / f"{uuid.uuid4().hex}.png"

    cmd = [
        "python",
        str(BASE_DIR / "neural_style" / "neural_style.py"),
        "eval",
        "--content-image",
        str(content_path),
        "--model",
        str(model_path),
        "--output-image",
        str(output_path),
    ]
    # Windows 下用 shell=True + 拼成字符串，更稳
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return output_path


# -----------------------------
# 辅助函数：任意风格 - 图像
# -----------------------------
def run_arbitrary_style_image(content_path: Path, style_image_path: Path) -> Path:
    output_path = OUTPUT_DIR / "images" / f"{uuid.uuid4().hex}.png"

    cmd = [
        "python",
        str(BASE_DIR / "neural_style" / "arbitrary_style.py"),
        "--content-image",
        str(content_path),
        "--style-image",
        str(style_image_path),
        "--output-image",
        str(output_path),
        # 不加 --accel
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return output_path


# -----------------------------
# 辅助函数：固定风格 - 视频
# -----------------------------
def run_fixed_style_video(video_path: Path, style_name: str) -> Path:
    if style_name not in STYLE_MODELS:
        raise ValueError(f"Unknown style: {style_name}")

    model_path = STYLE_MODELS[style_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_path = OUTPUT_DIR / "videos" / f"{uuid.uuid4().hex}.mp4"

    cmd = [
        "python",
        str(BASE_DIR / "neural_style" / "neural_style.py"),
        "video",
        "--content-video",
        str(video_path),
        "--model",
        str(model_path),
        "--output-video",
        str(output_path),
        # 不加 --accel
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return output_path


# -----------------------------
# 辅助函数：任意风格 - 视频
# -----------------------------
def run_arbitrary_style_video(video_path: Path, style_image_path: Path) -> Path:
    output_path = OUTPUT_DIR / "videos" / f"{uuid.uuid4().hex}.mp4"

    cmd = [
        "python",
        str(BASE_DIR / "neural_style" / "arbitrary_style.py"),
        "--content-video",
        str(video_path),
        "--style-image",
        str(style_image_path),
        "--output-video",
        str(output_path),
    ]
    # 同样用 shell=True，避免 Windows 参数问题
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return output_path


# ==================================================
# 1. 固定风格图像迁移: POST /stylize/fixed
# ==================================================
@app.post("/stylize/fixed")
async def stylize_fixed_image(
    content_image: UploadFile = File(...),
    style: FixedStyle = Form(...),
):
    """
    固定风格图像迁移
    - content_image: 上传的内容图像 (JPG/PNG)
    - style: 风格名称 (candy / mosaic / rain_princess / udnie)
    返回：PNG 图像文件二进制
    """
    try:
        # 保存上传的内容图像
        suffix = Path(content_image.filename).suffix or ".png"
        input_path = UPLOAD_DIR / "images" / f"{uuid.uuid4().hex}{suffix}"
        with input_path.open("wb") as f:
            f.write(await content_image.read())

        # 枚举 -> 实际字符串 ID，用于 STYLE_MODELS
        output_path = run_fixed_style_image(input_path, style.value)

        return FileResponse(
            path=str(output_path),
            media_type="image/png",
            filename=output_path.name,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ==================================================
# 2. 任意风格图像迁移: POST /stylize/arbitrary
# ==================================================
@app.post("/stylize/arbitrary")
async def stylize_arbitrary_image(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
):
    """
    任意风格图像迁移
    - content_image: 内容图像
    - style_image: 风格参考图像
    返回：PNG 图像文件二进制
    """
    try:
        content_suffix = Path(content_image.filename).suffix or ".png"
        style_suffix = Path(style_image.filename).suffix or ".png"

        content_path = UPLOAD_DIR / "images" / f"{uuid.uuid4().hex}{content_suffix}"
        style_path = UPLOAD_DIR / "images" / f"{uuid.uuid4().hex}{style_suffix}"

        with content_path.open("wb") as f:
            f.write(await content_image.read())
        with style_path.open("wb") as f:
            f.write(await style_image.read())

        output_path = run_arbitrary_style_image(content_path, style_path)

        return FileResponse(
            path=str(output_path),
            media_type="image/png",
            filename=output_path.name,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ==================================================
# 3. 视频风格迁移: POST /stylize/video
# ==================================================
@app.post("/stylize/video")
async def stylize_video(
    video_file: UploadFile = File(...),
    style_type: StyleType = Form(...),
    style: Optional[FixedStyle] = Form(None),
    style_image: Optional[UploadFile] = File(None),
):
    """
    视频风格迁移
    - video_file: MP4 视频
    - style_type: fixed / arbitrary
    - 当 style_type=fixed 时，必须提供 style
    - 当 style_type=arbitrary 时，必须提供 style_image
    """
    try:
        # 保存上传视频
        suffix = Path(video_file.filename).suffix or ".mp4"
        video_path = UPLOAD_DIR / "videos" / f"{uuid.uuid4().hex}{suffix}"
        with video_path.open("wb") as f:
            f.write(await video_file.read())

        # fixed 分支：使用预训练模型
        if style_type == StyleType.fixed:
            if style is None:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "style is required when style_type=fixed"},
                )
            output_path = run_fixed_style_video(video_path, style.value)

        # arbitrary 分支：使用任意风格图像
        else:
            if style_image is None:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "style_image is required when style_type=arbitrary"},
                )

            style_suffix = Path(style_image.filename).suffix or ".png"
            style_path = UPLOAD_DIR / "images" / f"{uuid.uuid4().hex}{style_suffix}"
            with style_path.open("wb") as f:
                f.write(await style_image.read())

            output_path = run_arbitrary_style_video(video_path, style_path)

        print("【后端即将返回的视频路径】", output_path)
        print("【这个文件是否存在？】", os.path.exists(output_path), "大小：",
              os.path.getsize(output_path) if os.path.exists(output_path) else -1)

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=output_path.name,
        )

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )
