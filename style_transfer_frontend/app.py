# app.py
import gradio as gr
import requests
import tempfile
import os
import uuid
from typing import Tuple

# 后端API配置
BACKEND_URL = "http://127.0.0.1:8000"  # 根据实际后端地址修改


class StyleTransferFrontend:
    def __init__(self):
        self.fixed_styles = {
            "Candy": "candy",
            "Mosaic": "mosaic",
            "Rain Princess": "rain_princess",
            "Udnie": "udnie",
        }

    def fixed_style_transfer(self, content_image, style_name: str) -> Tuple[str, str]:
        """固定风格迁移"""
        if content_image is None:
            return None, "请先上传内容图像"

        try:
            # 准备请求数据
            files = {"content_image": open(content_image, "rb")}
            data = {"style": self.fixed_styles[style_name]}

            # 调用后端API
            response = requests.post(
                f"{BACKEND_URL}/stylize/fixed",
                files=files,
                data=data
            )

            if response.status_code == 200:
                # 保存结果图片
                result_path = self._save_temp_image(response.content)
                return result_path, "风格迁移完成！"
            else:
                return None, f"处理失败: {response.text}"

        except Exception as e:
            return None, f"发生错误: {str(e)}"

    def arbitrary_style_transfer(self, content_image, style_image) -> Tuple[str, str]:
        """任意风格迁移"""
        if content_image is None or style_image is None:
            return None, "请先上传内容图像和风格图像"

        try:
            files = {
                "content_image": open(content_image, "rb"),
                "style_image": open(style_image, "rb")
            }

            response = requests.post(
                f"{BACKEND_URL}/stylize/arbitrary",
                files=files
            )

            if response.status_code == 200:
                result_path = self._save_temp_image(response.content)
                return result_path, "任意风格迁移完成！"
            else:
                return None, f"处理失败: {response.text}"

        except Exception as e:
            return None, f"发生错误: {str(e)}"

    def video_style_transfer(self, video_file, style_type: str, style_name: str = None, style_image=None) -> Tuple[
        str, str]:
        """视频风格迁移"""
        if video_file is None:
            return None, "请先上传视频文件"

        try:
            if isinstance(video_file, dict):
                # Gradio 3.x / 4.x 视频对象都会有 "name" 字段
                # "name" 是实际文件路径
                if "name" in video_file:
                    video_path = video_file["name"]
                elif "data" in video_file:
                    video_path = video_file["data"]
                else:
                    return None, "无法解析上传的视频，请重试"

            elif isinstance(video_file, str):
                video_path = video_file

            else:
                return None, "视频路径格式无法识别"

            files = {"video_file": open(video_path, "rb")}
            data = {"style_type": style_type}

            if style_type == "fixed":
                if not style_name:
                    return None, "请选择固定风格"
                data["style"] = self.fixed_styles[style_name]
            elif style_type == "arbitrary":
                if style_image is None:
                    return None, "请上传风格图像"
                files["style_image"] = open(style_image, "rb")

            response = requests.post(
                f"{BACKEND_URL}/stylize/video",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result_path = self._save_temp_video(response.content)
                return result_path, "视频风格迁移完成！"
            else:
                return None, f"处理失败: {response.text}"

        except Exception as e:
            return None, f"发生错误: {str(e)}"

    def _save_temp_image(self, image_data: bytes) -> str:
        """保存临时图片文件"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(image_data)
            return f.name

    def _save_temp_video(self, video_data: bytes) -> str:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"video_{uuid.uuid4().hex}.mp4")

        with open(temp_path, "wb") as f:
            f.write(video_data)

        print("前端保存的视频大小：", len(video_data))
        print("实际写入的临时文件大小：", os.path.getsize(temp_path))
        return temp_path


def create_interface():
    """创建Gradio界面"""
    frontend = StyleTransferFrontend()

    with gr.Blocks(
            title="AI艺术风格迁移系统",
            theme=gr.themes.Soft(),
            css="""
        .container { max-width: 1200px; margin: auto; }
        .result-container { display: flex; gap: 20px; }
        .image-preview { flex: 1; }
        """
    ) as interface:

        gr.Markdown("""
        # 🎨 AI艺术风格迁移系统
        将您的图片和视频转换为经典艺术风格！
        """)

        with gr.Tabs():
            # Tab 1: 固定风格图像迁移
            with gr.TabItem("🎭 固定风格图像迁移"):
                with gr.Row():
                    with gr.Column():
                        fixed_content = gr.Image(
                            label="上传内容图像",
                            type="filepath",
                            sources=["upload", "clipboard"]
                        )
                        fixed_style = gr.Dropdown(
                            choices=list(frontend.fixed_styles.keys()),
                            label="选择艺术风格",
                            value="Candy"
                        )
                        fixed_btn = gr.Button("开始风格迁移", variant="primary")

                    with gr.Column():
                        fixed_output = gr.Image(label="风格化结果")
                        fixed_message = gr.Textbox(label="处理状态", interactive=False)

                fixed_btn.click(
                    fn=frontend.fixed_style_transfer,
                    inputs=[fixed_content, fixed_style],
                    outputs=[fixed_output, fixed_message]
                )

            # Tab 2: 任意风格图像迁移
            with gr.TabItem("🔄 任意风格迁移"):
                with gr.Row():
                    with gr.Column():
                        arbitrary_content = gr.Image(
                            label="上传内容图像",
                            type="filepath"
                        )
                        arbitrary_style = gr.Image(
                            label="上传风格参考图像",
                            type="filepath"
                        )
                        arbitrary_btn = gr.Button("开始风格迁移", variant="primary")

                    with gr.Column():
                        arbitrary_output = gr.Image(label="风格化结果")
                        arbitrary_message = gr.Textbox(label="处理状态", interactive=False)

                arbitrary_btn.click(
                    fn=frontend.arbitrary_style_transfer,
                    inputs=[arbitrary_content, arbitrary_style],
                    outputs=[arbitrary_output, arbitrary_message]
                )

            # Tab 3: 视频风格迁移
            with gr.TabItem("🎬 视频风格迁移"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="上传视频文件",
                            sources=["upload"],
                            format = "mp4"  # 输入统一转成 mp4
                        )
                        video_style_type = gr.Radio(
                            choices=["fixed", "arbitrary"],
                            label="风格类型",
                            value="fixed"
                        )
                        video_style_select = gr.Dropdown(
                            choices=list(frontend.fixed_styles.keys()),
                            label="选择固定风格",
                            value="Candy",
                            visible=True
                        )
                        video_style_image = gr.Image(
                            label="上传风格图像",
                            type="filepath",
                            visible=False
                        )
                        video_btn = gr.Button("开始视频风格迁移", variant="primary")

                    with gr.Column():
                        video_output = gr.File(label="风格化视频")
                        video_message = gr.Textbox(label="处理状态", interactive=False)

                # 动态显示/隐藏风格选择组件
                def update_video_style_ui(style_type):
                    if style_type == "fixed":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)

                video_style_type.change(
                    fn=update_video_style_ui,
                    inputs=video_style_type,
                    outputs=[video_style_select, video_style_image]
                )

                video_btn.click(
                    fn=frontend.video_style_transfer,
                    inputs=[video_input, video_style_type, video_style_select, video_style_image],
                    outputs=[video_output, video_message]
                )

        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 功能说明：

            **🎭 固定风格图像迁移**
            - 上传内容图像，选择预训练的艺术风格
            - 支持风格：Candy、Mosaic、Rain Princess、Udnie

            **🔄 任意风格迁移**
            - 上传内容图像和风格参考图像
            - 系统将提取风格特征并应用到内容图像

            **🎬 视频风格迁移**
            - 上传视频文件，选择固定风格或上传风格图像
            - 系统将逐帧处理并生成风格化视频

            ### 支持格式：
            - 图像：JPG, PNG, JPEG
            - 视频：MP4

            ### 注意事项：
            - 建议图像分辨率不超过 4K
            - 视频处理时间较长，请耐心等待
            """)

    return interface


if __name__ == "__main__":
    # 创建界面并启动
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为True可生成公共链接
        debug=True
    )