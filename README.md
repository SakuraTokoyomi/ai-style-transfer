# ai-style-transfer
Course Project | &lt;HKU/CDS> &lt;7103C> | Term: Autumn 2025



## Dataset Download Guide (for all team members)

This project requires a subset of the COCO 2017 training images as the **content dataset** for training our style transfer models.

To ensure consistency and avoid downloading the full 18GB COCO dataset,
 **we provide a one-click script that automatically:**

1. Downloads COCO via FiftyOne (only the needed images)
2. Copies selected images into our project folder
3. Deletes the original downloaded files to save disk space

### ✅ 1. Environment Setup

Before running the script, make sure your environment is installed via `uv`（ask chatgpt if not installed）:

```
uv sync
```

This installs all dependencies, including:

- fiftyone
- pillow
- tqdm

You do **NOT** need to install COCO manually.

------

### 🚀 2. One-Click COCO Download

Run the script:

```
python neural_style/download_coco_fiftyone.py
```

This script will:

#### ✔ Step 1: Download COCO 2017 automatically

FiftyOne downloads ~120 random images (customizable) to:

```
C:/Users/<username>/fiftyone/coco-2017/
```

#### ✔ Step 2: Copy the images into our project directory

Images are saved to:

```
dataset/contents/
```

#### ✔ Step 3: Delete the original COCO folder to save disk space

Typically removes:

```
C:/Users/<username>/fiftyone/coco-2017/
```

You can see detailed progress in the console output.

------

### 📂 3. Expected Directory Structure

After running the script, your project directory will contain:

```
dataset/
  contents/
    00000001.jpg
    00000002.jpg
    ...
```

Only ~120 images are stored (configurable in script), which is enough for training our models and keeps the repository small.

------

### 🎛 4. Configuration (optional)

If you want to change the number of downloaded images:

Open the script:

```
neural_style/download_coco_fiftyone.py
```

Modify:

```
MAX_IMAGES = 120
```

For example:

- 120  → fastest, enough for demo
- 1,000 → recommended
- 12,000 → closest to real training, but slower(total 118k)

## 📘数据集下载指南

本项目需要 COCO 2017 训练集中的一部分图像作为内容数据，用于训练我们的风格迁移模型。
 为避免下载完整 18GB 的 COCO 数据集，我们提供了一个一键下载脚本，它能自动完成：

- 下载 COCO（仅需要的图像部分）
- 复制到项目目录
- 删除多余的原始文件（节省磁盘空间）

------

### **1. 环境准备**

请确保使用 uv 安装依赖(问gpt如果没有安装)：

```
uv sync
```

上述命令将自动安装：

- FiftyOne
- Pillow
- tqdm

你**不需要手动下载 COCO 数据集**。

------

### **2. 一键下载 COCO**

运行脚本：

```
python neural_style/download_and_prepare_coco.py
```

该脚本将自动执行：

1.通过 FiftyOne 下载 COCO 2017 训练图像至

```
C:/Users/<username>/fiftyone/coco-2017/
```

2.将挑选出的图像复制到：

```
dataset/contents/
```

3.删除原始 COCO 目录，节省磁盘空间

```
C:/Users/<username>/fiftyone/coco-2017/
```

你可以在控制台中看到日志输出

------

### **3. 预期目录结构**

脚本运行完成后，你的项目目录应为：

```
dataset/
  contents/
    00000001.jpg
    00000002.jpg
    ...
```

少量图像（如 120 张）足以用于示例训练。

------

### **4. （可选）调整下载的图像数量**

打开脚本：

```
neural_style/download_and_prepare_coco.py
```

修改参数：

```
MAX_IMAGES = 120
```

推荐值：

- 120 → 最快
- 1,000 → 比较理想
- 12,000 → 更接近真实训练规模(总数118000，模型训练成员自行选择)
