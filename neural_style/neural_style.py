import argparse
import os
import sys
import time
import re
import subprocess

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

# 添加视频处理相关的导入
import cv2
from tqdm import tqdm


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    # 修复设备选择逻辑
    if args.accel and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_model_filename = f"epoch_{args.epochs}_{timestamp}_{args.content_weight}_{args.style_weight}.model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    # 修复设备选择逻辑
    if args.accel and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()            
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


def process_video(args):
    """
    处理视频文件，将风格迁移应用到每一帧
    """
    # 设备选择
    if args.accel and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration for video processing")
    else:
        device = torch.device("cpu")
        print("Using CPU for video processing")

    # 加载模型
    print("Loading style transfer model...")
    style_model = TransformerNet()
    state_dict = torch.load(args.model)
    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    style_model.eval()

    # 打开视频文件
    print(f"Opening video file: {args.content_video}")
    cap = cv2.VideoCapture(args.content_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.content_video}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames")

    # -------------------------
    # 【修改位置 1】 修正宽高为 16 的倍数
    # -------------------------
    # 这一块必须在 VideoWriter 初始化之前
    new_w = width - (width % 16)
    new_h = height - (height % 16)

    # -------------------------
    # 创建视频写入器
    # -------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (new_w, new_h))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file {args.output_video}")
        cap.release()
        return

    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # 处理视频帧
    print("Processing video frames...")
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为模型输入格式
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            # 风格迁移
            with torch.no_grad():
                output_tensor = style_model(input_tensor).cpu()
            
            # 转换回图像格式
            output_image = output_tensor[0].clone().clamp(0, 255).numpy()
            output_image = output_image.transpose(1, 2, 0).astype(np.uint8)

            # 转换RGB到BGR
            output_frame = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            # 修正帧尺寸
            if output_frame.shape[1] != new_w or output_frame.shape[0] != new_h:
                output_frame = cv2.resize(output_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 写入视频
            out.write(output_frame)
            
            frame_count += 1
            pbar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print(f"Video processing completed! Output saved to: {args.output_video}")
    print(f"Processed {frame_count} frames in total")

    # ------------------------------
    # 🔊 合并音频（使用 ffmpeg 并检查返回码）
    # ------------------------------
    final_output = args.output_video.replace(".mp4", "_with_audio.mp4")

    # 检查 ffmpeg 是否在 PATH（可选，但推荐）
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print("ffmpeg not found or not executable. Please install ffmpeg and add it to PATH.")
        print("Audio merge skipped. Final file (silent):", args.output_video)
        return

    # 运行合并命令：复制视频流（来自 args.output_video）和音频流（来自 original）
    # 如果原始没有音频，ffmpeg 会失败；我们捕获并给出友好提示
    cmd = [
        "ffmpeg", "-y",
        "-i", args.output_video,
        "-i", args.content_video,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        final_output
    ]

    try:
        print("Running ffmpeg to merge audio:", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("Audio merged successfully! Final output:", final_output)
        # 可选：检查文件大小是否合理
        if os.path.exists(final_output):
            print("Final file size (bytes):", os.path.getsize(final_output))
        else:
            print("Warning: final output file not found after ffmpeg.")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed to merge audio. ffmpeg stderr:")
        print(e.stderr)
        print("Audio merge failed. You will have the silent video at:", args.output_video)
    except Exception as e:
        print("Unexpected error when running ffmpeg:", e)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument('--accel', action='store_true',
                                  help='use CUDA acceleration if available')
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")
    eval_arg_parser.add_argument('--accel', action='store_true',
                                 help='use CUDA acceleration if available')

    # 添加视频处理子命令
    video_arg_parser = subparsers.add_parser("video", help="parser for video stylizing arguments")
    video_arg_parser.add_argument("--content-video", type=str, required=True,
                                  help="path to content video you want to stylize")
    video_arg_parser.add_argument("--output-video", type=str, required=True,
                                  help="path for saving the output video")
    video_arg_parser.add_argument("--model", type=str, required=True,
                                  help="saved model to be used for stylizing the video")
    video_arg_parser.add_argument('--accel', action='store_true',
                                  help='use CUDA acceleration if available')

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train, eval, or video")
        sys.exit(1)
    
    # 修复加速器检查逻辑
    if args.accel and not torch.cuda.is_available():
        print("ERROR: CUDA acceleration is not available, try running on CPU")
        sys.exit(1)
    if not args.accel and torch.cuda.is_available():
        print("WARNING: CUDA is available, run with --accel to enable acceleration")

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    elif args.subcommand == "eval":
        stylize(args)
    elif args.subcommand == "video":
        process_video(args)


if __name__ == "__main__":
    main()