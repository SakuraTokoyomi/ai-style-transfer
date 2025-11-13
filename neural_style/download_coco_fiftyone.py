import os
import shutil
import fiftyone.zoo as foz
from tqdm import tqdm
from PIL import Image

# =============================
# CONFIG
# =============================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))      # 当前脚本所在目录
OUTPUT_DIR = os.path.join(PROJECT_DIR, "../dataset/contents") # 最终要保存到项目的目录
MAX_IMAGES = 120                                               # 可调整子集大小


def download_coco_images(max_images=MAX_IMAGES):
    print("\n==============================")
    print(" STEP 1: Download COCO with FiftyOne")
    print("==============================\n")

    # FiftyOne 会强制下载到用户目录 ~/.fiftyone
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=[],         # 不下载 labels, 只要图片
        max_samples=max_images
    )

    print("\n==> COCO download complete.")
    print("   Images downloaded to:")
    fof = dataset.values("filepath")
    if fof:
        print("    ", os.path.dirname(fof[0]))
    else:
        print("   <Empty dataset>")

    return dataset


def copy_images_to_project(dataset):
    print("\n==============================")
    print(" STEP 2: Copying images to project dataset directory")
    print("==============================\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepaths = dataset.values("filepath")

    print(f"==> Copying {len(filepaths)} images to {OUTPUT_DIR} ...")

    for src in tqdm(filepaths):
        filename = os.path.basename(src)
        dst = os.path.join(OUTPUT_DIR, filename)

        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG")
        except Exception as e:
            print(f"[Error] Failed to copy {src}: {e}")

    print(f"\n==> Copy complete! {len(filepaths)} images saved.\n")
    return filepaths


def delete_original_coco(filepaths):
    print("\n==============================")
    print(" STEP 3: Removing COCO original downloaded folder")
    print("==============================\n")

    if not filepaths:
        print("No files found, skipping cleanup.")
        return

    coco_dir = os.path.dirname(filepaths[0])

    print(f"==> Detected COCO original directory:\n    {coco_dir}")
    print("==> Removing directory ...")

    try:
        shutil.rmtree(os.path.dirname(coco_dir))  # 删除 coco-2017 整个目录
        print("==> Successfully removed original COCO dataset.\n")
    except Exception as e:
        print("[Warning] Failed to delete COCO folder:", e)
        print("You may delete it manually later.")


def main():
    dataset = download_coco_images()
    filepaths = copy_images_to_project(dataset)
    delete_original_coco(filepaths)

    print("==============================")
    print(" All steps complete! 🎉")
    print("==============================\n")
    print("Your training subset is now ready at:")
    print(f"  {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
