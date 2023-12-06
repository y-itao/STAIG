import argparse
from staig.adata_processing import LoadSingle10xAdata
import cv2
import os

def process_data(dataset, slide, patch_size, label):
    # Generate dataset path
    path = f"./Dataset/{dataset}/{slide}"

    # Initialize data loader
    loader = LoadSingle10xAdata(path=path, image_emb=False, label=label, filter_na=True)
    loader.load_data()
    if label:
        loader.load_label()
    adata = loader.adata

    if os.path.exists(os.path.join(path, "spatial", "tissue_full_image.tif")):
        print("File exists.")
    else:
        print("File does not exist.")

    print(cv2.__version__)


    # Read tiff image
    im = cv2.imread(os.path.join(path, "spatial", "tissue_full_image.tif"), cv2.IMREAD_COLOR)
    print(im)
    # Create directory for clipped images
    clip_image_path = os.path.join(path, 'clip_image')
    try:
        os.makedirs(clip_image_path)
        print("文件夹创建成功")
    except FileExistsError:
        print("文件夹已存在")

    # Process and save patches
    patches = []
    for i, coord in enumerate(adata.obsm['spatial']):
        # Calculate patch coordinates
        left = int(coord[0] - patch_size / 2)
        top = int(coord[1] - patch_size / 2)
        right = left + patch_size
        bottom = top + patch_size

        # Extract patch
        patch = im[top:bottom, left:right]

        # Save patch
        cv2.imwrite(os.path.join(clip_image_path, f'{i}.png'), patch)
        print(i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for patches.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--slide", type=str, required=True, help="Slide name")
    parser.add_argument("--patch_size", type=int, default=128, help="Size of the patch")
    parser.add_argument("--label", type=bool, default=False, help="Whether to load labels")

    args = parser.parse_args()

    process_data(args.dataset, args.slide, args.patch_size, args.label)
