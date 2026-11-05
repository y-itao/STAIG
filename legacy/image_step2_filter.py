import argparse
import cv2
import numpy as np
import os


def create_custom_mask(image_shape, x1, y1, x2, y2):
    rows, cols = image_shape
    mask = np.zeros((rows, cols), np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def process_images(dataset, slide, lower, upper):
    path = f"./Dataset/{dataset}/{slide}/clip_image/"
    output_path = f"./Dataset/{dataset}/{slide}/clip_image_filter"

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_path)
        print("文件夹创建成功")
    except FileExistsError:
        print("文件夹已存在")

    # Get the number of images in the directory
    num = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

    for i in range(num):
        print(i)

        # Load an image
        image = cv2.imread(os.path.join(path, f'{i}.png'), 0)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Perform Fourier Transform
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        # Create and apply a custom mask
        image_shape = image.shape
        custom_mask = create_custom_mask(image_shape, lower, lower, upper, upper)
        fshift_masked = fshift * custom_mask

        # Perform inverse Fourier Transform
        f_ishift = np.fft.ifftshift(fshift_masked)
        image_filtered = np.fft.ifft2(f_ishift)
        image_filtered = np.abs(image_filtered)
        image_filtered = cv2.GaussianBlur(image_filtered, (15, 15), 0)
        image_filtered_rgb = cv2.cvtColor(np.float32(image_filtered), cv2.COLOR_GRAY2RGB)

        # Save the processed image
        cv2.imwrite(os.path.join(output_path, f'{i}.png'), image_filtered_rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with Fourier Transform and custom mask.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--slide", type=str, required=True, help="Slide name")
    parser.add_argument("--lower", type=int, required=True, help="Lower bound for mask")
    parser.add_argument("--upper", type=int, required=True, help="Upper bound for mask")

    args = parser.parse_args()

    process_images(args.dataset, args.slide, args.lower, args.upper)
