import numpy as np
from PIL import Image


def create_chunks(img):
    width, height = img.size
    aspect_ratio = width / height
    chunk_height = 40
    chunk_aspect_ratio = 8
    chunk_width = chunk_height * chunk_aspect_ratio
    new_width = int(chunk_height * aspect_ratio)
    img.thumbnail((new_width, chunk_height))
    img_array = np.array(img)
    img_width = img_array.shape[1]
    print(img_width)
    pad_amount = chunk_height
    num_chunks = int(np.ceil(img_width / (chunk_width-pad_amount*2)))
    chunks = []
    chunk_img_width = chunk_width - 2*pad_amount
    for i in range(num_chunks):
        start_x = i * chunk_img_width - pad_amount
        end_x = (i + 1) * chunk_img_width + pad_amount
        chunk = np.zeros((chunk_height, chunk_width))
        if start_x < 0:
            chunk[:, pad_amount:chunk_width] = img_array[:, 0:min(end_x, img_width)]
        elif start_x > 0 and end_x < img_width:
            chunk[:, :end_x] = img_array[:, start_x:end_x]
        elif i==num_chunks-1 and end_x >= img_width:
            chunk[:, :img_width-start_x] = img_array[:, start_x:img_width]
        chunks.append(chunk)
    return chunks

if __name__ == '__main__':
    img = Image.open("image2.jpg").convert('L')
    chunks = create_chunks(img)
    for i, chunk in enumerate(chunks):
        Image.fromarray(chunk).convert('L').save(f"chunk_{i}.jpg")