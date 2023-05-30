import os

def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    transcription_result = response.full_text_annotation.text
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return transcription_result

if __name__ == '__main__':
    folder_loc = "./data/NHMD_ORIG_100_cropped"
    output_folder = "./data/VISION_API/NHMD_ORIG_100_cropped_outputs"
    for filename in os.listdir(folder_loc):
        path = os.path.join(folder_loc, filename)
        if os.path.isfile(path) and filename.startswith('sp'):
            print('Detecting text in {}'.format(filename))
            res = detect_document(path)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(res)
