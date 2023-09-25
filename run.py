import fire
from dependency_injector import containers, providers
from line_segmentation.PreciseLineSegmenter import PreciseLineSegmenter
from line_segmentation.HBLineSegmenter import HBLineSegmenter
from transcriber.wrappers.visionED_wrapper import VisionEncoderDecoderTranscriber
from transcriber.wrappers.hybrid_wrapper import HybridTranscriber
from transcriber.wrappers.vit_wrapper import VitTranscriber
from xmlgenerator import generate_xml
import os
from PIL import Image
import numpy as np
import time
from tqdm import tqdm

# more to come here
class LineSegmenterContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    selector = providers.Selector(
        config.segmenter,
        precise=providers.Factory(PreciseLineSegmenter),
        height_based=providers.Factory(HBLineSegmenter),
    )

class TranscriberContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    selector = providers.Selector(
        config.transcriber,
        visioned=providers.Factory(VisionEncoderDecoderTranscriber),
        hybrid=providers.Factory(HybridTranscriber),
        vit=providers.Factory(VitTranscriber),
    )

class NHMDPipeline(object):
    def __init__(self, config_path='./pipeline_config.json', save_images=False, out_dir='./out', out_type='txt'):
        os.makedirs(out_dir, exist_ok=True)
        if save_images:
            os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        self.out_dir = out_dir
        self.out_type = out_type
        self.save_images = save_images

        lscontainer = LineSegmenterContainer()
        lscontainer.config.from_json(config_path)
        self.segmenter = lscontainer.selector(config_path)

        tcontainer = TranscriberContainer()
        tcontainer.config.from_json(config_path)
        self.transcriber = tcontainer.selector(config_path)

    def evaluate_baseline(self, path, out_dir='./out'):
        """
        Only segmentation run for a single image.
        Image will get processed by a baseline processor and the coordinates
        will be stored in .txt file for an evaluation tool:
        https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme
        """
        _, _, clusters, _, _ = self.segmenter.segment_lines(path)
        baselines = ''
        out_dir = './best_NHMD_cbad_preds'
        if len(clusters) == 0:
            with open(os.path.join(out_dir, os.path.basename(path)[:-4] + '.txt'), 'w') as f:
                pass

        for i in range(1, len(clusters)):
            Si = clusters[i]
            for p in Si.keys():
                pointx = int(p[1] * (1/0.33))
                pointy = int(p[0] * (1/0.33))
                baselines += f'{pointx},{pointy};'
            baselines = baselines[:-1]
            baselines += '\n'
        with open(os.path.join(out_dir, os.path.basename(path)[:-4] + '.txt'), 'w') as f:
            f.write(baselines)

    def evaluate_baselines(self, path, out_dir='./out'):
        """
        Only segmentation run for a folder of images.
        Images will get processed by a baseline processor and the coordinates
        will be stored in .txt files for an evaluation tool:
        https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme
        """
        print('Started evaluation.')
        start = time.time()
        os.makedirs(out_dir, exist_ok=True)
        file_list = os.listdir(path)
        progress_bar = tqdm(total=len(file_list))
        for file in file_list:
            if (file.endswith('.jpg') or file.endswith('.png')):
                file_path = os.path.join(path, file)
                self.evaluate_baseline(file_path, out_dir)
            progress_bar.update(1)
        end = time.time()
        print(f"End of processing. Runtime: {int(end-start)} seconds")

    def process_image(self, path, id=None):
        """
        End-to-end transcription processor for a single image.
        Image will get segmented and transcribed.
        """
        if id is None:
            print('Starting processing...')
            start = time.time()
        lines, polygons, baselines, region_coords, scale = self.segmenter.segment_lines(path)
        predictions = []
        for idx, line in enumerate(lines):
            text_line = f'{id}_line_{idx}.jpg' if id is not None else f'line_{idx}.jpg'
            if self.save_images:
                img = Image.fromarray(line*255).convert('L')
                img.save(os.path.join(self.out_dir, 'images', text_line))
            pred = self.transcriber.transcribe(np.array(line*255))
            predictions.append({'file':text_line, 'pred':pred})

        if self.out_type == 'txt':
            txt_predictions = [f'{pred["file"]}\t{pred["pred"]}\n'for pred in predictions]
            with open(os.path.join(self.out_dir, f'{id}_result.txt'), 'w') as f:
                f.write(''.join(txt_predictions))
        elif self.out_type == 'xml':
            filename = path.split('/')[-1]
            transcriptions = [d["pred"] for d in predictions]
            generate_xml(filename, polygons, baselines,
                         region_coords, scale, transcriptions, self.out_dir)
        if id is None:
            end = time.time()
            print(f"End of processing. Inference time: {int(end-start)} seconds")

    def process_dir(self, path):
        """
        End-to-end transcription processor for a folder of images.
        Images will get segmented and transcribed.
        """
        print('Starting processing...')
        start = time.time()
        for file in os.listdir(path):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.process_image(os.path.join(path, file), file[:-4])
        end = time.time()
        print(f"End of processing. Inference time: {int(end-start)} seconds")

if __name__ == '__main__':
    fire.Fire(NHMDPipeline)