import fire
from dependency_injector import containers, providers
from htrocr.line_segmentation.PreciseLineSegmenter import PreciseLineSegmenter
from htrocr.line_segmentation.HBLineSegmenter import HBLineSegmenter
from htrocr.transcriber.wrappers.visionED_wrapper import VisionEncoderDecoderTranscriber
from htrocr.transcriber.wrappers.hybrid_wrapper import HybridTranscriber
from htrocr.transcriber.wrappers.vit_wrapper import VitTranscriber
from htrocr.xmlgenerator import generate_xml
import os
from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import json
import pkg_resources

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
    def __init__(self, config_path='./pipeline_config.json', save_images=False, out_dir='./out', out_type='txt', testing=False):
        os.makedirs(out_dir, exist_ok=True)
        if save_images:
            os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        self.out_dir = out_dir
        self.out_type = out_type
        self.save_images = save_images
        self.testing = testing

        path = os.path.dirname(__file__)
        config_path = os.path.join(path, "pipeline_config.json")
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["baseline_model_weight_path"] = os.path.join(path, 'line_segmentation/predictor/net/default.pb')

        with open(config_path, "w") as f:
            json.dump(data, f)

        lscontainer = LineSegmenterContainer()
        lscontainer.config.from_json(config_path)
        self.segmenter = lscontainer.selector(config_path)

        tcontainer = TranscriberContainer()
        tcontainer.config.from_json(config_path)
        self.transcriber = tcontainer.selector(config_path)
    
    def htrocr_usage(self, required_subject_path = None, transcription_model_weight_path = None,
            save_images=False, out_dir='./out', out_type='txt', testing=False, segmenter = "precise", transcriber= "visioned", 
            baseline_model_weight_path= "line_segmentation/predictor/net/default.pb",
            transcription_img_processor = "microsoft/trocr-base-handwritten",
            superpixel_confidence_thresh = 0.1, min_textline_height = 10,
            downsize_scale = 0.33, crop_ucph_border = True, crop_ucph_border_size = 545,
            neighbour_connectivity_ratio = 0.5, fixed_interline_dist = 100, max_contour = 5,
            contour_adjuster = 5, descender_point_adjuster = 20, use_border_padding = False,
            generate_border_padding_size = 5, border_padding_mode = "constant", greyscale_for_border_padding = 0.9,
            use_rotation_angle = False, rotate_angle_mode = "nearest", angle_fill_greyscale = 0.9):
        
        """
        Full usage of pipeline. Includes configuration of the pipeline_config.json file and self.__init__. Mainly intended for package usage.
        """
        
        path = os.path.dirname(__file__)
        config_path = os.path.join(path, "pipeline_config.json")
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["segmenter"] = segmenter
        data["transcriber"] = transcriber
        if baseline_model_weight_path == "line_segmentation/predictor/net/default.pb":
            data["baseline_model_weight_path"] = os.path.join(path, baseline_model_weight_path)
        else:
            data["baseline_model_weight_path"] = baseline_model_weight_path
        data["transcription_model_weight_path"] = transcription_model_weight_path
        data["transcription_img_processor"] = transcription_img_processor
        data["super_pixel_confidence_thresh"] = superpixel_confidence_thresh
        data["min_textline_height"] = min_textline_height
        data["downsize_scale"] = downsize_scale
        data["crop_ucph_border"] = crop_ucph_border
        data["crop_ucph_border_size"] = crop_ucph_border_size
        data["neighbour_connectivity_ratio"] = neighbour_connectivity_ratio
        data["fixed_interline_dist"] = fixed_interline_dist
        data["max_contour"] = max_contour
        data["contour_adjuster"] = contour_adjuster
        data["descender_point_adjuster"] = descender_point_adjuster
        data["use_border_padding"] = use_border_padding
        data["generate_border_padding_size"] = generate_border_padding_size
        data["border_padding_mode"] = border_padding_mode
        data["greyscale_for_border_padding"] = greyscale_for_border_padding
        data["use_rotation_angle"] = use_rotation_angle
        data["rotate_angle_mode"] = rotate_angle_mode
        data["angle_fill_greyscale"] = angle_fill_greyscale

        with open(config_path, "w") as f:
            json.dump(data, f)
        
        pipeline = NHMDPipeline(config_path, save_images, out_dir, out_type, testing)

        if required_subject_path is None or transcription_model_weight_path is None:
            print("Paths to image(s) and model are required. 'required_subject_path' and 'transcription_model_weight_path'")
        else:    
                if required_subject_path.endswith(".jpg") or required_subject_path.endswith(".png"):
                    pipeline.process_image(required_subject_path)
                else:
                    pipeline.process_dir(required_subject_path)
            




    def evaluate_baseline(self, path, out_dir='./out'):
        """
        Only segmentation run for a single image.
        Image will get processed by a baseline processor and the coordinates
        will be stored in .txt file for an evaluation tool:
        https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme
        """
        _, _, clusters, _, _ = self.segmenter.segment_lines(path)
        baselines = ''
        #  out_dir = './best_NHMD_cbad_preds'
        if len(clusters) == 0:
            with open(os.path.join(out_dir, os.path.basename(path)[:-4] + '.txt'), 'w') as f:
                pass

        for i in range(1, len(clusters)):
            Si = clusters[i]
            for p in Si.keys():
                pointx = int(p[1] * (1/self.config["downsize_scale"]))
                pointy = int(p[0] * (1/self.config["downsize_scale"]))
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
            if self.testing:
                text_line = ""
            
            predictions.append({'file':text_line, 'pred':pred})

        if self.out_type == 'txt':
            txt_predictions = [f'{pred["file"]}\t{pred["pred"]}\n'for pred in predictions]
            if self.testing:
                output_path = os.path.join('./scripts/data/results', f'{id}_result.txt')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # os.path.join('.scripts/data/results', f'{id}_result.txt')
                with open(output_path, 'w', encoding="utf-8") as f:
                    for l in txt_predictions:
                        f.write(l.lstrip())
            else:
                with open(os.path.join(self.out_dir, f'{id}_result.txt'), 'w', encoding="utf-8") as f:
                    for l in txt_predictions:
                        f.write(l.lstrip())
                    # f.write(''.join(txt_predictions))
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