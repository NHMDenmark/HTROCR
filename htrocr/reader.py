import argparse
from htrocr.run import NHMDPipeline as pipe

def transcribe():
   
    
    pipe.htrocr_usage(pipe, manyargs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Detect and read handwritten text from images.")
    parser.add_argument('--required_subject_path', required=True, help='Path to the subject image or images directory.')
    parser.add_argument('--transcription_model_weight_path', required=True, help='Path to the model directory.')
    parser.add_argument('--save_images', help='Setting for saving the textlines.', default=False)
    parser.add_argument('--out_dir', help='Sets the directory for results.', default='./out')
    parser.add_argument('--out_type', help='Choose the output type.',default='txt')
    parser.add_argument('--segmenter', help='Segmenter setting.',default='precise')
    parser.add_argument('--transcriber', help='Transcriber setting.',default='visioned')
    parser.add_argument('--baseline_model_weight_path', help='Sets path to baseline model weights.',default='line_segmentation/predictor/net/default.pb')
    parser.add_argument('--transcription_img_processor', help='Set image processor.',default='microsoft/trocr-base-handwritten')
    parser.add_argument('--superpixel_confidence_thresh', help='Confidence threshold setting for super pixels.',default=0.1)
    parser.add_argument('--min_textline_height', help='Min textline height for detecting text in image.',default=10)
    parser.add_argument('--downsize_scale', help='Sets downsize scale for image processing.',default=0.33)
    parser.add_argument('--crop_ucph_border', help='Crop right side of the image.',default=True)
    parser.add_argument('--crop_ucph_border_size', help='Crop sizing.',default=545)
    parser.add_argument('--neighbour_connectivity_ratio', help='Sets the neighbouring super pixels connectivity ratio',default=0.5)
    parser.add_argument('--fixed_interline_dist', help='Sets the cut off point for interline distance.',default=100)
    parser.add_argument('--max_contour', help='Sets max contour adjustment.',default=5)
    parser.add_argument('--contour_adjuster', help='Sets the contour adjustment.',default=5)
    parser.add_argument('--descender_point_adjuster', help='Sets descender point distance for lines.',default=20)
    parser.add_argument('--use_border_padding', help='Set border padding usage for cropped lines.',default=False)
    parser.add_argument('--generate_border_padding_size', help='Sets size of border padding.',default=5)
    parser.add_argument('--border_padding_mode', help='Sets border padding background type(fex. constant or edge).',default='constant')
    parser.add_argument('--greyscale_for_border_padding', help='Sets border background color.',default=0.9)
    parser.add_argument('--use_rotation_angle', help='Set adjust rotation angle for lines according to their angle.',default=False)
    parser.add_argument('--rotate_angle_mode', help='Sets angle rotation background type(fex. constant or nearest)',default='nearest')
    parser.add_argument('--angle_fill_greyscale', help='Sets angle background fill color.',default=0.9)

    args = parser.parse_args()
    
    pipe.htrocr_usage(pipe, args)