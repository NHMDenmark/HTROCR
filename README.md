<a name="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">HTROCR package</h3>

</div>


<!-- ABOUT THE PROJECT -->

## About The Project

For more information go to https://github.com/NHMDenmark/HTROCR/blob/master/README.md

<!-- GETTING STARTED -->

## Quick guide for use in a terminal

Image(s) needs to be either .jpg or .png. A trained model needs to be provided.
In an environment with python:

pip install git+https://github.com/Baeist/NHMD-digitisation.git@lworkspace#egg=htrocr

python

from htrocr.run import NHMDPipeline as pipe

pipe.htrocr_usage(pipe, required_subject_path="path/to/image/or/directory", transcription_model_weight_path="path/to/trocr/model")

Insert paths to images/model in the above. A new directory called "out" will be created, at the terminals location with the results in it.

Full list of possible arguments for htrocr_usage including default values: 
            self, required_subject_path = None, transcription_model_weight_path = None,
            save_images=False, out_dir='./out', out_type='txt', testing=False, segmenter = "precise", transcriber= "visioned", 
            baseline_model_weight_path= "line_segmentation/predictor/net/default.pb",
            transcription_img_processor = "microsoft/trocr-base-handwritten",
            superpixel_confidence_thresh = 0.1, min_textline_height = 10,
            downsize_scale = 0.33, crop_ucph_border = True, crop_ucph_border_size = 545,
            neighbour_connectivity_ratio = 0.5, fixed_interline_dist = 100, max_contour = 5,
            contour_adjuster = 5, descender_point_adjuster = 20, use_border_padding = False,
            generate_border_padding_size = 5, border_padding_mode = "constant", greyscale_for_border_padding = 0.9,
            use_rotation_angle = False, rotate_angle_mode = "nearest", angle_fill_greyscale = 0.9