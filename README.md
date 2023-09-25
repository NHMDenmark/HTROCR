# HTROCR Package

## About The Project

This is a package meant for transcribing herbarium sheets with handwritten notes in them for NHMD.
In order to make use of the package you must have a trained trocr model. 
For more information go to [Original Project](https://github.com/NHMDenmark/HTROCR/blob/master/README.md)

## Quick guide for use in a terminal setting

Image(s) needs to be either .jpg or .png. A trained model needs to be provided.
In an environment with python install the package

```bash
pip install git+https://github.com/NHMDenmark/HTROCR.git@package#egg=htrocr
```

Then enter python mode

```bash
python
```

Finally make use of the package

```python
from htrocr.run import NHMDPipeline as pipe

#Insert paths to images/model. A new directory called ./out will be created. Out will include both txt files with the results and also optionally images of the found lines.
#Please note that the crop_ucp_border is initially set to true, which crops the border(right side) found on ucp prepared documents.
pipe.htrocr_usage(pipe, required_subject_path="path/to/image/or/directory", transcription_model_weight_path="path/to/trocr/model/directory/path")
```

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

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)
