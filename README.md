<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">NHMD Digitisation</h3>

  <p align="center">
    Historical label recognition of herbarium sheets from Natural History Museum of Denmark
  </p>

  <p>

![Pipeline run example][pipeline-example]

  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Setup</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

![Product Name Screen Shot][pipeline-screenshot]

NHMD herbarium sheet digitisation pipeline. Solution is based on a 2 stage procedure. First, the text line baselines are detected within the document by relying on the tranformed ARU-Net baseline detector [[1]](#1) pipeline. Baselines are then transformed into text line segmentations. The resulting lines may have ascender and descender features from other lines, however, it has been shown by Romero et al. [[2]](#2) that these have little influence. Obtained images are then forwarded to transcription module, where different transcription variants were proposed. The best performing method is based on TrOCR BASE [[3]](#3) configuration.

The project was done as part of MSc thesis: "Herbarium sheet label data digitisation using handwritten text recognition" at University of Copenhagen. Results for transcription module displayed 11.93% Character Error Rate performance on a custom 817 NHMD line sample dataset. However, depending on a document, an addition of a segmentation tool may reduce this performance due to incorrect segmentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

- Python 3.9

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/LinasEin/NHMD-digitisation.git
   ```
2. Prepare conda environment
   ```sh
   conda create -n test_env python=3.9
   pip install -r requirements.txt
   ```
3. Execute pipelne using run.py script
   ```sh
   python run.py process_image --path="./path/to/image.jpg" --out_type="txt" --save_images=True --out_dir='./out'
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

- Run the pipeline on a single sample:

  ```sh
   python run.py process_image --path="./path/to/image.jpg" --out_type="txt" --save_images=True --out_dir='./out'
  ```

- Run the pipeline on a folder of images:

  ```sh
   python run.py process_dir --path="./path/to/folder" --out_type="txt" --save_images=True --out_dir='./out'
  ```

- Supported --out_types: "<i>txt</i>", "<i>xml</i>"

- Sample right border processing is not supported. If samples to be transcribed do not have this border - make sure to change the pipeline_config.json and set "crop_ucph_border" to <b>false</b>.

- In the same pipeline_config.json processing parameters can be adjusted.

- Use command "evaluate_baseline(s)" instead of "process_image(dir)" to prepare predictions for accuracy evaluation using baseline evaluation scheme.
  ```sh
   python -u run.py evaluate_baselines --path="./path"
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Known issues

- Rotated labels are not normalised before transcription.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Linas Einikis - einikis.lin@gmail.com

Project Link: [https://github.com/LinasEin/NHMD-digitisation](https://github.com/LinasEin/NHMD-digitisation)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

- [RETRO](https://www.retrodigitalisering.dk/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

<a id="1">[1]</a>
T. Grüning, G. Leifert, T. Strauß, J. Michael, and R. Labahn, “A two- stage method for text line detection in historical documents,” Interna- tional Journal on Document Analysis and Recognition (IJDAR), vol. 22, pp. 285–302, Sep 2019.

<a id="2">[2]</a>
V. Romero, J. A. Sánchez, V. Bosch, K. Depuydt and J. de Does, "Influence of text line segmentation in Handwritten Text Recognition," 2015 13th International Conference on Document Analysis and Recognition (ICDAR), Tunis, Tunisia, 2015, pp. 536-540, doi: 10.1109/ICDAR.2015.7333819.

<a id="3">[3]</a>
M. Li, T. Lv, L. Cui, Y. Lu, D. A. F. Florêncio, C. Zhang, Z. Li, and F. Wei, “Trocr: Transformer-based optical character recognition with pre-trained models,” CoRR, vol. abs/2109.10282, 2021.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[pipeline-screenshot]: resources/pipeline.png
[pipeline-example]: resources/example.png
