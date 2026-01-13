# The Code is currently in development stage.

# Interpretable artificial intelligence-based determination of glioma IDH mutation status directly from histology slides



## Environment
## Pre-requisites:
* Linux (Tested on Ubuntu 22.04)
* NVIDIA GPU (Tested on Nvidia A6000)

After setting up anaconda, first install openslide:
```bash
sudo apt-get install openslide-tools
```

Next, use the environment configuration file located in docs/clam.yaml to create a conda environment:
```bash
conda env create -n idh_classifier -f docs/idh_classifier.yaml
```

Activate the environment:
```bash
conda activate idh_classifier
```

Once inside the created environment, to install smooth-topk (first cd to a location that is outside the project folder and is suitable for cloning new git repositories):
```bash
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install
```

## WSI Segmentation and Patching 



```bash
data/slides_20x/
	├── patient_1_slide_a.svs
	├── patient_1_slide_b.svs
	└── ...
data/slides_40x/
	├── patient_2_slide_a.svs
	├── patient_2_slide_b.svs
	└── ...
```

### 🛠 Workflow Logic

The pipeline automatically adjusts the extraction scale and file paths based on the input magnification. 
This ensures that the physical area covered by a patch remains consistent or follows your specific protocol.

| Input Argument | Target Data Directory | Output Directory | Patch Size |
| :--- | :--- | :--- | :--- |
| `20x` | `data/slides_20x/` | `data/slides_patches_20x/` | **256** |
| `40x` | `data/slides_40x/` | `data/slides_patches_40x/` | **512** |


### Execution Examples
```shell
# Process 20x slides with 256px patches
./create_patches.sh 20x

# Process 40x slides with 512px patches
./create_patches.sh 40x
```
By setting `20x` to `256` and `40x` to `512`, you are effectively keeping the **field of view (FOV)** of each patch identical in terms of physical microns (assuming the 40x scan has twice the resolution of the 20x scan). This is a standard best practice in pathology machine learning to ensure the model sees the same amount of tissue per tile regardless of the scanner settings.

#### Output Directory Structure
```bash
data/slides_patches_20x/
	├── masks
    		├── patient_1_slide_a.png
    		├── patient_1_slide_b.png
    		└── ...
	├── patches
    		├── patient_1_slide_a.h5
    		├── patient_1_slide_b.h5
    		└── ...
	├── stitches
    		├── patient_1_slide_a.png
    		├── patient_1_slide_b.png
    		└── ...
	└── slides_processed.csv

data/slides_patches_40x/
├── masks
        ├── patient_1_slide_a.png
        ├── patient_1_slide_b.png
        └── ...
├── patches
        ├── patient_1_slide_a.h5
        ├── patient_1_slide_b.h5
        └── ...
├── stitches
        ├── patient_1_slide_a.png
        ├── patient_1_slide_b.png
        └── ...
└── slides_processed.csv
```

### 🧹 Patch Cleanup (Step 2)
After initial patching, the pipeline runs a **Cleanup Script** to filter out low-quality tiles.
#### Filtering Criteria:
1. **White Space**: Patches with >85% background are removed.
2. **Stain Detection**: Uses HED (Hematoxylin-Eosin-DAB) color deconvolution to ensure tissue is actually present.
3. **HSV Filtering**: Removes blurry or out-of-focus areas based on saturation and value thresholds.

#### Why this is necessary:
Whole Slide Images often contain artifacts, marker ink, or large empty regions. By cleaning the `.h5` files, you reduce the noise in your training set and significantly speed up the feature extraction (encoding) step.

## Creating Features
Run the extraction script by specifying magnification, batch size, and the desired model backbone. The script dynamically maps to the correct data and coordinate directories based on the magnification provided.

### Usage
```shell
./extract_features.sh <MAG> <BATCH_SIZE> <BACKBONE>
```
Example: 
```shell
./extract_features.sh 20x 256 uni
```

### Supported Backbones

We support several **state-of-the-art self-supervised and supervised models** for histopathology.  
For more details about each model, please refer to the original repositories to request access and follow their specific licensing terms.

- **UNI** : [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
- **CTransPath** : [https://github.com/Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath)
- **RetCCL** : [https://github.com/Xiyue-Wang/RetCCL](https://github.com/Xiyue-Wang/RetCCL)
- **HIPT-4K** : [https://github.com/mahmoodlab/HIPT](https://github.com/mahmoodlab/HIPT)
- **Lunit ViT** : [https://github.com/lunit-io/benchmark-ssl-pathology](https://github.com/lunit-io/benchmark-ssl-pathology)
- **SimCLR** : [https://github.com/ozanciga/self-supervised-histopathology](https://github.com/ozanciga/self-supervised-histopathology)
- **ResNet-50** : ImageNet pretrained

## Training the models

### Usage Instructions
To run the training script, pass the **magnification level** and **backbone name** as arguments:
```bash
chmod +x train.sh
./train.sh <MAG> <BACKBONE>
```

Example: 
```bash
./train.sh 20x uni
```
### Methodology
This codebase is heavily based on CLAM(https://github.com/mahmoodlab/CLAM/). However, unlike CLAM, instance-level clustering is not used.
The model is trained using a pure Attention-based Multiple Instance Learning (MIL) framework.

## Evaluation 


## Hovernet


## Acknowledgement

## References

## Citations

Shubham Innani, W Robert Bell, Hannah Harmsen, MacLean P Nasrallah, Bhakti Baheti, Spyridon Bakas, Interpretable artificial intelligence based determination of glioma IDH mutation status directly from histology slides, Neuro-Oncology Advances, Volume 7, Issue 1, January-December 2025, vdaf140, https://doi.org/10.1093/noajnl/vdaf140

```bash

@article{10.1093/noajnl/vdaf140,
    author = {Innani, Shubham and Bell, W Robert and Harmsen, Hannah and Nasrallah, MacLean P and Baheti, Bhakti and Bakas, Spyridon},
    title = {Interpretable artificial intelligence based determination of glioma IDH mutation status directly from histology slides},
    journal = {Neuro-Oncology Advances},
    volume = {7},
    number = {1},
    pages = {vdaf140},
    year = {2025},
    month = {07},
    issn = {2632-2498},
    doi = {10.1093/noajnl/vdaf140},
    url = {https://doi.org/10.1093/noajnl/vdaf140},
    eprint = {https://academic.oup.com/noa/article-pdf/7/1/vdaf140/63738190/vdaf140.pdf},
}
```

