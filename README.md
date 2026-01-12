# Interpretable artificial intelligence-based determination of glioma IDH mutation status directly from histology slides


## Environment
## Pre-requisites:
* Linux (Tested on Ubuntu 22.04)
* NVIDIA GPU (Tested on Nvidia A6000)
* Python (3.7.5), h5py (2.10.0), matplotlib (3.1.1), numpy (1.17.3), opencv-python (4.1.1.26), openslide-python (1.1.1), openslide (3.4.1), pandas (0.25.3), pillow (6.2.1), PyTorch (1.3.1), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.4.2), smooth-topk.

After setting up anaconda, first install openslide:
```bash
sudo apt-get install openslide-tools
```

Next, use the environment configuration file located in docs/clam.yaml to create a conda environment:
```bash
conda env create -n clam -f docs/idh_classifier.yaml
```

Activate the environment:
```bash
conda activate clam
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

```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```


The **patches** folder contains arrays of extracted tissue patches from each slide (one .h5 file per slide, where each entry corresponds to the coordinates of the top-left corner of a patch)
The **stitches** folder contains downsampled visualizations of stitched tissue patches (one image per slide) (Optional, not used for downstream tasks)
The auto-generated csv file **process_list_autogen.csv** contains a list of all slides processed, along with their segmentation/patching parameters used.

### 🧹 Patch Cleanup (Step 2)
After initial patching, the pipeline runs a **Cleanup Script** to filter out low-quality tiles.
#### Filtering Criteria:
1. **White Space**: Patches with >85% background are removed.
2. **Stain Detection**: Uses HED (Hematoxylin-Eosin-DAB) color deconvolution to ensure tissue is actually present.
3. **HSV Filtering**: Removes blurry or out-of-focus areas based on saturation and value thresholds.

#### Why this is necessary:
Whole Slide Images often contain artifacts, marker ink, or large empty regions. By cleaning the `.h5` files, you reduce the noise in your training set and significantly speed up the feature extraction (encoding) step.

## Creating Features
Imagenet
SimCLR
HIPT4k
LunitViT
CTranspath
UNI
RetCCL

## Training the models
Attention MIL
Mean MIL
MaxMIL
TransMIL
DSMIL

## Evaluation 


## Hovernet


## Acknowledgement


## Citations


