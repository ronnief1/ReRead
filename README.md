![image](https://github.com/user-attachments/assets/53b51e53-bf2b-42d9-bd40-7f87bd004e3d)

This is the official repository of the paper "RetFiner: A Vision-Language Refinement Scheme for Retinal Foundation Models", by Ronald Fecso, José Morano, Ursula Schmidt-Erfurth, and Hrvoje Bogunović, accepted for presentation at [MICCAI 2025](https://conferences.miccai.org/2025/en/).

#### [[`arXiv`](https://arxiv.org/abs/2506.22149)]

## SOTA RetFined weights

If you want to skip the fine-tuning step and just want the retinal FM vision weights resulting from our refinement scheme:

> [!IMPORTANT]
> The weights of the models will be released for the MICCAI 2025 conference, which takes place from September 23 to 27, 2025. Shortly before then, we will update this section with download links.


- **RetFiner-R** (RetFiner-tuned weights for RETFound)
- **RetFiner-V** (RetFiner-tuned weights for VisionFM)
- **RetFiner-U** (RetFiner-tuned weights for UrFound)


## RetFining
If you want to run RetFiner on your vision model:

Navigate into RetFiner/

Create a new virtual environment in RetFiner/ and install requirements.txt

Text encoder weights: Download BERT weights here and put them under RetFiner/pretrained_weights/:   Will be released upon acceptance.

Vision encoder weights: Put your vision model in RetFiner/

Our in-house image-text training data is private so you will need to use your own. Edit the dataloader in RetFiner/ImageCaptionDataset.py accordingly. __getitem__ should return a list consisting of two elements: an image (torch tensor) and a report (string).

Then in the command line run:
```sh
python train.py --model_weights path/to/yourvisionmodel
```

Once your model is trained, run the following script to extract the vision backbone. This will save it under ../linear_probing/_weights. Note this has only been tested on RETFound, VisionFM, and our in-house MAE. You may need to alter it for another FM.
```sh
python get_vision_backbone_for_linprobing.py --path_to_model models/<model name>/best-model.ckpt
```

## Linear probing

Once you have your RetFined model, navigate into ../linear_probing/, set up a new virtual environment there, and then activate it. Then install requirements.txt.

Then you can run one of the .sh scripts based on which model you have.

For example, in retfound.sh, you would change the ft_weights arg to _weights/<my_model_name>. Adjust the data sets arg accordingly.

Results are found in __results/.

## Linear probing datasets

- Duke iAMD: https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm
- Harvard Glaucoma: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP
- Noor Eye Hospital: https://hrabbani.site123.me/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme
- OCTDL: https://data.mendeley.com/datasets/sncdhf53xc/4
- OCTID: https://borealisdata.ca/dataverse/OCTID
- NEHUT: https://data.mendeley.com/datasets/8kt969dhx6/1



## License

The models and associated code are released under the CC-BY-NC 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. See [LICENSE](LICENSE) for more details.

Specifically, the following licenses apply to the base models and fine-tuned models:


| Model Name      | Base Model | Original License | Fine-Tuned License |
|-----------------|------------|------------------|---------------------|
| **RetFiner-U**  | [UrFound](https://github.com/yukkai/UrFound) | [MIT](https://opensource.org/licenses/MIT) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| **RetFiner-R**  | [RETFound](https://github.com/rmaphoh/RETFound_MAE) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| **RetFiner-V**  | [VisionFM](https://github.com/ABILab-CUHK/VisionFM) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |



## Citation

If you use any of our models, please do the following:

1. **Cite the original base models**:
   - **UrFound**: Yu, Kai, et al. "UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2024.
   - **RETFound**: Zhou, Yukun, et al. "A foundation model for generalizable disease detection from retinal images." Nature 622.7981 (2023): 156-163.
   - **VisionFM**: Qiu, Jianing, et al. "Development and validation of a multimodal multitask vision foundation model for generalist ophthalmic artificial intelligence." NEJM AI 1.12 (2024): AIoa2300221.
2. **Cite this work**:
   ```bibtex
   @misc{fecso2025retfiner,
         title={{RetFiner}: A Vision-Language Refinement Scheme for Retinal Foundation Models}, 
         author={Ronald Fecso and José Morano and Ursula Schmidt-Erfurth and Hrvoje Bogunović},
         year={2025},
         eprint={2506.22149},
         archivePrefix={arXiv},
         primaryClass={cs.CV},
         url={https://arxiv.org/abs/2506.22149},
   }
   ```
