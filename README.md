# ReRead: An Efficient Vision-Language <ins>RE</ins>finement Scheme for <ins>RE</ins>tinal Found<ins>A</ins>tion Mo<ins>D</ins>els

## SOTA Fine-tuned weights
If you want to skip the fine-tuning step and just want the retinal FM vision weights resulting from our fine-tuning scheme:  
**Fine-tuned weights for RETFound**: https://drive.google.com/file/d/1XEkemQqEZMf_ayJPS9udlkQAzCL2HRsf/view?usp=drive_link
**Fine-tuned weights for VisionFM**: https://drive.google.com/file/d/1KPxULUnhiU0IrlVuA1vy687niSkJNJdz/view?usp=drive_link


## Fine-tuning
If you want to run our fine-tuning scheme on your vision model:

Navigate into ReRead/

Create a new virtual environment in ReRead/ and install requirements.txt

Text encoder weights: Download BERT weights here and put them under ReRead/pretrained_weights/:   Will be released upon acceptance.

Vision encoder weights: Put your vision model in ReRead/  

Our in-house image-text training data is private so you will need to use your own. Edit the dataloader in ReRead/ImageCaptionDataset.py accordingly. __getitem__ should return a list consisting of two elements: an image (torch tensor) and a report (string).

Then in the command line run:
```sh
python train.py --model_weights path/to/yourvisionmodel
```

Once your model is trained, run the following script to extract the vision backbone. This will save it under ../linear_probing/_weights. Note this has only been tested on RETFound, VisionFM, and our in-house MAE. You may need to alter it for another FM.
```sh
python get_vision_backbone_for_linprobing.py --path_to_model models/<model name>/best-model.ckpt
```

## Linear probing

Once you have your fine-tuned model, navigate into ../linear_probing/, set up a new virtual environment there, and then activate it. Then install requirements.txt.

Then you can run one of the .sh scripts based on which model you have.

For example, in retfound.sh, you would change the ft_weights arg to _weights/<my_model_name>. Adjust the data sets arg accordingly.

Results are found in __results/.

## Linear probing datasets
Duke iAMD: https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm  
Harvard Glaucoma: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP  
Noor Eye Hospital: https://hrabbani.site123.me/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme  
OCTDL: https://data.mendeley.com/datasets/sncdhf53xc/4  
OCTID: https://borealisdata.ca/dataverse/OCTID  
NEHUT: https://data.mendeley.com/datasets/8kt969dhx6/1
