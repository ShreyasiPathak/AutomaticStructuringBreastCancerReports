# Automatic Structuring of Breast Cancer Radiology Reports
This project is about "Automatic Structuring of Breast Cancer Radiology Reports for Quality Assurance", done as a masters thesis at Data Science group, University of Twente, Netherlands.
Heading_content_identification folder contains codes for heading and content identification of reports.
AutomaticStructuring folder contains codes for automatic structuring of mammography findings. Conditional Random Field used for automatic structuring. The folder has 3 models - a baseline and 2 hierarchical models.
CRF Model B also contains a code (labeling_to_xml.py) to convert the predicted labels of a report to xml format.

The models have been trained on Dutch mammography reports from ZGT hospital in Netherlands.

## For converting free text Dutch mammography reports to structured format, do the following:
Download the repository. Run the code predict_labels.py in CRF Model A or CRF Model B. The func "mainFunc" in predict_labels.py takes the path, where the free text reports are located. For our case, input data is located in data folder. As an output, the models generate a structured report in XML format and this report in stored in the filename "testSample_predicted_output.xml". This output file gets stored in the folder CRF Model A or CRF model B, depending on which model you are using for automatic structuring. 
Output files for our sample data (sample data can be found in data folder) can be found in folder CRF Model A and CRF Model B, by the filename - testSample_predicted_output.xml

## Description of the folders and the codes
Automatic Structuring: Contains codes for converting free-text reports to structured format
Heading_Content_Identification: Contains codes for predicting if a sentence is a heading or not heading or title, predicts given a sentence, which section of the report does it belong to - clinical data, findings, conclusion, title

## Automatic Structuring folder
CRF Model A: Codes for hierarchical model A (details of the model found in our paper)
CRF Model B: Codes for hierarchical model B (details of the model found in our paper)
CRF Baseline: Codes for baseline model (details of the model found in our paper)
data: Contains sample input data in XML format (Dutch free text mammography report) - testSample_input.xml
      Contains sample ground truth structured report in XML format (labeled by radiologists) - testSample_groundtruth.xml

## CRF Model A, CRF Model B and CRF Baseline folder
CRF_features_cascadedCRF.py: features used for training CRF models
CRF_measures_cascadedCRF.py: performance metrics of the models - token level and phrase level performance
CRF_advancedmodel1_onpredicted.py: training and prediction of the models. Comparing the predicted structure to the groundtruth to find the performance of the model.
labeling_to_xml.py: Converting the predicted labels to a pretty XML format and saving to to a xml file by the name "testSample_predicted_output.cml"
predict_labels.py: Can be used by any user to predict the structure of free-text mammography report. Call the function mainFunc.
CRFmodelA_trainedmodel.pkl and CRFmodelB_trainedmodel.pkl: Contains the trained model on Dutch mammography dataset

## Paper
Automatic Structuring of Breast Cancer for Radiology Reports for Quality Assurance (https://ieeexplore.ieee.org/abstract/document/8637387)
For more details on the work, please refer to the masters thesis in the following link (https://essay.utwente.nl/76327/)

# Citation:
Chicago citation:
Pathak, Shreyasi, Jorit van Rossen, Onno Vijlbrief, Jeroen Geerdink, Christin Seifert, and Maurice van Keulen. "Automatic Structuring of Breast Cancer Radiology Reports for Quality Assurance." In 2018 IEEE International Conference on Data Mining Workshops (ICDMW), pp. 732-739. IEEE, 2018.

Bibtex:
@inproceedings{pathak2018automatic,
  title={Automatic Structuring of Breast Cancer Radiology Reports for Quality Assurance},
  author={Pathak, Shreyasi and van Rossen, Jorit and Vijlbrief, Onno and Geerdink, Jeroen and Seifert, Christin and van Keulen, Maurice},
  booktitle={2018 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={732--739},
  year={2018},
  organization={IEEE}
}
