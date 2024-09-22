# UAV_classification_with_Mamba

## Abstract

Welcome to my project!
This was the final project of my Masters degree, advised by Dr. Alon Kipnis.
Its goal was to classify flying objects (bird, airplane, UAV or static) with specific attention to detecting unmanned aerial vehicles (UAVs).
The data at our disposal is the flight trajectories data of the objects, extracted from a tracking device.
The project included an initial phase of processing and feature extraction for evaluating baseline methods (DT, RF, XGB, SVM, MLP).
The main focus though was using a generative Mamba model to create class-specific predictions and classify an object by error comparison. 

To review the project I recommend going through its stages as follows:
- Read the report at: [`PAMAP2_data_statistics.ipynb`](PAMAP2_data_statistics.ipynb)
- Review the EDA notebook and the preprocessing and visualization notebook for familiarity with the data
- Review the main version of preprocessing
- Evaluate the results obtained with the best feature-based models: RF, XGB
- Explore the adjusted Mamba model
- Follow a Mamba model training process
- See the classification method
- Evaluate the final results


  
It reproduces results for the following paper on the subject of Human Activity Recognition (HAR):

Haojie Ma, Wenzhong Li, Xiao Zhang, Songcheng Gao, and Sanglu Lu. 2019. **AttnSense: Multi-level Attention Mechanism For Multimodal Human Activity Recognition**. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019. 3109–3115.

Our report can be found here:
[AttnSense Multi-level Attention Mechanism For Multimodal Human Activity.pdf](AttnSense%20Multi-level%20Attention%20Mechanism%20For%20Multimodal%20Human%20Activity.pdf)

## The Notebooks

To run the notebooks locally, you first need to get the PAMAP2 dataset (see instructions below).

- [`PAMAP2_data_statistics.ipynb`](PAMAP2_data_statistics.ipynb) - This notebook analyzes the data and shows interesting statistics.
- [`PAMAP2_data_preprocessing.ipynb`](PAMAP2_data_preprocessing.ipynb) - This notebook performs all preprocessing of the data. It includes optional variations of:
   - Adding augmentation + controlling noise magnitude
   - Changing Sequence length
   - Choosing different subject as test set
   - With or without FFT
   etc.
   

Enjoy!
