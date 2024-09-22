# UAV classification with Mamba

## Abstract

Welcome to my project!\
This was the final project of my Masters degree, advised by Dr. Alon Kipnis.\
Its goal was to classify flying objects (bird, airplane, UAV or static) with specific attention to detecting unmanned aerial vehicles (UAVs).\
The data at our disposal is the flight trajectories data of the objects, extracted from a tracking device.\
The project included an initial phase of processing and feature extraction for evaluating baseline methods (DT, RF, XGB, SVM, MLP).\
The main focus though was using a generative Mamba model to create class-specific predictions and classify an object by error comparison.

## The Notebooks
The project's notebooks were used in a google colab environment and are presented here as is.\
To review the project I recommend going through its stages as follows:
- Read the [report](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Final_Project___UAV_classification_with_Mamba.pdf)
- Review the [EDA](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Basic%20EDA%20and%20Dataset%20Analysis%20-%20full.ipynb) notebook and the [Preprocessing and Visualization](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Preprocessing%20and%20Visualization%20-%20full.ipynb) notebook for familiarity with the data
- Review the main version of [preprocessing](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Mamba%20Preprocessing/Preprocessing%20for%20Mamba.ipynb)
- Follow the feature extraction process and classification for the best feature-based models: [RF](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Baseline%20Models%20Training/UAV%20project%20full%20RF%20all%20lengths%20Mamba%20compatible%20dataset.ipynb), [XGB](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Baseline%20Models%20Training/UAV%20project%20full%20XGB%20all%20lengths%20Mamba%20compatible%20dataset.ipynb)
- Compare [baseline methods' results](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Analysis%20and%20plotting/feature_extraction_results_analysis%20full%20Mamba%20compatible%20-%20ALL.ipynb)
- Explore the adjusted [Mamba model](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Mamba%20Model.ipynb)
- Follow a [Mamba model training](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Mamba%20Training/Mamba%20UAV%20training.ipynb) process
- See the [Mamba classification](https://github.com/ayalaraanan/UAV_classification_with_Mamba/blob/main/Mamba%20Classification/Mamba_Classification%20combinations%20optimization.ipynb) method and results

![alt text](http://url/to/img.png)


Enjoy!
