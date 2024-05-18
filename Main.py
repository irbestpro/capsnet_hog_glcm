'''These Codes Are Written By Mehdi Touyserkani
    Email Address: Ir_Bestpro@yahoo.com
    Website: Https://www.Ir-Bestpro.com
 '''

#_____Import SkLearn_________________

from skimage import io
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.transform import resize
from sklearn.model_selection import cross_val_score

#____Import Loader____________________

import loader

#_____Import Tensorflow in python_____

import tensorflow as tf

#____Import torch Library____________

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#______Import SVM Classifier And PCA_______

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#___Import Operating System Variable_

import os
import numpy as np
import numpy

from HOG_utils import(Direction,HOG,Magnitude,gradient)
from GLCM_utils import GLCM
from Capsnet import (Capsnet,Config)

#____Ignore Warnings_____________

import warnings
warnings.filterwarnings("ignore")

#______Data Files______________

path_to_data = os.path.join('.', 'data')
path_to_jpg = os.path.join(path_to_data, 'raw_jpg')
path_to_preprocessed = os.path.join(path_to_data, 'preprocessed')
image_height = 128
image_width = 128
binary_thresh = 3
gaussian_sigma = 5

#_______glcm Histogram Data________

glcmData = []

#_____HOG Histogram Data__________

hogData = []

#_____Capsnet Features____________

capsData = []

#______Collect Data________________

collected = []

#_______Preprocessing_______________

def preprocess(image):
    image = image[20:-25, 25:-25, :]
    assert isinstance(image, np.ndarray)
    grayscale = rgb2gray(image)
    grayscale = gaussian(grayscale, sigma=gaussian_sigma) # Image Smoothing

    binary = grayscale * 255 > binary_thresh

    labelled = label(binary)
    regions = regionprops(labelled)
    areas = [region.area for region in regions]
    arg = int(np.argmax(areas))
    h1, w1, h2, w2 = regions[arg].bbox
    extracted = image[h1: h2, w1: w2, :]
    return (resize(extracted, (image_height, image_width)) * 255).astype(np.uint8)

##_____Create Classification Models_____________________

def make_classification_model(X,Y):
    svclassifier = SVC(kernel='rbf')#,gamma= 1)#gamma = 10 
    return cross_val_score(svclassifier, X, Y, cv=5).mean() # K-Fold Cross Validation

def make_dt_classification_model(X,Y):
    classifier = tree.DecisionTreeClassifier()
    return cross_val_score(classifier, X, Y, cv=5).mean() # K-Fold Cross Validation

def make_rf_classification_model(X,Y):
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    return cross_val_score(classifier, X, Y, cv=5).mean() # K-Fold Cross Validation

def make_knn_classification_model(X,Y):
    classifier = KNeighborsClassifier(n_neighbors=3)
    return cross_val_score(classifier, X, Y, cv=5).mean() # K-Fold Cross Validation

##_______Feature Selection________________________

def apply_PCA(X):

    ##____Apply Principle Component Analysis_(PCA)______

    pca = PCA(n_components = X.shape[1],svd_solver='full') # use n Components
    X_pca = pca.fit_transform(X) #apply pca Transform
    explained = pca.explained_variance_ratio_ #Explained Variance Of Model
    #Scores = pca.score_samples(pca)

    pricipal = 90
    pspace = explained[0]
    Ind = 1

    for i in range(1,len(explained)):
        if pspace < pricipal:
            pspace = explained[i] + pspace ##Cumulative Sum
            Ind = i
        else:
            break

    return X_pca[:,0:Ind]

##______OverSpampling Method_______________________________

def over_sampling(X,Y):
    
    ##_____OverSampling Method(SMOTE)______________________
    
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0) 
    balanced_Data, balanced_Labels = ros.fit_resample(X, Y)# Create Artificial Samples
    return balanced_Data,balanced_Labels

##_____Start Source Codes___________________________________

if __name__ == '__main__':
    
    ##_____Loading Train And Test Images/Labels From Preprocessed Data________
    
    (train_images, train_labels), (test_images, test_labels) = loader.make_dataset(loader.load())

    all_data = np.vstack((train_images,test_images))
    all_labels = np.vstack((train_labels,test_labels))
    all_labels = np.argmax(all_labels, axis = -1) # Convert Binary Categorical to Integer

    ##______Capsule Network Initialization________________________
    
    config = Config.Config()# Get Best Network Configuration
    capsule_net = Capsnet.CapsNet(config) # Create Capsnet Model With Input Parameters
    optimizer = torch.optim.Adam(capsule_net.parameters())
    optimizer.zero_grad()

    ##_____glcm Initialization_____________________________________

    pattern = GLCM.GLCM(8, 3) ## Calling with P = 8 And R = 3 (Standard Parameters Values)

    ##______HOG Initialization___________________________________

    h_Filter = numpy.array([-1, 0, 1]) ##  Horizontal Sobel Filter Mask
    v_Filter = numpy.array([[-1],   ## Vertical Sobel Filter Mask
                             [0],
                             [1]])
    
    hist_bins = numpy.array([10,30,50,70,90,110,130,150,170]) #  Set Degress Array With 9 Bins from 10-170 (Can Be between 0 and 160!)

    ##________Start Extracting____________________________________

    counter = 0
    for data in all_data:
        
        ##____Calling Traditional Local Binary Patterns in skimage ToolBox______
        
        glcm_statistics = pattern.Pattern(data) ## Extract Preprocessed Image Pattern By Histogram
        glcmData.append(glcm_statistics) ## Append Pattern To List

        ##______Calling Traditional Histogram Of Oriented Gradients(HOG) toolBox____________

        h_gradient = gradient.calculate_gradient(data, h_Filter)  ## Calculate Horizontal Gradient
        v_gradient = gradient.calculate_gradient(data, v_Filter) ## Calculate Vertical Gradient

        grad_magnitude = Magnitude.magnitude(h_gradient, v_gradient) #Magnitude (SQRT(x^2 + Y^2))
        grad_direction = Direction.direction(h_gradient, v_gradient) # Direction --> arctan(gy / gx)

        directions = grad_direction % 180  # Divide  Direction By 180 For direction Normalization

        ## _____Calculate Gradient Histograms By 9 Bins__________

        direction = grad_direction[:8, :8]
        magnitude = grad_magnitude[:8, :8]
        HOG_cell_hist = HOG.HOG_cell_histogram(direction, magnitude, hist_bins)

        ##____Append HOG Bins By Every Images__________

        hogData.append(HOG_cell_hist.flatten())

        ##_____CapsNet Feature Extraction_______________

        imgs = torch.from_numpy(data)
        imgs = imgs.unsqueeze(0).unsqueeze(0)
        output = capsule_net(Variable(imgs),all_labels[counter]) # train capsnet and Create Model
        temp_output = output.view(160).detach().numpy()

        ##_____Append capsData _________________________

        capsData.append(temp_output)

        ##_____Collect Features_________________________

        temp = np.concatenate((HOG_cell_hist,glcm_statistics))
        temp1 = np.concatenate((temp_output,temp))

        ##_______Collect Data_________________________________

        collected.append(temp1)
        print("Extracting Hog,glcm,Capsule Network Features Iteration #" , counter+1)
        counter = counter + 1


    predictors = collected

    ##____Save Variable In Pickle_________________________

    predictors = np.array(predictors)# All Feature Extractors
    glcmData = np.array(glcmData) #glcm Features
    hogData = np.array(hogData) #Hog Features
    capsData = np.array(capsData) #Caps Net Features
    glcm_hog = np.column_stack((glcmData,hogData)) #glcm+HOG Features
    glcm_caps = np.column_stack((glcmData,capsData))
    hog_caps = np.column_stack((hogData,capsData))

    ##_____SMOTE OverSampling____________________________________

    over_glcm,over_glcm_labels = over_sampling(glcmData,all_labels)
    over_hog,over_hog_labels = over_sampling(hogData,all_labels)
    over_caps,over_caps_labels = over_sampling(capsData,all_labels)
    over_glcm_hog,over_glcm_hog_labels = over_sampling(glcm_hog,all_labels)
    over_glcm_caps,over_glcm_caps_labels = over_sampling(glcm_caps,all_labels)
    over_hog_caps,over_hog_caps_labels = over_sampling(hog_caps,all_labels)
    over_all,over_all_labels = over_sampling(predictors,all_labels)

    ##_____Apply PCA____________________________________________

    over_glcm = apply_PCA(over_glcm)
    over_hog = apply_PCA(over_hog)
    over_caps = apply_PCA(over_caps)
    over_glcm_hog = apply_PCA(over_glcm_hog)
    over_glcm_caps = apply_PCA(over_glcm_caps)
    over_hog_caps = apply_PCA(over_hog_caps)
    over_all = apply_PCA(over_all)

    # ____________Final Classificatio Using SVM__________________

    ##______Hog Features Classification___________________________

    scores_hog = make_classification_model(over_hog,over_hog_labels)

    ##______glcm Features Classification___________________________

    scores_glcm = make_classification_model(over_glcm,over_glcm_labels)

    ##______Hog + glcm Features Classification______________________

    scores_hog_glcm = make_classification_model(over_glcm_hog,over_glcm_hog_labels)

    ##______Caps Features Classification______________________

    scores_caps = make_classification_model(over_caps,over_caps_labels)    

    ##______Caps + glcm Features Classification______________________

    scores_caps_glcm = make_classification_model(over_glcm_caps,over_glcm_caps_labels)

    ##______Caps + HOG Features Classification______________________

    scores_caps_hog = make_classification_model(over_hog_caps,over_hog_caps_labels)

    ##______Caps + Hog + glcm Features Classification_________________

    scores_caps_hog_glcm = make_classification_model(predictors,all_labels)

    #____________Final Classificatio Using Decision Tree_______________

    ##______Hog Features Classification___________________________

    scores_hog_dt = make_dt_classification_model(over_hog,over_hog_labels)

    ##______glcm Features Classification___________________________

    scores_glcm_dt = make_dt_classification_model(over_glcm,over_glcm_labels)

    ##______Hog + glcm Features Classification______________________

    scores_hog_glcm_dt = make_dt_classification_model(over_glcm_hog,over_glcm_hog_labels)

    ##______Caps Features Classification______________________

    scores_caps_dt = make_dt_classification_model(over_caps,over_caps_labels)    

    ##______Caps + glcm Features Classification______________________

    scores_caps_glcm_dt = make_dt_classification_model(over_glcm_caps,over_glcm_caps_labels)

    ##______Caps + HOG Features Classification______________________

    scores_caps_hog_dt = make_dt_classification_model(over_hog_caps,over_hog_caps_labels)

    ##______Caps + Hog + glcm Features Classification_________________

    scores_caps_hog_glcm_dt = make_dt_classification_model(predictors,all_labels)

    #____________Final Classificatio Using Random Forest______________

    ##______Hog Features Classification___________________________

    scores_hog_rf = make_rf_classification_model(over_hog,over_hog_labels)

    ##______glcm Features Classification___________________________

    scores_glcm_rf = make_rf_classification_model(over_glcm,over_glcm_labels)

    ##______Hog + glcm Features Classification______________________

    scores_hog_glcm_rf = make_rf_classification_model(over_glcm_hog,over_glcm_hog_labels)

    ##______Caps Features Classification______________________

    scores_caps_rf = make_rf_classification_model(over_caps,over_caps_labels)    

    ##______Caps + glcm Features Classification______________________

    scores_caps_glcm_rf = make_rf_classification_model(over_glcm_caps,over_glcm_caps_labels)

    ##______Caps + HOG Features Classification______________________

    scores_caps_hog_rf = make_rf_classification_model(over_hog_caps,over_hog_caps_labels)

    ##______Caps + Hog + glcm Features Classification_________________

    scores_caps_hog_glcm_rf = make_rf_classification_model(predictors,all_labels)

    #____________Final Classificatio Using KNN____________________

    ##______Hog Features Classification___________________________

    scores_hog_knn = make_knn_classification_model(over_hog,over_hog_labels)

    ##______glcm Features Classification___________________________

    scores_glcm_knn = make_knn_classification_model(over_glcm,over_glcm_labels)

    ##______Hog + glcm Features Classification______________________

    scores_hog_glcm_knn = make_knn_classification_model(over_glcm_hog,over_glcm_hog_labels)

    ##______Caps Features Classification______________________

    scores_caps_knn = make_knn_classification_model(over_caps,over_caps_labels)    

    ##______Caps + glcm Features Classification______________________

    scores_caps_glcm_knn = make_knn_classification_model(over_glcm_caps,over_glcm_caps_labels)

    ##______Caps + HOG Features Classification______________________

    scores_caps_hog_knn = make_knn_classification_model(over_hog_caps,over_hog_caps_labels)

    ##______Caps + Hog + glcm Features Classification_________________

    scores_caps_hog_glcm_knn = make_knn_classification_model(predictors,all_labels)

    ##________Print Final Results of All Permutations__________________

    from prettytable import PrettyTable
    result = PrettyTable()
    result.field_names = ["No", "Method(with SVM)", "Accuracy"]
    result.add_row([1, "HOG", scores_hog])
    result.add_row([2, "glcm", scores_glcm])
    result.add_row([3, "Caps Net", scores_caps])
    result.add_row([4, "Hog + glcm", scores_hog_glcm])
    result.add_row([5, "HOG + Caps Net", scores_caps_hog])
    result.add_row([6, "glcm + Caps Net", scores_caps_glcm])
    result.add_row([7, "glcm + Caps Net + HOG", scores_caps_hog_glcm])
    print(result)

    result = PrettyTable()
    result.field_names = ["No", "Method(with RF)", "Accuracy"]
    result.add_row([1, "HOG", scores_hog_rf])
    result.add_row([2, "glcm", scores_glcm_rf])
    result.add_row([3, "Caps Net", scores_caps_rf])
    result.add_row([4, "Hog + glcm", scores_hog_glcm_rf])
    result.add_row([5, "HOG + Caps Net", scores_caps_hog_rf])
    result.add_row([6, "glcm + Caps Net", scores_caps_glcm_rf])
    result.add_row([7, "glcm + Caps Net + HOG", scores_caps_hog_glcm_rf])
    print(result)

    result = PrettyTable()
    result.field_names = ["No", "Method(with Decision Tree)", "Accuracy"]
    result.add_row([1, "HOG", scores_hog_dt])
    result.add_row([2, "glcm", scores_glcm_dt])
    result.add_row([3, "Caps Net", scores_caps_dt])
    result.add_row([4, "Hog + glcm", scores_hog_glcm_dt])
    result.add_row([5, "HOG + Caps Net", scores_caps_hog_dt])
    result.add_row([6, "glcm + Caps Net", scores_caps_glcm_dt])
    result.add_row([7, "glcm + Caps Net + HOG", scores_caps_hog_glcm_dt])
    print(result)

    result = PrettyTable()
    result.field_names = ["No", "Method(with KNN)", "Accuracy"]
    result.add_row([1, "HOG", scores_hog_knn])
    result.add_row([2, "glcm", scores_glcm_knn])
    result.add_row([3, "Caps Net", scores_caps_knn])
    result.add_row([4, "Hog + glcm", scores_hog_glcm_knn])
    result.add_row([5, "HOG + Caps Net", scores_caps_hog_knn])
    result.add_row([6, "glcm + Caps Net", scores_caps_glcm_knn])
    result.add_row([7, "glcm + Caps Net + HOG", scores_caps_hog_glcm_knn])
    print(result)

    ###_________________________________________________________


    
    

    
