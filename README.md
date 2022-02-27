# Image Registration Using ITK

## Abstract

The following repository consists of implementation of image registration of two image volumes from different MRI modalities using ITK. The registration algorithm was implemented based on ITK and the components of the framework were selected based on visual inspection along with theoretical validation and trial and error.

                    Before Image Registration                                          After Image Registration

<img src="https://user-images.githubusercontent.com/52663918/155876226-7fb5039a-d0dd-49bd-9474-739ba96267a3.gif" width="450"/> <img src="https://user-images.githubusercontent.com/52663918/155876235-203408ed-c84c-4829-a4d1-28e277b67958.gif" width="450"/>

<img src="https://user-images.githubusercontent.com/52663918/155876251-c4b0a181-b08b-413c-9378-00214a3f6857.gif" width="450"/> <img src="https://user-images.githubusercontent.com/52663918/155876263-06d02a36-354c-4971-a381-b6d1ec105925.gif" width="450"/>

## Dataset

Here, 2 MRI volumes of the same subject acquired as a part of Female dataset of Visible Human Project are used. The volumes are acquired using 2 different MRI modalities: 1) T2-Weighted MRI and 2) T1-Weighted MRI.  Here, the task of registering T2-Weighted MRI volume **(Moving Image)**
 to a T1-Weighted MRI volume **(Fixed Image)** is conducted.  

<img src="Image%20Regi%204d8c5/git1.png" width="700"/>


## Getting Started

Demo code with scripts (Colab): 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jathurshan0330/Image-Registration-ITK/blob/master/DEMO_Image_Registraionipynb.ipynb)

Image Registration Algorithm Development (Colab): 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jathurshan0330/Image-Registration-ITK/blob/master/170248G_Image_Registration.ipynb)

### Installation
```
pip install -r requirements.txt
```
### Script for Image Registration

The parameters for the registration framework are finetuned and set as default values. 

- Transformation : itk.TranslationTransform
- Interpolator : itk.LinearInterpolateImageFunction
- Metric : itk.MattesMutualInformationImageToImageMetricv4
- Optimizer : itk.RegularStepGradientDescentOptimizerv4
- Registration : itk.ImageRegistrationMethodv4

Run the following script for image registration:
```
python registraion.py --fixed_img_path "Path to Fixed Image in .vtk" --moving_img_path "Path to Moving Image in .vtk" --output_path "Path to Output"
```
### Parameters
Input arguements
```
parser.add_argument('--fixed_img_path', type=str, help='Path fixed image')
parser.add_argument('--moving_img_path', type=str,help='Path moving image')
parser.add_argument('--output_path', type=str, help='Path to save the registered image')
```

Optimization parameters
```
--lr => Learning rate optimizer, default = 4 
--Min_step_length => Minimum Step Length, default = 0.0005 
--relax_factor => Relaxation Factor, default = 0.5
--num_iter => Number of Iterations, default = 200 
```

Metric parameters
```
--num_hist_bins =>  Number of Histogram Bins, default = 24
```
Registration parameters
```
--num_levels => Number of Levels, default = 1 
--smooth_sigma_level => Smoothing Sigmas Per Level, default = 0 
--shrink_level => Shrink Factors Per Level, default = 1
```

## Optimization Results

![opt results.JPG](Image%20Regi%204d8c5/opt_results.jpg)

![plot.png](Image%20Regi%204d8c5/plot.png)

## Qualitative Results

![result13.png](Image%20Regi%204d8c5/result13.png)

![result13_2.png](Image%20Regi%204d8c5/result13_2.png)

## Additional Comparisons

![result5.png](Image%20Regi%204d8c5/result5.png)

![16.png](Image%20Regi%204d8c5/16.png)

![19.png](Image%20Regi%204d8c5/19.png)

![result21.png](Image%20Regi%204d8c5/result21.png)

![results26.png](Image%20Regi%204d8c5/results26.png)
