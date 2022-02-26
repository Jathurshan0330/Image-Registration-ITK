# Image Registration Using ITK

## Abstract

The following repository consists of implementation of image registration of two image volumes from different MRI modalities using ITK. The registration algorithm was implemented based on ITK and the components of the framework were selected based on visual inspection along with theoretical validation and trial and error.

## Dataset

Here, 2 MRI volumes of the same subject acquired as a part of Female dataset of Visible Human Project are used. The volumes are acquired using 2 different MRI modalities: 1) T2-Weighted MRI and 2) T1-Weighted MRI.  Here, the task of registering T2-Weighted MRI volume **(Moving Image)**
 to a T1-Weighted MRI volume **(Fixed Image)** is conducted.  

![git1.png](Image%20Regi%204d8c5/git1.png)

## Getting Started

### Installation

`pip install -r requirements.txt`

### Script for Image Registration

The parameters for the registration framework are finetuned and set as default values. 

- Transformation : itk.TranslationTransform
- Metric : itk.MattesMutualInformationImageToImageMetricv4
- Optimizer : itk.RegularStepGradientDescentOptimizerv4
- Registration : itk.ImageRegistrationMethodv4

Run the following script for image registration:

python [registraion.py](http://registraion.py/) --fixed_img_path "Path to Fixed Image in .vtk" --moving_img_path "Path to Moving Image in .vtk" --output_path "Path to Output"

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