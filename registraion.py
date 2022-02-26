import sys
from distutils.version import StrictVersion as VS
import itk
import numpy as np
import itk
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from seaborn.matrix import clustermap
import warnings
warnings.filterwarnings("ignore")
import argparse
import os


def parse_option():
    parser = argparse.ArgumentParser('Arguements for Image Registration')


    parser.add_argument('--fixed_img_path', type=str, help='Path fixed image')
    parser.add_argument('--moving_img_path', type=str,help='Path moving image')
    parser.add_argument('--output_path', type=str, help='Path to save the registered image')

    
    #Optimization parameters
    parser.add_argument('--lr', type=float, default = 4 ,  help='Learning rate optimizer')
    parser.add_argument('--Min_step_length', type=float, default = 0.0005 ,  help='Minimum Step Length')
    parser.add_argument('--relax_factor', type=float, default = 0.5 ,  help='Relaxation Factor')
    parser.add_argument('--num_iter', type=int, default = 200 ,  help='Number of Iterations')
    
    #Metric parameters
    parser.add_argument('--num_hist_bins', type=int, default = 24 ,  help='Number of Histogram Bins')

    #Registration parameters
    parser.add_argument('--num_levels', type=int, default = 1 ,  help='Number of Levels')
    parser.add_argument('--smooth_sigma_level', type=float, default = 0 ,  help='Smoothing Sigmas Per Level')	
    parser.add_argument('--shrink_level', type=int, default = 1 ,  help='Shrink Factors Per Level')
          
    opt = parser.parse_args()
    
    return opt


def main():
    args = parse_option()
    msg = 'fixed_img_path: {0}\n'\
          'moving_img_path: {1}\n'\
          'output_path: {2}\n'\
          'lr: {3}\n'\
          'Min_step_length: {4}\n'\
          'relax_factor: {5}\n'\
          'num_iter: {6}\n'\
          'num_hist_bins: {7}\n'\
          'num_levels: {8}\n'\
          'smooth_sigma_level: {9}\n'\
          'shrink_level: {10}\n'.format(
                          args.fixed_img_path,args.moving_img_path,args.output_path, args.lr,
                          args.Min_step_length,args.relax_factor,args.num_iter,
                          args.num_hist_bins,args.num_levels,args.smooth_sigma_level,args.shrink_level)
    print(msg)


    #Images and Output files
    pixel_type = itk.F
    fixed_image_file = args.fixed_img_path
    moving_image_file = args.moving_img_path
    reg_image_file_trans  = os.path.join(args.output_path, 'reg_trans.vtk')
    diff_img_before_file_trans = os.path.join(args.output_path , 'img_before_trans.vtk')
    diff_img_after_file_trans  = os.path.join(args.output_path ,'img_before_trans.vtk')

    fixed_img = itk.imread(fixed_image_file, pixel_type)
    moving_img = itk.imread(moving_image_file, pixel_type)

    dim = fixed_img.GetImageDimension()
    fixed_img_type = itk.Image[pixel_type, dim]
    moving_img_type = itk.Image[pixel_type, dim]

    print("Fixed Image ====================>")
    print(f"Shape       : {fixed_img.shape} ")
    print(f"Pixel Type  : {fixed_img_type} ")
    print("Moving Image ===================>")
    print(f"Shape       : {moving_img.shape} ")
    print(f"Pixel Type  : {moving_img_type} ")

    ########################### Transformation ###########################
    transform_type = itk.TranslationTransform[itk.D,dim]
    initial_transform = transform_type.New()

    ############################# Optimizer ##############################
    opt = itk.RegularStepGradientDescentOptimizerv4.New( LearningRate=args.lr,        ###
                                                        MinimumStepLength=args.Min_step_length,###
                                                        RelaxationFactor=args.relax_factor,   ###
                                                        NumberOfIterations=args.num_iter) ###

    ############################## Metric ##################################
    metric = itk.MattesMutualInformationImageToImageMetricv4[fixed_img_type, moving_img_type].New()
    metric.SetNumberOfHistogramBins(args.num_hist_bins)  ####
    metric.SetUseMovingImageGradientFilter(False)
    metric.SetUseFixedImageGradientFilter(False)

    ########################### Registration ###############################
    reg = itk.ImageRegistrationMethodv4[fixed_img_type, moving_img_type].New( FixedImage = fixed_img, 
                                                                            MovingImage = moving_img,
                                                                            Metric = metric, Optimizer=opt, 
                                                                            InitialTransform=initial_transform)

    movingInitialTransform = transform_type.New()
    initialParameters = movingInitialTransform.GetParameters()
    initialParameters[0] = 0
    initialParameters[1] = 0
    initialParameters[2] = 0
    movingInitialTransform.SetParameters(initialParameters)
    reg.SetMovingInitialTransform(movingInitialTransform)



    identityTransform = transform_type.New()
    identityTransform.SetIdentity()
    reg.SetFixedInitialTransform(identityTransform)

    #------------------------------------
    reg.SetNumberOfLevels(args.num_levels)                  ###
    reg.SetSmoothingSigmasPerLevel([args.smooth_sigma_level])       ###
    reg.SetShrinkFactorsPerLevel([args.shrink_level])         ###
    #------------------------------------

    iter = []
    val = []
    def iterationUpdate():
        currentParameter = reg.GetTransform().GetParameters()
        iter.append(opt.GetCurrentIteration())
        val.append(opt.GetValue())
        print(
            "Index : %i -->  Metric : %f   Translation (X,Y,Z) : (%f %f %f)"
            % (
                opt.GetCurrentIteration(),
                opt.GetValue(),
                currentParameter.GetElement(0),
                currentParameter.GetElement(1),
                currentParameter.GetElement(2),

            )
        )
        
    iterationCommand = itk.PyCommand.New()
    iterationCommand.SetCommandCallable(iterationUpdate)
    opt.AddObserver(itk.IterationEvent(), iterationCommand)

    print(" Starting Registration ===========================>")
    reg.Update()
    print(" Registration Completed ===========================>")




    transform = reg.GetTransform()
    finalParameters = transform.GetParameters()
    translationAlongX = finalParameters.GetElement(0)
    translationAlongY = finalParameters.GetElement(1)
    translationAlongZ = finalParameters.GetElement(2)
    numberOfIterations = opt.GetCurrentIteration()

    bestValue = opt.GetValue()

    print("Result:")
    print(" Translation X = " + str(translationAlongX))
    print(" Translation Y = " + str(translationAlongY))
    print(" Translation Z = " + str(translationAlongZ))
    print(" Iterations    = " + str(numberOfIterations))
    print(" Metric value  = " + str(bestValue))
    print("=============================================>")


    plt.figure()
    plt.plot(iter, val)
    plt.xlabel("Iterations")
    plt.ylabel("Metric (Mattes Mutual Information)")
    plt.title("Metric vs Iterations")
    plt.show()

    CompositeTransformType = itk.CompositeTransform[itk.D, dim]
    outputCompositeTransform = CompositeTransformType.New()
    outputCompositeTransform.AddTransform(movingInitialTransform)
    outputCompositeTransform.AddTransform(reg.GetModifiableTransform())

    resampler = itk.ResampleImageFilter.New(Input = moving_img,
                                            Transform = outputCompositeTransform,
                                            UseReferenceImage = True,
                                            ReferenceImage = fixed_img)
    resampler.SetDefaultPixelValue(100)

    OutputPixelType = itk.F#ctype('unsigned char')
    OutputImageType = itk.Image[OutputPixelType, dim]

    caster = itk.CastImageFilter[fixed_img_type,
            OutputImageType].New(Input = resampler)

    writer = itk.ImageFileWriter.New(Input=caster, FileName=reg_image_file_trans)
    writer.SetFileName(reg_image_file_trans)
    writer.Update()

    difference = itk.SubtractImageFilter.New(Input1 = fixed_img,
                                            Input2 = resampler)

    intensityRescaler = itk.RescaleIntensityImageFilter[fixed_img_type,
                        OutputImageType].New(
                            Input=difference,
                            OutputMinimum=itk.NumericTraits[OutputPixelType].min(),
                            OutputMaximum=itk.NumericTraits[OutputPixelType].max())

    resampler.SetDefaultPixelValue(1)
    writer.SetInput(intensityRescaler.GetOutput())
    writer.SetFileName(diff_img_after_file_trans)
    writer.Update()

    resampler.SetTransform(identityTransform)
    writer.SetFileName(diff_img_before_file_trans)
    writer.Update()

if __name__ == '__main__':
    main()
