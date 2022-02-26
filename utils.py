import numpy as np
import itk
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from seaborn.matrix import clustermap

def plotting_images(img_list,title_list, cmap = 'gray'):
  fig, ax = plt.subplots(1,len(img_list), figsize=(20,8))
  for i in range(len(img_list)):
    ax[i].imshow(img_list[i], cmap = cmap, interpolation ='bilinear'), 
    ax[i].set_title(title_list[i],fontsize = 15),ax[i].axis('off')
  plt.show()


def interactive_plot(img_list,title_list, cmap = 'gray'):
  def plotting_images(slice):
    fig, ax = plt.subplots(1,len(img_list), figsize=(20,8))
    for i in range(len(img_list)):
      ax[i].imshow(img_list[i][slice], cmap = cmap, interpolation ='bilinear'), 
      ax[i].set_title(title_list[i],fontsize = 15),ax[i].axis('off')
    plt.show()
  select_slice = IntSlider(min=0, max=img_list[0].shape[0]-1, description='Select Slice', continuous_update=False)
  return interactive(plotting_images, slice=select_slice)