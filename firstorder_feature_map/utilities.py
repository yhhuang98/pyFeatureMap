import radiomics
import SimpleITK as sitk


""" Image discretization """
def fixed_count_discretization(image_img, mask_img, bin_count=64):
    feature_classes = radiomics.getFeatureClasses()
    feature_class = feature_classes['firstorder'](image_img, mask_img,
                                                  binCount=bin_count)
    image_arr = feature_class.discretizedImageArray

    return image_arr
    