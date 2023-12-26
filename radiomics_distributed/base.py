import copy
import inspect
#import logging
import os
import traceback
import dask.distributed as dd
import numpy
import SimpleITK as sitk
import six
import numpy as np
from tqdm import tqdm
import ray
from . import getProgressReporter, imageoperations

def featureNameContained(targetFeatureName, featureFullNames = None, mode='exact'):
  # always return true if the feature full names are not specified
  if featureFullNames is None:
    featureFullNames = []
  if len(featureFullNames) == 0:
    return True
  if mode == 'exact':
    mask = [targetFeatureName == featureFullName for featureFullName in featureFullNames]
  elif mode == 'partial':
    mask = [targetFeatureName in featureFullName for featureFullName in featureFullNames]
  else:
    return False
  return bool(np.any(mask, axis=None))

class RadiomicsFeatureCalculationController:
  def __init__(self, featureClasses, enabledFeatures, **kwargs):
    # global #logger
    self.binningParameterName = 'binCount'
    self.binningParameterValues = None
    if 'binCount' in kwargs:
      self.binningParameterValues = kwargs.get('binCount')
    if 'binWidth' in kwargs:
      self.binningParameterValues = kwargs.get('binWidth')
      self.binningParameterName = 'binWidth'

    self.label = kwargs.get('label', 1)
    self.voxelBased = kwargs.get('voxelBased', False)
    self.patchNumber = kwargs.get('patchNumber', 1)
    self.images = dict()
    self.masks = dict()
    self.global_settings = kwargs
    self.settings = dict()
    self.featureClasses = featureClasses
    self.enabledFeatures = enabledFeatures
    self.featureMapValues = {}
    self.featureMapMasks = {}


  def addImageType(self, inputImage, inputMask, imageTypeName, **settings):
    self.images[imageTypeName] = inputImage
    self.masks[imageTypeName] = inputMask
    settings_temp = settings.copy()
    if 'binCount' in settings_temp:
      settings_temp.pop('binCount')
    if 'binWidth' in settings_temp:
      settings_temp.pop('binWidth')
    settings_temp['pixelSpacing'] = inputImage.GetSpacing()
    self.settings[imageTypeName] = settings_temp

  def getValidFeatureNames(self, featureClassName, featureNames):
    validFeatureNames = []
    for featureName, deprecated in self.featureClasses[featureClassName].getFeatureNames().items():
      if featureNames is None or len(featureNames) == 0:
        if not deprecated:
          validFeatureNames.append(featureName)
      elif featureName in featureNames:
        validFeatureNames.append(featureName)
    return validFeatureNames

  def calculateShapeFeatures(self, mask, shape_type, featureFullNames=None):
    featureVector = {}
    # no shape feature contained in the feature full names
    if not featureNameContained(shape_type, featureFullNames, mode='partial'):
      return featureVector
    #logger.info('Computing %s', shape_type)
    featureNames = self.enabledFeatures[shape_type]
    # Pad inputMask to prevent index-out-of-range errors
    #logger.debug('Padding the mask with 0s')
    cpif = sitk.ConstantPadImageFilter()
    padding = numpy.tile(1, 3)
    try:
      cpif.SetPadLowerBound(padding)
      cpif.SetPadUpperBound(padding)
    except TypeError:
      # newer versions of SITK/python want a tuple or list
      cpif.SetPadLowerBound(padding.tolist())
      cpif.SetPadUpperBound(padding.tolist())
    croppedMaskPadded = cpif.Execute(mask)
    imageArray = None
    maskArray = sitk.GetArrayFromImage(croppedMaskPadded)
    pixelSpacing = mask.GetSpacing()
    settings = self.global_settings.copy()
    settings['pixelSpacing'] = pixelSpacing
    validFeatureNames = self.getValidFeatureNames(shape_type, featureNames)
    enabledFeatureFullNames = dict()
    for featureName in validFeatureNames:
      newFeatureName = 'original_%s_%s' % (shape_type, featureName)
      # no feature name matched in the feature full names
      if not featureNameContained(newFeatureName, featureFullNames, mode='exact'):
        continue
      enabledFeatureFullNames[featureName] = newFeatureName
    featureVector = self.calculateSegments(shape_type, imageArray, maskArray, enabledFeatureFullNames, **settings)

    # featureVector = consolidateFeatureResults(featureVector,has_client=False)
    return featureVector

  def calculateFeatures(self, featureFullNames=None, parallel=False, exportDirectory=None):
    # Calculate feature classes
    featureVector = {}
    for imageTypeName, image in self.images.items():
      # no image type matched in the feature full names
      if not featureNameContained(imageTypeName, featureFullNames, mode='partial'):
        continue
      mask = self.masks[imageTypeName]
      settings = self.settings[imageTypeName]
      settings['pixelSpacing'] = image.GetSpacing()
      imageArray = sitk.GetArrayFromImage(image)
      maskArray = sitk.GetArrayFromImage(mask) == self.label
      for featureClassName, featureNames in six.iteritems(self.enabledFeatures):
        # Handle calculation of shape features separately
        if featureClassName.startswith('shape'):
          continue
        if featureClassName not in self.featureClasses:
          continue
          # no feature class matched in the feature full names
        if not featureNameContained(featureClassName, featureFullNames, mode='partial'):
          continue
        for binningParameterValue in self.binningParameterValues:
          if self.binningParameterName == 'binWidth':
            binningFeatureName = '%.2f_%s' % (binningParameterValue, self.binningParameterName)
          else:
            binningFeatureName = '%d_%s' % (binningParameterValue, self.binningParameterName)
          # no binning parameter matched in the feature full names
          if not featureNameContained(binningFeatureName, featureFullNames, mode='partial'):
            continue
          validFeatureNames = self.getValidFeatureNames(featureClassName, featureNames)
          enabledFeatureFullNames = dict()
          for featureName in validFeatureNames:
            newFeatureName = '%s_%s_%s_' % (imageTypeName, featureClassName, featureName) + binningFeatureName
            # no feature name matched in the feature full names
            if not featureNameContained(newFeatureName, featureFullNames, mode='exact'):
              continue
            enabledFeatureFullNames[featureName] = newFeatureName
          if len(enabledFeatureFullNames) == 0:
            continue
          if self.voxelBased:
            print('Calculating feature maps for feature class {0} with enabled feature full names {1}.'.format(featureClassName, enabledFeatureFullNames))
            self.featureMapValues = {}
            self.featureMapMasks = {}
            for featureName, fullFeatureName in enabledFeatureFullNames.items():
              if featureName not in self.featureMapMasks:
                self.featureMapMasks[featureName] = numpy.full(np.flip(image.GetSize()), False, numpy.bool)
              if featureName not in self.featureMapValues:
                initial_feature_map = numpy.full(np.flip(image.GetSize()), numpy.nan)
                self.featureMapValues[featureName] = initial_feature_map
            self.calculateVoxels(featureClassName, imageArray, maskArray, enabledFeatureFullNames,
                                 self.patchNumber, parallel=parallel, **settings,
                                 **{self.binningParameterName:binningParameterValue})
            featureMaps = self.combineFeatureMaps(enabledFeatureFullNames, image, exportDirectory)
            # featureMaps = combineFeatureMaps(partitionedFeatureMapsFutures, image, enabledFeatureFullNames=enabledFeatureFullNames,
            #                                  exportDirectory=exportDirectory, parallel= client is not None and featureClassName != 'firstorder')
            featureVector.update(featureMaps)
          else:
            featureVector.update(self.calculateSegments(featureClassName, imageArray, maskArray, enabledFeatureFullNames,
                                             **settings,**{self.binningParameterName:binningParameterValue}))
    return featureVector

  def insertFeatureMap(self, featureMaps, coordinateOffset):
    if len(featureMaps) == 0:
      print('Empty partitioned feature maps.')
    # starttime = timeit.default_timer()
    for featureName, featureMap in featureMaps.items():
      labelledVoxelCoordinates = featureMap[0]
      globalLabelledVoxelCoordinates = labelledVoxelCoordinates + np.repeat([coordinateOffset],
                                                                            labelledVoxelCoordinates.shape[1],
                                                                            axis=0).transpose()
      globalLabelledVoxelCoordinates = tuple(globalLabelledVoxelCoordinates)
      self.featureMapValues[featureName][globalLabelledVoxelCoordinates] = featureMap[1]
      self.featureMapMasks[featureName][globalLabelledVoxelCoordinates] = True


  def combineFeatureMaps(self, enabledFeatureFullNames, image, exportDirectory=None):
    finalFeatureMaps = {}
    for featureName, featureMap in self.featureMapValues.items():
      if enabledFeatureFullNames is not None and featureName in enabledFeatureFullNames:
        fullFeatureName = enabledFeatureFullNames[featureName]
      else:
        fullFeatureName = featureName
      minValue = float(np.nanmin(featureMap, axis=None))
      maxValue = float(np.nanmax(featureMap, axis=None))
      featureMap = np.nan_to_num(featureMap, nan=minValue - (maxValue - minValue) / 1000)
      featureMap = sitk.GetImageFromArray(featureMap)
      featureMap.CopyInformation(image)
      if exportDirectory is not None:
        sitk.WriteImage(featureMap, os.path.join(exportDirectory, fullFeatureName + '.mha'))
        print('Feature map {0} exported to {1}.'.format(fullFeatureName, exportDirectory))
        featureMapMask = sitk.GetImageFromArray(self.featureMapMasks[featureName].astype(float))
        featureMapMask.CopyInformation(image)
        sitk.WriteImage(featureMapMask, os.path.join(exportDirectory, fullFeatureName + '_mask.mha'))
      finalFeatureMaps[fullFeatureName] = featureMap
    return finalFeatureMaps

  def calculateSegments(self, featureClassName, imageArray, maskArray, enabledFeatureFullNames, **settings):
    if len(enabledFeatureFullNames) == 0:
      return {}
    featureClass = self.featureClasses[featureClassName](imageArray, maskArray, **settings)
    for featureName in enabledFeatureFullNames.keys():
      featureClass.enableFeatureByName(featureName)
    results = featureClass._calculateFeatures()
    featureValues = dict()
    for featureName, featureValue in results.items():
      featureValues[enabledFeatureFullNames[featureName]] = featureValue
    return featureValues


  def calculateVoxels(self, featureClassName, imageArray, maskArray, enabledFeatureFullNames, patch_number, parallel=False,
                      **settings):
    if len(enabledFeatureFullNames) == 0:
      return {}
    kernel_radius = settings.get('kernelRadius', 1)
    print('Kernel radius: {0}'.format(kernel_radius))
    patch_numbers = []
    starting_indexes = []
    ending_indexes = []
    for dim in range(imageArray.ndim):
      patch_size = int(np.ceil((imageArray.shape[dim] - (2 * kernel_radius)) / patch_number))
      actual_patch_number = int((imageArray.shape[dim] - (2 * kernel_radius)) / patch_size) + 1
      starting_indexes_1d = np.arange(actual_patch_number) * patch_size + kernel_radius
      ending_indexes_1d = starting_indexes_1d + patch_size
      ending_indexes_1d[-1] = imageArray.shape[dim] - kernel_radius
      patch_numbers.append(actual_patch_number)
      starting_indexes.append(starting_indexes_1d)
      ending_indexes.append(ending_indexes_1d)
    starting_indexes = np.meshgrid(*starting_indexes, indexing='ij')
    starting_indexes = [item.flatten() for item in starting_indexes]
    starting_indexes = list(zip(*starting_indexes))
    ending_indexes = np.meshgrid(*ending_indexes, indexing='ij')
    ending_indexes = [item.flatten() for item in ending_indexes]
    ending_indexes = list(zip(*ending_indexes))
    # sitk.WriteImage(self.inputImage, '/media/radiomics/My Book/RadiomicsResearchProjects/RadVI/feature_maps/Subject_01/feature_maps/original_image.mha')

    print('{0} partition planned.'.format(len(starting_indexes)))

    globalFeatureClass = self.featureClasses[featureClassName](imageArray, maskArray, **settings)
    for featureName in enabledFeatureFullNames.keys():
      globalFeatureClass.enableFeatureByName(featureName)
    if featureClassName == 'firstorder':
      return [(globalFeatureClass._calculateVoxels(), np.zeros(imageArray.ndim))]
    if not parallel:
      with tqdm(total=len(starting_indexes)) as pbar:
        for starting_index_all_dim, ending_index_all_dim in zip(starting_indexes, ending_indexes):
          featureClassPartitioned = globalFeatureClass.imageCropping(np.array(starting_index_all_dim),
                                                                     np.array(ending_index_all_dim), kernel_radius)
          voxel_count = featureClassPartitioned.labelledVoxelCoordinates.shape[1]
          if voxel_count == 0:
            pbar.update(1)
            continue
          # print('The partitioned feature class has voxel counts of {0}.'.format(voxel_count))
          coordinateOffset = np.array(starting_index_all_dim) - kernel_radius
          self.insertFeatureMap(featureClassPartitioned._calculateVoxels(), coordinateOffset)
          pbar.update(1)
    else:
      futures = dict()
      globalFeatureClass_shared = ray.put(globalFeatureClass)
      for starting_index_all_dim, ending_index_all_dim in zip(starting_indexes,
                                                              ending_indexes):
        coordinateOffset = np.array(starting_index_all_dim) - kernel_radius
        future = calculateVoxels.remote(globalFeatureClass_shared, starting_index_all_dim, ending_index_all_dim,
                                        kernel_radius)
        futures[future] = coordinateOffset
      # print('{0} partitioned feature class under calculation.'.format(len(futures)), flush=True)

      unfinished_futures = list(futures.keys())
      failed_futures = dict()
      with tqdm(total=len(futures)) as pbar:
        while len(unfinished_futures):
          finished_futures, unfinished_futures = ray.wait(unfinished_futures)
          coordinateOffset = futures[finished_futures[0]]
          try:
            result = ray.get(finished_futures[0], timeout=1)
            self.insertFeatureMap(result, coordinateOffset)
            pbar.update(1)
          except Exception as e:
            # print('Partitioned feature map calculation failed: {0}.'.format(e))  # TODO: remove this line after debugging
            failed_futures[finished_futures[0]] = coordinateOffset

        # TODO: cause errors sometimes, need to be fixed
        #for featureClassPartitioned, coordinateOffset in failed_futures:
        #  self.insertFeatureMap(featureClassPartitioned._calculateVoxels(), coordinateOffset)
        #  pbar.update(1)

@ray.remote
def calculateVoxels(globalFeatureClass, starting_index_all_dim, ending_index_all_dim, kernel_radius):
  featureClassPartitioned = globalFeatureClass.imageCropping(np.array(starting_index_all_dim),
                                                             np.array(ending_index_all_dim), kernel_radius)
  voxel_count = featureClassPartitioned.labelledVoxelCoordinates.shape[1]
  if voxel_count == 0:
    return
  result = featureClassPartitioned._calculateVoxels()
  return result



class RadiomicsFeaturesBase(object):
  """
  This is the abstract class, which defines the common interface for the feature classes. All feature classes inherit
  (directly of indirectly) from this class.

  At initialization, image and labelmap are passed as SimpleITK image objects (``inputImage`` and ``inputMask``,
  respectively.) The motivation for using SimpleITK images as input is to keep the possibility of reusing the
  optimized feature calculators implemented in SimpleITK in the future. If either the image or the mask is None,
  initialization fails and a warning is #logged (does not raise an error).

  Logging is set up using a child #logger from the parent 'radiomics' #logger. This retains the toolbox structure in
  the generated #log. The child #logger is named after the module containing the feature class (e.g. 'radiomics.glcm').

  Any pre calculations needed before the feature functions are called can be added by overriding the
  ``_initSegmentBasedCalculation`` function, which prepares the input for feature extraction. If image discretization is
  needed, this can be implemented by adding a call to ``_applyBinning`` to this initialization function, which also
  instantiates coefficients holding the maximum ('Ng') and unique ('GrayLevels') that can be found inside the ROI after
  binning. This function also instantiates the `matrix` variable, which holds the discretized image (the `imageArray`
  variable will hold only original gray levels).

  The following variables are instantiated at initialization:

  - kwargs: dictionary holding all customized settings passed to this feature class.
  - label: label value of Region of Interest (ROI) in labelmap. If key is not present, a default value of 1 is used.
  - featureNames: list containing the names of features defined in the feature class. See :py:func:`getFeatureNames`
  - inputImage: SimpleITK image object of the input image (dimensions x, y, z)

  The following variables are instantiated by the ``_initSegmentBasedCalculation`` function:

  - inputMask: SimpleITK image object of the input labelmap (dimensions x, y, z)
  - imageArray: numpy array of the gray values in the input image (dimensions z, y, x)
  - maskArray: numpy boolean array with elements set to ``True`` where labelmap = label, ``False`` otherwise,
    (dimensions z, y, x).
  - labelledVoxelCoordinates: tuple of 3 numpy arrays containing the z, x and y coordinates of the voxels included in
    the ROI, respectively. Length of each array is equal to total number of voxels inside ROI.
  - matrix: copy of the imageArray variable, with gray values inside ROI discretized using the specified binWidth.
    This variable is only instantiated if a call to ``_applyBinning`` is added to an override of
    ``_initSegmentBasedCalculation`` in the feature class.

  .. note::
    Although some variables listed here have similar names to customization settings, they do *not* represent all the
    possible settings on the feature class level. These variables are listed here to help developers develop new feature
    classes, which make use of these variables. For more information on customization, see
    :ref:`radiomics-customization-label`, which includes a comprehensive list of all possible settings, including
    default values and explanation of usage.
  """

  def __init__(self, imageArray, maskArray, **kwargs):
    #self.logger = #logging.getLogger(self.__module__)
    #self.logger.debug('Initializing feature class')

    # self.progressReporter = getProgressReporter

    self.settings = kwargs

    self.voxelBased = kwargs.get('voxelBased', False)

    self.coefficients = {'pixelSpacing':kwargs.get('pixelSpacing')}

    # all features are disabled by default
    self.enabledFeatures = {}
    self.featureValues = {}

    self.featureNames = self.getFeatureNames()

    self.maskArray = maskArray
    self.imageArray = imageArray
    self.voxelBased = kwargs.get('voxelBased')

    if self.voxelBased:
      self._initVoxelBasedCalculation()
    # else:
    #   self._initSegmentBasedCalculation()

  # def _initSegmentBasedCalculation(self):
  #   self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)  # boolean array

  def imageCropping(self, starting_indexes, ending_indexes, margin):
    radiomicsFeatureBaseCopy = copy.copy(self)
    image_chunk = self.imageArray
    mask_chunk_with_margin = self.maskArray
    labeledCoordinatesMask = []
    for dim, starting_index, ending_index, in zip(range(len(starting_indexes)), starting_indexes,
                                                  ending_indexes):
      image_chunk = np.take(image_chunk, np.arange(starting_index - margin, ending_index + margin),
                            axis=dim, mode='clip')
      mask_chunk_with_margin = np.take(mask_chunk_with_margin,
                                       np.arange(starting_index - margin, ending_index + margin),
                                       axis=dim, mode='clip')
      labeledCoordinatesMask.append(np.all([self.labelledVoxelCoordinates[dim,:]>=starting_index,self.labelledVoxelCoordinates[dim,:]<ending_index],axis=0))
    radiomicsFeatureBaseCopy.imageArray = image_chunk
    radiomicsFeatureBaseCopy.maskArray = mask_chunk_with_margin
    labeledCoordinatesMask = np.all(labeledCoordinatesMask,axis=0)
    croppedLabeldCoordinates = self.labelledVoxelCoordinates[:,labeledCoordinatesMask]
    radiomicsFeatureBaseCopy.labelledVoxelCoordinates = croppedLabeldCoordinates-np.repeat([starting_indexes-margin],
                                                                                           croppedLabeldCoordinates.shape[1],
                                                                                           axis=0).transpose()
    return radiomicsFeatureBaseCopy


  def _initVoxelBasedCalculation(self):
    self.masked = self.settings.get('maskedKernel', True)
    # maskArray = sitk.GetArrayFromImage(self.inputMask) == self.label  # boolean array
    self.labelledVoxelCoordinates = numpy.array(numpy.where(self.maskArray))
    # Set up the mask array for the gray value discretization
    if not self.masked:
      # This will cause the discretization to use the entire image
      self.maskArray = numpy.ones(self.imageArray.shape, dtype='bool')

  def _calculateCoefficients(self, voxelCoordinates=None):
    """
    Last steps to prepare the class for extraction. This function calculates the texture matrices and coefficients in
    the respective feature classes
    """
    pass

  def _applyBinning(self):
    discretizedImageArray, _ = imageoperations.binImage(self.imageArray, self.maskArray, **self.settings)
    self.coefficients['grayLevels'] = numpy.unique(discretizedImageArray[self.maskArray])
    self.coefficients['Ng'] = int(numpy.max(self.coefficients['grayLevels']))  # max gray level in the ROI
    return discretizedImageArray

  def enableFeatureByName(self, featureName, enable=True):
    """
    Enables or disables feature specified by ``featureName``. If feature is not present in this class, a lookup error is
    raised. ``enable`` specifies whether to enable or disable the feature.
    """
    if featureName not in self.featureNames:
      raise LookupError('Feature not found: ' + featureName)
    #if self.featureNames[featureName]:
      #self.logger.warning('Feature %s is deprecated, use with caution!', featureName)
    self.enabledFeatures[featureName] = enable

  def enableAllFeatures(self):
    """
    Enables all features found in this class for calculation.

    .. note::
      Features that have been marked "deprecated" are not enabled by this function. They can still be enabled manually by
      a call to :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`,
      :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()`
      or in the parameter file (by specifying the feature by name, not when enabling all features).
      However, in most cases this will still result only in a deprecation warning.
    """
    for featureName, is_deprecated in six.iteritems(self.featureNames):
      # only enable non-deprecated features here
      if not is_deprecated:
        self.enableFeatureByName(featureName, True)

  def disableAllFeatures(self):
    """
    Disables all features. Additionally resets any calculated features.
    """
    self.enabledFeatures = {}
    self.featureValues = {}

  @classmethod
  def getFeatureNames(cls):
    """
    Dynamically enumerates features defined in the feature class. Features are identified by the
    ``get<Feature>FeatureValue`` signature, where <Feature> is the name of the feature (unique on the class level).

    Found features are returned as a dictionary of the feature names, where the value ``True`` if the
    feature is deprecated, ``False`` otherwise (``{<Feature1>:<deprecated>, <Feature2>:<deprecated>, ...}``).

    This function is called at initialization, found features are stored in the ``featureNames`` variable.
    """
    attributes = inspect.getmembers(cls)
    features = {a[0][3:-12]: getattr(a[1], '_is_deprecated', False) for a in attributes
                if a[0].startswith('get') and a[0].endswith('FeatureValue')}
    return features

  def execute(self, client=None):
    """
    Calculates all features enabled in  ``enabledFeatures``. A feature is enabled if it's key is present in this
    dictionary and it's value is True.

    Calculated values are stored in the ``featureValues`` dictionary, with feature name as key and the calculated
    feature value as value. If an exception is thrown during calculation, the error is #logged, and the value is set to
    NaN.
    """
    if len(self.enabledFeatures) == 0:
      self.enableAllFeatures()

    if self.voxelBased:
      self._calculateVoxels()
    else:
      self._calculateSegment(client)

    return self.featureValues


  def _calculateVoxels(self):
    voxelBatch = self.settings.get('voxelBatch', -1)
    featureMaps = dict()
    # initial_feature_map = numpy.full(featureClassCopy.imageArray.shape, initValue, dtype='float')
    # for feature, enabled in six.iteritems(featureClassCopy.enabledFeatures):
    #   if enabled:
    #     # self.featureValues[feature] = []
    #     featureValues[feature] = initial_feature_map
    voxel_count = self.labelledVoxelCoordinates.shape[1]
    if voxelBatch < 0:
      voxelBatch = voxel_count
    # n_batches = numpy.ceil(float(voxel_count) / voxelBatch)
    voxel_batch_idx = 0
    while voxel_batch_idx < voxel_count:
      ending_index = voxel_batch_idx + voxelBatch
      if ending_index >=voxel_count:
        ending_index = voxel_count
      voxelCoords = self.labelledVoxelCoordinates[:, voxel_batch_idx:ending_index]
      for featureName, featureValues in self._calculateFeatures(voxelCoordinates=voxelCoords).items():
        if featureName in featureMaps:
          existingVoxelCoords, existingFeatureValue = featureMaps[featureName]
          featureMaps[featureName] = (np.concatenate([existingVoxelCoords, voxelCoords], axis=1),
                                      np.concatenate([existingFeatureValue, featureValues], axis=None))
        else:
          featureMaps[featureName] = (voxelCoords,featureValues)
      voxel_batch_idx += voxelBatch
    # del featureClassCopy
    return featureMaps



    # # Initialize the output with empty numpy arrays
    # initial_feature_map = numpy.full(list(self.coefficients['pixelSpacing'])[::-1], -1000, dtype='float')
    # for feature, enabled in six.iteritems(self.enabledFeatures):
    #   if enabled:
    #     # self.featureValues[feature] = []
    #     self.featureValues[feature] = initial_feature_map
    # min_feature_values = dict()
    # if hasattr(self, 'discretizedImageArray'):
    #   labelledVoxelCoordinates = numpy.where(self.maskArray)
    #   self._calculateFeatures(labelledVoxelCoordinates)
    # else:
    #   background_mask, min_feature_values = self.patch_based_voxel_wise_featue_calculation(patch_number= 5)
    #   # Convert the output to simple ITK image objects
    #   background_mask = background_mask.astype(bool)
    #   self.featureValues['mask'] = sitk.GetImageFromArray((~background_mask).astype(int))
    #   self.featureValues['mask'].CopyInformation(self.inputImage)
    #   self.featureValues['mask'] = sitk.Cast(self.featureValues['mask'], sitk.sitkUInt8)
    #   self.featureValues['image'] = self.inputImage
    #
    # for feature, enabled in six.iteritems(self.enabledFeatures):
    #   if enabled:
    #     if feature in min_feature_values:
    #       self.featureValues[feature][background_mask] = min_feature_values[feature]-1
    #     self.featureValues[feature] = sitk.GetImageFromArray(self.featureValues[feature])
    #     self.featureValues[feature].CopyInformation(self.inputImage)

  @staticmethod
  def staticInitializeCalculation(featureClass, voxelCoords):
    featureClassCopy = featureClass.copy()
    # del featureClassCopy.inputImage
    # del featureClassCopy.inputMask
    # del featureClassCopy.labelledVoxelCoordinates
    # featureClassCopy.inputImage = None
    # featureClassCopy.inputMask = None
    # featureClassCopy.labelledVoxelCoordinates = None
    # featureClassCopy.enabledFeatures = featureClass.enabledFeatures
    # featureClassCopy.featureNames = featureClass.featureNames
    featureClassCopy._initCalculation(voxelCoords)
    del featureClassCopy.imageArray
    del featureClassCopy.maskArray
    featureClassCopy.imageArray = None
    featureClassCopy.maskArray = None
    return featureClassCopy


  def _calculateSegment(self, client=None):
    # Get the feature values using the current segment.
    for success, featureName, featureValue in self._calculateFeatures(client):
      # Always store the result. In case of an error, featureValue will be NaN
      if isinstance(featureValue, numpy.ndarray):
        featureValue = numpy.squeeze(featureValue)
        if len(featureValue) == 1:
          featureValue = featureValue[0]
      self.featureValues[featureName] = featureValue


  @staticmethod
  def singleFeatureCalculation(featureClass, featureName):
    result = (False, featureName, numpy.nan)
    try:
      # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
      result = (True, featureName, getattr(featureClass, 'get%sFeatureValue' % featureName)())
    except DeprecationWarning as deprecatedFeature:
      pass
      # Add a debug #log message, as a warning is usually shown and would entail a too verbose output
      #featureClass.logger.debug('Feature %s is deprecated: %s', featureName, deprecatedFeature.args[0])
    except Exception:
      print('FAILED: %s', traceback.format_exc())
      #featureClass.logger.error('FAILED: %s', traceback.format_exc())
    return result

  @staticmethod
  def staticCalculateFeatures(featureClass, voxelCoordinates=None):
    featureClass._initCalculation(voxelCoordinates)
    # featureClass.#logger.debug('Calculating features')
    for feature, enabled in six.iteritems(featureClass.enabledFeatures):
      if enabled:
        try:
          # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
          yield True, feature, getattr(featureClass, 'get%sFeatureValue' % feature)()
        except DeprecationWarning as deprecatedFeature:
          pass
          # Add a debug #log message, as a warning is usually shown and would entail a too verbose output
          #featureClass.logger.debug('Feature %s is deprecated: %s', feature, deprecatedFeature.args[0])
        except Exception:
          #featureClass.logger.error('FAILED: %s', traceback.format_exc())
          yield False, feature, numpy.nan

  def _calculateFeatures(self,voxelCoordinates=None):
    # Initialize the calculation
    # This function serves to calculate the texture matrices where applicable
    #self.logger.debug('Calculating features')
    results = dict()
    coefficients = self._calculateCoefficients(voxelCoordinates)
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
        try:
          featureValue = getattr(self, 'get%sFeatureValue' % feature)(coefficients)
          featureValue = numpy.squeeze(featureValue)
          if featureValue.size == 1:
            featureValue = float(featureValue)
        except DeprecationWarning as deprecatedFeature:
          # Add a debug #log message, as a warning is usually shown and would entail a too verbose output
          #self.logger.debug('Feature %s is deprecated: %s', feature, deprecatedFeature.args[0])
          continue
        except Exception:
          #self.logger.error('FAILED: %s', traceback.format_exc())
          continue
        results[feature] = featureValue
    return results

    # for feature, enabled in six.iteritems(self.enabledFeatures):
    #   if enabled:
    #     try:
    #       # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
    #       yield True, feature, getattr(self, 'get%sFeatureValue' % feature)()
    #     except DeprecationWarning as deprecatedFeature:
    #       # Add a debug #log message, as a warning is usually shown and would entail a too verbose output
    #       self.#logger.debug('Feature %s is deprecated: %s', feature, deprecatedFeature.args[0])
    #     except Exception:
    #       self.#logger.error('FAILED: %s', traceback.format_exc())
    #       yield False, feature, numpy.nan
