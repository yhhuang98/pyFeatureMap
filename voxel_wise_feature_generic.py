import os, timeit
import SimpleITK as sitk
import numpy as np

from radiomics_distributed import featureextractor, utils as ut




if __name__ == '__main__':
    debug = False
    interp = True

    label = 1
    selectedFeatureFullNames = [
                                #'log-sigma-3-0-mm-3D_glrlm_GrayLevelNonUniformityNormalized_32_binCount',
                                #'log-sigma-3-0-mm-3D_glszm_GrayLevelNonUniformityNormalized_32_binCount',
                                #'log-sigma-3-0-mm-3D_glrlm_GrayLevelVariance_32_binCount',
                                #'log-sigma-3-0-mm-3D_glrlm_RunEntropy_32_binCount',
                                #'log-sigma-3-0-mm-3D_gldm_GrayLevelVariance_32_binCount',
                                'log-sigma-3-0-mm-3D_glcm_SumSquares_32_binCount',
                                # 'log-sigma-3-0-mm-3D_glcm_ClusterProminence_32_binCount',
                                # 'log-sigma-3-mm-3D_glcm_ClusterTendency_32_binCount',
                                # 'log-sigma-3-mm-3D_glcm_SumEntropy_32_binCount',
                                ]

    preprocessed_data_directory = r'../data'
    export_directory = r'../FM'
    patient_ids = os.listdir(preprocessed_data_directory)

    image_feature_extraction_parameters = ut.parse_from_yaml(r'./parameters/voxel_wise_feature_parameters.yaml')
    image_feature_extraction_parameters['setting']['label'] = label
    extractor = featureextractor.RadiomicsFeatureExtractor(image_feature_extraction_parameters)
    #ray.init(address='192.168.1.158:6379')

    failed_patients = []
    starttime = timeit.default_timer()
    for patient_id in patient_ids[:]:
        if debug:
            patient_id = patient_ids[0]

        patient_directory = os.path.join(preprocessed_data_directory, patient_id)
        if not os.path.isdir(patient_directory):
            continue

        patient_export_directory = os.path.join(export_directory, patient_id)
        if not os.path.exists(patient_export_directory):
            os.makedirs(patient_export_directory)

            image_filepath = os.path.join(patient_directory, 'early_contrast_DCE.mha')

            # create a temporary 1x1x1 image
            image_img = sitk.ReadImage(image_filepath)
            if interp:
                higest_resolution = np.min(image_img.GetSpacing())
                image_img = ut.resampleImg_to_resolution(image_img, [higest_resolution, higest_resolution, higest_resolution], sitk.sitkLinear)
                image_filepath = os.path.join(patient_directory, 'early_contrast_DCE_temp.mha')
                sitk.WriteImage(image_img, image_filepath)

            # fake mask: all 1
            fake_mask_arr = sitk.GetArrayFromImage(image_img)
            fake_mask_arr[fake_mask_arr >= 0] = 1
            fake_mask_img = sitk.GetImageFromArray(fake_mask_arr)
            fake_mask_img.CopyInformation(image_img)
            mask_filepath = os.path.join(patient_directory, 'mask_total.mha')
            sitk.WriteImage(fake_mask_img, mask_filepath)


            try:
                extractor.execute(image_filepath, mask_filepath, voxelBased=True, parallel=True,
                                           exportDirectory=patient_export_directory, selectedFeatureFullNames=selectedFeatureFullNames)
                # post-processing, remove the temporary image and total mask
                if interp:
                    os.remove(image_filepath)
                os.remove(mask_filepath)
                os.remove(os.path.join(patient_export_directory, 'log-sigma-3-0-mm-3D_glcm_SumSquares_32_binCount_mask.mha'))

                # post processing, resample patient_export_directory/log-sigma-3-0-mm-3D_glcm_SumSquares_32_binCount.mha to early_contrast_DCE.mha using ut.resampleImg_as
                if interp:
                    FM_raw_img = sitk.ReadImage(os.path.join(patient_export_directory, 'log-sigma-3-0-mm-3D_glcm_SumSquares_32_binCount.mha'))
                    FM_ref_img = sitk.ReadImage(os.path.join(patient_directory, 'early_contrast_DCE.mha'))
                    FM_recover_img = ut.resampleImg_as(FM_raw_img, FM_ref_img, sitk.sitkLinear)
                    sitk.WriteImage(FM_recover_img, os.path.join(patient_export_directory, 'log-sigma-3-0-mm-3D_glcm_SumSquares_32_binCount.mha'))

            except Exception as e:
                failed_patients.append(patient_id)
                print('Radiomics feature calculation failed for patient {0}: {1}.'.format(patient_id, e))
        else:
            print('Radiomics feature calculation for patient {0} has been done, skipped.'.format(patient_id))

        if debug:
            break

    print('Finally, the following patients failed for radiomics feature extraction: {0}'.format(failed_patients))
    print("The time difference is :", timeit.default_timer() - starttime)

