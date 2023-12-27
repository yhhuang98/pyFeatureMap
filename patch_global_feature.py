import os, timeit, ray
import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from radiomics_distributed import featureextractor, utils as ut




def patch_calculation_pipeline(patient_id, patch_num, modality_name, mask_name,
                                   preprocessed_data_directory, export_directory, image_feature_extraction_parameter_path):
    cache_frequency = 32
    final_failed_roi = []
    feature_filepath = os.path.join(export_directory, 'patch_features_px_{}.csv'.format(patient_id))
    image_feature_extraction_parameters = ut.parse_from_yaml(image_feature_extraction_parameter_path)['radiomicsCalculation']

    feature_table = []
    if os.path.exists(feature_filepath):
        current_feature_table = pd.read_csv(feature_filepath, index_col=0)
        feature_table.append(current_feature_table)

    roi_ids = range(1, patch_num + 1)
    roi_num = len(roi_ids)

    starttime = timeit.default_timer()

    roi_index = 0
    failed_roi = []
    with tqdm(total=roi_num) as pbar:
        while roi_index < roi_num:
            roi_ending_index = roi_index+cache_frequency
            if roi_ending_index > roi_num:
                roi_ending_index = roi_num
            print('Remote jobs submitted for ROI {0} to {1}.'.format(str(roi_ids[roi_index]).zfill(4), str(roi_ids[roi_ending_index-1]).zfill(4)))
            futures = dict()
            for roi_id in roi_ids[roi_index:roi_ending_index]:
                image_feature_extraction_parameters['setting']['label'] = roi_id
                extractor = featureextractor.RadiomicsFeatureExtractor(image_feature_extraction_parameters)
                roi_id = 'ROI_'+str(roi_id).zfill(4)

                patient_directory = os.path.join(preprocessed_data_directory, patient_id)
                if not os.path.isdir(patient_directory):
                    continue
                image_filepath = os.path.join(patient_directory, '{}.mha'.format(modality_name))
                mask_filepath = os.path.join(patient_directory, '{}.mha'.format(mask_name))
                try:
                    future = extractor.execute(image_filepath, mask_filepath, parallel=True)
                    futures[future] = roi_id
                except Exception as e:
                    print('Feature calculation failed for ROI {0}: {1}.'.format(roi_id, e))
                    final_failed_roi.append(roi_id)
            unfinished_futures = list(futures.keys())
            while len(unfinished_futures):
                finished_futures, unfinished_futures = ray.wait(unfinished_futures)
                roi_id = futures[finished_futures[0]]
                try:
                    result = ray.get(finished_futures[0], timeout=1)
                    feature_series = pd.Series(result)
                    feature_series.name = roi_id
                    feature_table.append(feature_series)
                    pbar.update(1)
                except Exception as e:
                    print('Radiomics feature calculation failed for ROI {0}: {1}.'.format(roi_id, e))
                    final_failed_roi.append(roi_id)
            try:
                feature_table = pd.concat(feature_table, axis=1)
                feature_table.to_csv(feature_filepath)
                feature_table = [feature_table]
            except Exception as e:
                print(e)
            roi_index += cache_frequency
        for roi_id in failed_roi:
            patient_directory = os.path.join(preprocessed_data_directory, patient_id)
            if not os.path.isdir(patient_directory):
                continue
            image_filepath = os.path.join(patient_directory, '{}.mha'.format(modality_name))
            mask_filepath = os.path.join(patient_directory, '{}.mha'.format(mask_name))
            try:
                print('Calculating radiomics features for ROI {0} in sequence.'.format(roi_id))
                single_roi_radiomics = extractor.execute(image_filepath, mask_filepath)
            except Exception as e:
                print('Sequential radiomics feature calculation failed for ROI {0}: {1}.'.format(roi_id, e))
                final_failed_roi.append(roi_id)
                continue
            feature_series = pd.Series(single_roi_radiomics)
            feature_series.name = roi_id
            feature_table.append(feature_series)
            cache_counter += 1
            if cache_counter >= cache_frequency:
                cache_counter = 0
                pd.concat(feature_table, axis=1).to_csv(feature_filepath)
            pbar.update(1)
    print("The time difference is :", timeit.default_timer() - starttime)
    feature_table = pd.concat(feature_table, axis=1)
    feature_table.to_csv(feature_filepath)
    print('Finally, the following patients failed for radiomics feature extraction: {0}'.format(final_failed_roi))




if __name__ == '__main__':
    debug = True

    modality_name = 'T50_ct'
    mask_name = 'init_subregion_T50_15_non_corrected'

    preprocessed_data_directory = r'.\demo_data\Std_4DCT'
    export_directory = r'.\demo_data\radiomics_output'
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    image_feature_extraction_parameter_path = r'./parameters/image_feature_extraction_parameters.yaml'
    patient_ids = [patient_id for patient_id in os.listdir(preprocessed_data_directory)]
    for patient_id in patient_ids:
        if debug == True:
            patient_id = patient_ids[0]

        patch_mask = sitk.ReadImage(os.path.join(preprocessed_data_directory, patient_id, mask_name + '.mha'))
        patch_num = np.max(sitk.GetArrayFromImage(patch_mask))
        print('{0} patches planned for {1}.'.format(str(patch_num), patient_id))

        patch_calculation_pipeline(patient_id, patch_num, modality_name, mask_name, preprocessed_data_directory,
                                   export_directory, image_feature_extraction_parameter_path)

        if debug == True:
            break
