import os, timeit, ray
import pandas as pd

from radiomics_distributed import featureextractor, utils as ut, preprocessing
from tqdm import tqdm


def cohort_calculation_pipeline(label, modality_name, mask_name, preprocessed_data_directory,
                                   export_directory,image_feature_extraction_parameter_path):
    cache_frequency = 32

    final_failed_patients = []
    feature_filepath = os.path.join(export_directory,'roi_features_label_{}.csv'.format(str(label).zfill(2)))
    image_feature_extraction_parameters = ut.parse_from_yaml(image_feature_extraction_parameter_path)['radiomicsCalculation']
    image_feature_extraction_parameters['setting']['label'] = label

    extractor = featureextractor.RadiomicsFeatureExtractor(image_feature_extraction_parameters)

    feature_table = []
    existing_patient_ids = []
    if os.path.exists(feature_filepath):
        current_feature_table = pd.read_csv(feature_filepath, index_col=0)
        existing_patient_ids = current_feature_table.columns.values
        feature_table.append(current_feature_table)

    patient_ids = [patient_id for patient_id in os.listdir(preprocessed_data_directory) if patient_id not in existing_patient_ids]
    patient_num = len(patient_ids)

    starttime = timeit.default_timer()

    patient_index = 0
    failed_patients = []
    with tqdm(total=patient_num) as pbar:
        while patient_index < patient_num:
            patient_ending_index = patient_index + cache_frequency
            if patient_ending_index > patient_num:
                patient_ending_index = patient_num

            futures = dict()
            for patient_id in patient_ids[patient_index:patient_ending_index]:
                patient_directory = os.path.join(preprocessed_data_directory, patient_id)
                if not os.path.isdir(patient_directory):
                    continue

                image_filepath = os.path.join(patient_directory, '{}.mha'.format(modality_name))
                mask_filepath = os.path.join(patient_directory, '{}.mha'.format(mask_name))
                try:
                    future = extractor.execute(image_filepath, mask_filepath, parallel=True)
                    print('Remote job submitted for patient {0}.'.format(patient_id))
                    futures[future] = patient_id
                except Exception as e:
                    print('Feature calculation failed for patient {0}: {1}.'.format(patient_id, e))
                    final_failed_patients.append(patient_id)

            unfinished_futures = list(futures.keys())
            while len(unfinished_futures):
                finished_futures, unfinished_futures = ray.wait(unfinished_futures)
                patient_id = futures[finished_futures[0]]
                try:
                    result = ray.get(finished_futures[0], timeout=1)
                    feature_series = pd.Series(result)
                    feature_series.name = patient_id
                    feature_table.append(feature_series)
                    pbar.update(1)
                except Exception as e:
                    print('Radiomics feature calculation failed for patient {0}: {1}.'.format(patient_id, e))
                    failed_patients.append(patient_id)
            try:
                feature_table = pd.concat(feature_table,axis=1)
                feature_table.to_csv(feature_filepath)
                feature_table = [feature_table]
            except Exception as e:
                print(e)
            patient_index += cache_frequency

        for patient_id in failed_patients:
            patient_directory = os.path.join(preprocessed_data_directory, patient_id)
            if not os.path.isdir(patient_directory):
                continue
            image_filepath = os.path.join(patient_directory, '{}.mha'.format(modality_name))
            mask_filepath = os.path.join(patient_directory, '{}.mha'.format(mask_name))
            try:
                print('Calculating radiomics features for patient {0} in sequence.'.format(patient_id))
                single_patient_radiomics = extractor.execute(image_filepath, mask_filepath)
            except Exception as e:
                print('Sequential radiomics feature calculation failed for patient {0}: {1}.'.format(patient_id, e))
                final_failed_patients.append(patient_id)
                continue
            feature_series = pd.Series(single_patient_radiomics)
            feature_series.name = patient_id
            feature_table.append(feature_series)
            cache_counter += 1
            if cache_counter >= cache_frequency:
                cache_counter = 0
                pd.concat(feature_table, axis=1).to_csv(feature_filepath)
            pbar.update(1)

    print("The time difference is :", timeit.default_timer() - starttime)
    feature_table = pd.concat(feature_table, axis=1)
    feature_table.to_csv(feature_filepath)
    print('Finally, the following patients failed for radiomics feature extraction: {0}'.format(final_failed_patients))



def cohort_preprocessing_pipeline(image_feature_extraction_parameter_path, preprocessing_export_directory,
                                    cleaned_database_directory):

    image_feature_extraction_parameters = ut.parse_from_yaml(image_feature_extraction_parameter_path)
    image_preprocessing_parameters = image_feature_extraction_parameters.get('preprocessing')

    patient_ids = os.listdir(cleaned_database_directory)
    roi_names = image_feature_extraction_parameters.get('roiNames', [])

    for patient_id in patient_ids:
        cleaned_patient_database_directory = os.path.join(cleaned_database_directory, patient_id)
        patient_preprocessing_export_directory = os.path.join(preprocessing_export_directory, patient_id)
        if not os.path.exists(cleaned_patient_database_directory):
            return
        if not patient_preprocessing_export_directory:
            os.makedirs(patient_preprocessing_export_directory)

        preprocessing.single_patient_image_preprocessing(cleaned_patient_database_directory,
                                                         patient_preprocessing_export_directory,
                                                         image_preprocessing_parameters, roi_names)


if __name__ == '__main__':
    perform_preprocessing = False
    perform_feature_extraction = True

    label = 1
    modality_name = 'T50_ct'
    mask_name = 'T50_mask'

    cleaned_database_directory = r'.\demo_data\Std_4DCT'
    preprocessed_data_directory = r'.\demo_data\Std_4DCT'
    export_directory = r'.\demo_data\radiomics_output'
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    image_feature_extraction_parameter_path = r'./parameters/image_feature_extraction_parameters.yaml'
    if perform_preprocessing:
        print('-'*32)
        print('Preprocessing data...')
        cohort_preprocessing_pipeline(image_feature_extraction_parameter_path, preprocessing_export_directory=preprocessed_data_directory,
                                      cleaned_database_directory=cleaned_database_directory)
        print('Preprocessing finished.')
    else:
        print('Preprocessing skipped.')

    if perform_feature_extraction:
        print('-'*32)
        print('Extracting ROI-wise radiomics features...')
        cohort_calculation_pipeline(label, modality_name, mask_name, preprocessed_data_directory,
                                   export_directory, image_feature_extraction_parameter_path)
        print('Radiomics feature extraction finished.')
    else:
        print('Radiomics feature extraction skipped.')
