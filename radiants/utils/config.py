import argparse
import os


POSSIBLE_WORKFLOWS = ['bet', 'registration', 'segmentation', 'radiomics']


def cmdline_input_config():

    PARSER = argparse.ArgumentParser()
    
    PARSER.add_argument('--input_dir', '-i', type=str,
                        help=('Exisisting directory with the subject(s) to process'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--workflow', '-wf', type=str, choices=POSSIBLE_WORKFLOWS,
                        default='bet',
                        help=('Workflow that has to be run. If the inputs for the '
                              'choosen workflow does not exist, Radiants will try to'
                              'generate them automatically based on the workflow '
                              'dependencies. Default is "bet".'))
    PARSER.add_argument('--normalize-to-mr-rt', '-nmr', action='store_true',
                        help=('Whether or not to register each MR image in post RT folder '
                              'to the cT1 or T1 in the MR-RT directory (if present). '
                              'This is only considered if "registration" workflow is '
                              'selected. Default is False.'))
    PARSER.add_argument('--normalize-to-rt-ct', '-nct', action='store_true',
                        help=('Whether or not to normalize each MR image in post RT folder '
                              'to the CT used for RT planning in the RT directory (if present). '
                              'This is only considered if "registration" workflow is '
                              'selected. Default is False. '
                              'N.B. if MR-RT is not present, this step cannot be performed!'))
    PARSER.add_argument('--gtv_seg_model_dir', '-gtv_md', type=str, default=None,
                        help=('Directory with the model parameters to segment the GTV. '
                              'It is assumed that the network has been trained with nnUNet, '
                              'using cT1 and FLAIR contrasts. This is only considered if '
                              '"segmentation" workflow is selected.'))
    PARSER.add_argument('--tumor_seg_model_dir', '-tumor_md', type=str, default=None,
                        help=('Directory with the model parameters to segment the tumor into '
                              '4 different classes (based on BRATS challenge). '
                              'It is assumed that the network has been trained with nnUNet, '
                              'using cT1 and FLAIR contrasts. This is only considered if '
                              '"segmentation" workflow is selected.'))
    PARSER.add_argument('--num-cores', '-nc', type=int, default=0,
                        help=('Number of cores to use to run the workflow '
                              'in parallel. Default is 0, which means the workflow '
                              'will run linearly.'))
    PARSER.add_argument('--local-sink', '-ls', action='store_true',
                        help=('Whether or not to store the processed files in a local database. '
                              'Default is False.'))
    PARSER.add_argument('--local-project-id', '-lpid', type=str,
                        help=('Local project ID. If not provided, and local-sink was selected,'
                              ' the default name for the database will be "Radiants_database".'))
    PARSER.add_argument('--local-dir', '-ldir', type=str,
                        help=('Directory to store the local database. If not provided, and '
                              'local-sink was selected, the "work_dir" will be used.'))
    
    return PARSER


def create_subject_list(base_dir, subjects_to_process=[]):
    
    if os.path.isdir(base_dir):
        sub_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
        if subjects_to_process:
            sub_list = [x for x in sub_list if x in subjects_to_process]

    return sub_list, base_dir
