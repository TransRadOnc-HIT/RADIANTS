import argparse
from radiants.workflows.lung_segmentation import LungSegmentation
from radiants.utils.config import create_subject_list


def main():

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--input_dir', '-i', type=str,
                        help=('Exisisting directory with the subject(s) to process'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--weights', nargs='+', type=str, default=None,
                        help=('Path to the CNN weights to be used for the inference '
                              ' More than one weight can be used, in that case the median '
                              'prediction will be returned.'))
    PARSER.add_argument('--num-cores', '-nc', type=int, default=0,
                        help=('Number of cores to use to run the workflow '
                              'in parallel. Default is 0, which means the workflow '
                              'will run linearly.'))
    
    ARGS = PARSER.parse_args()

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, subjects_to_process=[])

    for sub_id in sub_list:

        print('Processing subject {}'.format(sub_id))

        workflow_st = LungSegmentation(
            sub_id=sub_id, input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            network_weights=ARGS.weights, cores=ARGS.num_cores)

        wf = workflow_st.workflow_setup(check_dependencies=True)
        workflow_st.runner(wf)

    print('Done!')


if __name__ == "__main__":
    main()
