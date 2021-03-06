from radiants.workflows.segmentation import TumorSegmentation
from radiants.workflows.registration import RegistrationWorkflow
from radiants.workflows.bet import BETWorkflow
from radiants.utils.config import cmdline_input_config, create_subject_list
from radiants.workflows.radiomics import Radiomics


def main():

    PARSER = cmdline_input_config()
    
    ARGS = PARSER.parse_args()
    
    if ARGS.workflow == 'bet':

        workflow = BETWorkflow

    elif ARGS.workflow == 'registration':

        workflow = RegistrationWorkflow

    elif ARGS.workflow == 'segmentation':

        workflow = TumorSegmentation
    
    elif ARGS.workflow == 'radiomics':

        workflow = Radiomics

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, subjects_to_process=[])

    for sub_id in sub_list:

        print('Processing subject {}'.format(sub_id))

        workflow_st = workflow(
            sub_id=sub_id, input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            local_sink=ARGS.local_sink, normilize_mr_rt=ARGS.normalize_to_mr_rt,
            local_project_id=ARGS.local_project_id, normilize_rtct=ARGS.normalize_to_rt_ct,
            local_basedir=ARGS.local_dir, gtv_model=ARGS.gtv_seg_model_dir,
            tumor_model=ARGS.tumor_seg_model_dir, cores=ARGS.num_cores)

        wf = workflow_st.workflow_setup(check_dependencies=True)
        workflow_st.runner(wf)

    print('Done!')


if __name__ == "__main__":
    main()
