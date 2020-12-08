"Script to run image registration"
from core.utils.config import cmdline_input_config, create_subject_list
from radiants.workflows.registration import RegistrationWorkflow


if __name__ == "__main__":

    PARSER = cmdline_input_config()

    PARSER.add_argument('--num-cores', '-nc', type=int, default=0,
                        help=('Number of cores to use to run the registration workflow '
                              'in parallel. Default is 0, which means the workflow '
                              'will run linearly.'))

    ARGS = PARSER.parse_args()

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, ARGS.xnat_source,
                                             ARGS.cluster_source,
                                             subjects_to_process=[])

    for sub_id in sub_list:
        
        workflow = RegistrationWorkflow(
            sub_id=sub_id, input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            local_source=ARGS.local_source, local_sink=ARGS.local_sink,
            local_project_id=ARGS.local_project_id,
            local_basedir=ARGS.local_dir)

        wf = workflow.workflow_setup()
        workflow.runner(wf, cores=ARGS.num_cores)

    print('Done!')
