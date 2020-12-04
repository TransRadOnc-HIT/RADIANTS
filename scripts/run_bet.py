"Script to run brain extraction using HD-BET"
from radiants.workflows.bet import BETWorkflow
from core.utils.config import cmdline_input_config, create_subject_list


if __name__ == "__main__":

    PARSER = cmdline_input_config()

    ARGS = PARSER.parse_args()

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, ARGS.xnat_source,
                                             ARGS.cluster_source,
                                             subjects_to_process=[])

    for sub_id in sub_list:

        print('Processing subject {}'.format(sub_id))

        workflow = BETWorkflow(
            sub_id=sub_id, input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            local_source=ARGS.local_source, local_sink=ARGS.local_sink,
            local_project_id=ARGS.local_project_id,
            local_basedir=ARGS.local_dir)

        wf = workflow.workflow_setup()
        workflow.runner(wf)

    print('Done!')
