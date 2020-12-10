from radiants.workflows.segmentation import TumorSegmentation
from core.utils.config import cmdline_input_config, create_subject_list


if __name__ == "__main__":

    PARSER = cmdline_input_config()

    PARSER.add_argument('--gtv_seg_model_dir', '-gtv_md', type=str, default=None,
                        help=('Directory with the model parameters, trained with nnUNet.'))
    PARSER.add_argument('--tumor_seg_model_dir', '-tumor_md', type=str, default=None,
                        help=('Directory with the model parameters, trained with nnUNet.'))
    PARSER.add_argument('--normalize', '-n', action='store_true',
                        help=('Whether or not to normalize the segmented tumors to the '
                              '"reference" and/or "T10" images, if present.'))

    ARGS = PARSER.parse_args()

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, ARGS.xnat_source,
                                             ARGS.cluster_source,
                                             subjects_to_process=[])

    for sub_id in sub_list:

        print('Processing subject {}'.format(sub_id))

        workflow = TumorSegmentation(
            ARGS.gtv_seg_model_dir, ARGS.tumor_seg_model_dir, normalize=ARGS.normalize,
            sub_id=sub_id, input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            local_source=ARGS.local_source, local_sink=ARGS.local_sink,
            local_project_id=ARGS.local_project_id,
            local_basedir=ARGS.local_dir)

        wf = workflow.workflow_setup(check_dependencies=True)
        workflow.runner(wf)

    print('Done!')
