"Segmentation workflows"
import nipype
from nipype.interfaces.utility import Merge
from radiants.interfaces.utils import NNUnetPreparation
from radiants.interfaces.mic import HDGlioPredict, NNUnetInference
from core.workflows.base import BaseWorkflow
from nipype.interfaces.ants import ApplyTransforms
from radiants.workflows.registration import RegistrationWorkflow, POSSIBLE_REF


NEEDED_SEQUENCES = ['T1KM', 'T1', 'FLAIR', 'T2']


class TumorSegmentation(BaseWorkflow):
    
    def __init__(self, gtv_model=None, tumor_model=None,
                 normalize=True, **kwargs):
        
        super().__init__(**kwargs)
        self.gtv_model = gtv_model
        self.tumor_model = tumor_model
        self.normalize = normalize

    @staticmethod
    def workflow_inputspecs():

        input_specs = {}

        input_specs['format'] = '.nii.gz'
        dependencies = {}
        dependencies[RegistrationWorkflow] = [
            ['_reg', '.nii.gz', NEEDED_SEQUENCES, 'all'],
            ['_reg2MR_RT_warp', '.nii.gz', ['T1KM', 'T1'], 'mrrt'],
            ['_reg2MR_RT_linear_mat', '.mat', ['T1KM', 'T1'], 'mrrt'],
            ['_reg_reg2RTCT_linear_mat', '.mat', ['T1KM', 'T1'], 'rt']]
        formats = {}
        for k in dependencies:
            for entry in dependencies[k]:
                formats[entry[0]] = entry[1]
        input_specs['data_formats'] = formats
        input_specs['input_suffix'] = [
            '_reg', '_reg2MR_RT_warp', '_reg2MR_RT_linear_mat',
            '_reg_reg2RTCT_linear_mat']
        input_specs['prefix'] = []
        input_specs['dependencies'] = dependencies

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        output_specs['format'] = '.nii.gz'
        output_specs['suffix'] = ['GTV_predicted']
        output_specs['prefix'] = []

        return output_specs

    def datasource(self, create_database=True,
                   dict_sequences=None, check_dependencies=True):
        BaseWorkflow.datasource(self, create_database=create_database,
                                dict_sequences=dict_sequences,
                                check_dependencies=check_dependencies,
                                possible_sequences=NEEDED_SEQUENCES)

    def workflow(self):

        tumor_model = self.tumor_model
        gtv_model = self.gtv_model
        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id

        tosegment = {**dict_sequences['MR-RT'], **dict_sequences['OT']}
        workflow = nipype.Workflow('segmentation_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]
        substitutions += [('/segmentation.nii.gz', '/Tumor_predicted.nii.gz')]
        substitutions += [('/GTV/subject1.nii.gz', '/GTV_predicted.nii.gz')]
        substitutions += [('/tumor/subject1.nii.gz', '/GTV_predicted_2modalities.nii.gz')]

        mr_rt_ref = None
        rtct = None

        if dict_sequences['MR-RT'] and self.normilize_mr_rt:
            ref_session = list(dict_sequences['MR-RT'].keys())[0]
            ref_scans = dict_sequences['MR-RT'][ref_session]['scans']
            for pr in POSSIBLE_REF:
                for scan in ref_scans:
                    if pr in scan.split('_')[0]:
                        mr_rt_ref = scan.split('_')[0]
                        break
                else:
                    continue
                break

        if dict_sequences['RT'] and self.normilize_rtct:
            rt_session = list(dict_sequences['RT'].keys())[0]
            ct_name = dict_sequences['RT'][rt_session]['rtct']
            if ct_name is not None and mr_rt_ref is not None:
                rtct = '{0}_rtct'.format(rt_session, ct_name)

        for key in tosegment:
            session = tosegment[key]
            if session['scans'] is not None:
                scans = session['scans']
#                 scans = [x for x in scans if 'reg2' not in x]
                if ('T1KM_reg' in scans and 'FLAIR_reg' in scans
                        and 'T1_reg' in scans and 'T2_reg' in scans):
                    hd_glio = True
                else:
                    hd_glio = True
                if 'T1KM_reg' not in scans or 'FLAIR_reg' not in scans:
                    two_modalities_seg = False
                else:
                    two_modalities_seg = True
                if hd_glio:
                    tumor_seg  = nipype.Node(
                        interface=HDGlioPredict(),
                        name='{}_tumor_segmentation'.format(key))
                    tumor_seg.inputs.out_file = 'Tumor_segmentation'
                    workflow.connect(datasource, '{}_T1KM_reg'.format(key),
                                     tumor_seg, 'ct1')
                    workflow.connect(datasource, '{}_T2_reg'.format(key),
                                     tumor_seg, 't2')
                    workflow.connect(datasource, '{}_FLAIR_reg'.format(key),
                                     tumor_seg, 'flair')
                    workflow.connect(datasource, '{}_T1_reg'.format(key),
                                     tumor_seg, 't1')
                    workflow.connect(
                        tumor_seg, 'out_file', datasink,
                        'results.subid.{}.@tumor_seg'.format(key))
                    workflow, datasink = self.apply_transformations(
                        'out_file', tumor_seg, 'Tumor_segmentation', datasource,
                        workflow, mr_rt_ref, rtct, key, ref_session, datasink)
                
                if two_modalities_seg:
                    mi = nipype.Node(Merge(2), name='{}_merge'.format(key))
                    gtv_seg_data_prep = nipype.Node(
                        interface=NNUnetPreparation(),
                        name='{}_two_modalities_seg_data_prep'.format(key))
                    workflow.connect(datasource, '{}_T1KM_reg'.format(key),
                                     mi, 'in1')
                    workflow.connect(datasource, '{}_FLAIR_reg'.format(key),
                                     mi, 'in2')
                    workflow.connect(mi, 'out', gtv_seg_data_prep,
                                     'images')
                    if gtv_model is not None:
                        gtv_seg = nipype.Node(
                            interface=NNUnetInference(),
                            name='{}_gtv_segmentation'.format(key))
                        gtv_seg.inputs.model_folder = gtv_model
                        gtv_seg.inputs.prefix = 'gtv'
                        workflow.connect(
                            gtv_seg_data_prep, 'output_folder',
                            gtv_seg, 'input_folder')
                        workflow.connect(gtv_seg, 'output_file', datasink,
                                         'results.subid.{}.GTV.@gtv_seg'.format(key))
                        workflow, datasink = self.apply_transformations(
                            'output_file', gtv_seg, 'GTV_segmentation', datasource,
                            workflow, mr_rt_ref, rtct, key, ref_session, datasink)
                    if tumor_model is not None:
                        tumor_seg_2mods = nipype.Node(
                            interface=NNUnetInference(),
                            name='{}_tumor_seg_2mods'.format(key))
                        tumor_seg_2mods.inputs.model_folder = tumor_model
                        tumor_seg_2mods.inputs.prefix = 'tumor_2mod'
                        workflow.connect(
                            gtv_seg_data_prep, 'output_folder',
                            tumor_seg_2mods, 'input_folder')
                        workflow.connect(tumor_seg_2mods, 'output_file', datasink,
                                         'results.subid.{}.tumor.@tumor_seg2mod'.format(key))
                        workflow, datasink = self.apply_transformations(
                            'output_file', tumor_seg_2mods, 'Tumor_segmentation_2modalities',
                            datasource, workflow, mr_rt_ref, rtct, key, ref_session,
                            datasink)

        datasink.inputs.substitutions = substitutions

        return workflow

    def apply_transformations(self, tonormalize, node, node_name, datasource,
                              workflow, mr_rt_ref, rtct, session, ref_session,
                              datasink):

        if rtct is not None:
            apply_ts_rt_ref = nipype.Node(
                interface=ApplyTransforms(),
                name='{}_{}_norm2RT'.format(session, node_name))
            apply_ts_rt_ref.inputs.output_image = (
                '{}_reg2RTCT.nii.gz'.format(node_name))
            workflow.connect(node, tonormalize, apply_ts_rt_ref,
                             'input_image')
            workflow.connect(datasource, rtct, apply_ts_rt_ref,
                             'reference_image')
            workflow.connect(
                apply_ts_rt_ref, 'output_image', datasink,
                'results.subid.{0}.@{1}_reg2RTCT'.format(
                    session, node_name))
            if session != ref_session:
                merge_rt_ref = nipype.Node(interface=Merge(3),
                                name='{}_{}_merge_rt'.format(session, node_name))
                merge_rt_ref.inputs.ravel_inputs = True
                workflow.connect(datasource, '{}_{}_reg_reg2RTCT_linear_mat'.format(
                    ref_session, mr_rt_ref), merge_rt_ref, 'in1')
                workflow.connect(datasource, '{}_{}_reg2MR_RT_linear_mat'.format(
                    session, mr_rt_ref), merge_rt_ref, 'in3')
                workflow.connect(datasource, '{}_{}_reg2MR_RT_warp'.format(
                    session, mr_rt_ref), merge_rt_ref, 'in2')
                workflow.connect(merge_rt_ref, 'out', apply_ts_rt_ref, 'transforms')
            else:
                workflow.connect(datasource, '{}_{}_reg_reg2RTCT_linear_mat'.format(
                session, mr_rt_ref), apply_ts_rt_ref, 'transforms')

        if mr_rt_ref is not None and session != ref_session:
            
            apply_ts_rt_ref = nipype.Node(
                interface=ApplyTransforms(),
                name='{0}_{1}_norm2MR_RT'.format(session, node_name))
            apply_ts_rt_ref.inputs.output_image = (
                '{}_reg2MR_RT.nii.gz'.format(node_name))
            workflow.connect(node, tonormalize, apply_ts_rt_ref,
                             'input_image')
            workflow.connect(datasource, '{}_{}_reg'.format(
                    session, mr_rt_ref), apply_ts_rt_ref, 'reference_image')
            workflow.connect(
                apply_ts_rt_ref, 'output_image', datasink,
                'results.subid.{0}.@{1}_reg2MR_RT'.format(
                    session, node_name))

            merge_rt_ref = nipype.Node(interface=Merge(2),
                            name= '{0}_{1}_merge_mrrt'.format(session, node_name))
            merge_rt_ref.inputs.ravel_inputs = True
            workflow.connect(datasource, '{0}_{1}_reg2MR_RT_linear_mat'.format(
                session, mr_rt_ref), merge_rt_ref, 'in2')
            workflow.connect(datasource, '{0}_{1}_reg2MR_RT_warp'.format(
                session, mr_rt_ref), merge_rt_ref, 'in1')
            workflow.connect(merge_rt_ref, 'out', apply_ts_rt_ref, 'transforms')
    
        return workflow, datasink
