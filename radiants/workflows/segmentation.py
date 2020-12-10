"Segmentation workflows"
import nipype
from nipype.interfaces.utility import Merge
from radiants.interfaces.utils import NNUnetPreparation
from radiants.interfaces.mic import HDGlioPredict, NNUnetInference
from core.workflows.base import BaseWorkflow
from nipype.interfaces.ants import ApplyTransforms
from radiants.workflows.registration import RegistrationWorkflow
from radiants.workflows.bet import BETWorkflow


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
        dependencies[RegistrationWorkflow] = [['_reg', '.nii.gz']]
#         dependencies[BETWorkflow] = [['_preproc', '.nii.gz']]
        input_specs['input_suffix'] = ['_reg']
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

        for key in tosegment:
            session = tosegment[key]
            if session['scans'] is not None:
                scans = session['scans']
                scans = [x for x in scans if 'reg' in x]
                if len(scans) != 4:
                    hd_glio = False
                elif len(scans) == 4:
                    hd_glio = True
                if 'T1KM_reg' not in scans or 'FLAIR_reg' not in scans:
                    two_modalities_seg = False
                else:
                    two_modalities_seg = True
                if hd_glio:
                    tumor_seg  = nipype.Node(
                        interface=HDGlioPredict(),
                        name='{}_tumor_segmentation'.format(key))
                    tumor_seg.inputs.out_file = 'tumor_segmentation'
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
                                         'results.subid.{}.@gtv_seg'.format(key))
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
                                         'results.subid.{}.@tumor_seg2mod'.format(key))

        datasink.inputs.substitutions = substitutions

        return workflow

    def apply_transformations(self):
        
        base_workflow = self.seg_workflow

        datasource = self.data_source
        to_transform = self.to_transform

        workflow = nipype.Workflow('apply_transformations_workflow',
                                   base_dir=self.nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=self.result_dir), "datasink")
    
        substitutions = [('subid', self.sub_id)]
        substitutions += [('results/', 'tumor_segmentation_results/')]
        for image in to_transform:
            base_name = image.replace('.', '_')
            outname = image.split('.')[0].upper()
            if self.reference:
                apply_ts_ref = nipype.MapNode(interface=ApplyTransforms(),
                                              iterfield=['input_image', 'transforms'],
                                              name='apply_ts_ref{}'.format(base_name))
                apply_ts_ref.inputs.interpolation = 'NearestNeighbor'
        
                workflow.connect(datasource, 'reference', apply_ts_ref, 'reference_image')
                if self.t10:
                    merge_ref_ts = nipype.MapNode(interface=Merge(3),
                                                  iterfield=['in1', 'in2', 'in3'],
                                                  name='merge_ct_ts{}'.format(base_name))
                    workflow.connect(datasource, 't12ct_mat', merge_ref_ts, 'in1')
                    workflow.connect(datasource, 'reg2t1_warp', merge_ref_ts, 'in2')
                    workflow.connect(datasource, 'reg2t1_mat', merge_ref_ts, 'in3')
                    workflow.connect(merge_ref_ts, 'out', apply_ts_ref, 'transforms')
                else:
                    workflow.connect(datasource, 't12ct_mat', apply_ts_ref, 'transforms')
                workflow.connect(base_workflow, image, apply_ts_ref, 'input_image')
                workflow.connect(apply_ts_ref, 'output_image', datasink,
                                 'results.subid.@{}_reg2ref'.format(base_name))
        
            if self.t10:
                merge_t10_ts = nipype.MapNode(interface=Merge(2),
                                              iterfield=['in1', 'in2'],
                                              name='merge_t10_ts{}'.format(base_name))
                apply_ts_t10 = nipype.MapNode(interface=ApplyTransforms(),
                                              iterfield=['input_image', 'transforms'],
                                              name='apply_ts_t10{}'.format(base_name))
                apply_ts_t10.inputs.interpolation = 'NearestNeighbor'
        
                workflow.connect(datasource, 't1_0', apply_ts_t10, 'reference_image')
                workflow.connect(datasource, 'reg2t1_warp', merge_t10_ts, 'in1')
                workflow.connect(datasource, 'reg2t1_mat', merge_t10_ts, 'in2')
                workflow.connect(merge_t10_ts, 'out', apply_ts_t10, 'transforms')
            
                workflow.connect(base_workflow, image, apply_ts_t10, 'input_image')
                workflow.connect(apply_ts_t10, 'output_image', datasink,
                                 'results.subid.@{}_reg2T10'.format(base_name))
     
            for i, session in enumerate(self.sessions):
                substitutions += [('_apply_ts_t10{0}{1}/{2}_trans.nii.gz'
                                   .format(base_name, i, to_transform[image]),
                                   session+'/'+'{}_reg2T1ref.nii.gz'.format(outname))]
                substitutions += [('_apply_ts_ref{0}{1}/{2}_trans.nii.gz'
                                   .format(base_name, i, to_transform[image]),
                                   session+'/'+'{}_reg2CT.nii.gz'.format(outname))]
    
        datasink.inputs.substitutions =substitutions

        workflow = self.datasink(workflow, datasink)
    
        return workflow
