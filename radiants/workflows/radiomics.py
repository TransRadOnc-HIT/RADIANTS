"Segmentation workflows"
import nipype
from nipype.interfaces.utility import Merge
from radiants.interfaces.utils import NNUnetPreparation
from radiants.interfaces.mic import HDGlioPredict, NNUnetInference
from core.workflows.base import BaseWorkflow
from nipype.interfaces.ants import ApplyTransforms
from radiants.workflows.registration import RegistrationWorkflow, POSSIBLE_REF
from radiants.workflows.segmentation import TumorSegmentation
from radiants.interfaces.pyradiomics import FeatureExtraction


NEEDED_SEQUENCES = ['T1KM', 'T1', 'FLAIR', 'T2']


class Radiomics(BaseWorkflow):
    
    def __init__(self, images=[], rois=[], **kwargs):
        
        super().__init__(**kwargs)
        self.images = images
        self.rois = rois
        self.additional_inputs = images+rois

    @staticmethod
    def workflow_inputspecs(additional_inputs=None):

        input_specs = {}

        input_specs['format'] = '.nii.gz'
        dependencies = {}
        input_specs['inputs'] = {
            'GTVPredicted': {'mandatory': True, 'format': '.nii.gz',
                             'dependency': TumorSegmentation,
                             'possible_sequences': [], 'multiplicity': 'all',
                             'composite': ['T1KM_reg', 'FLAIR_reg']},
            'GTVPredicted-2modalities': {
                'mandatory': True, 'format': '.nii.gz',
                'dependency': TumorSegmentation, 'possible_sequences': [],
                'multiplicity': 'all', 'composite': ['T1KM_reg', 'FLAIR_reg']},
            'TumorPredicted': {'mandatory': False, 'format': '.nii.gz',
                               'dependency': TumorSegmentation,
                               'possible_sequences': [], 'multiplicity': 'all',
                               'composite': ['T1KM_reg', 'FLAIR_reg', 'T1_reg', 'T2_reg']},
            '_reg': {'mandatory': True, 'format': '.nii.gz', 'dependency': RegistrationWorkflow,
                     'possible_sequences': NEEDED_SEQUENCES, 'multiplicity': 'all',
                     'composite': None}}
        dependencies[TumorSegmentation] = {
            'GTVPredicted': {'mandatory': False, 'format': '.nii.gz'},
            'GTVPredicted-2modalities': {'mandatory': False, 'format': '.nii.gz'},
            'TumorPredicted': {'mandatory': False, 'format': '.nii.gz'}}
        dependencies[RegistrationWorkflow] = {
            '_reg': {'mandatory': False, 'format': '.nii.gz'}}
        formats = {}
        for k in dependencies:
            for entry in dependencies[k]:
                formats[entry] = dependencies[k][entry]['format']
        input_specs['data_formats'] = formats
        input_specs['dependencies'] = dependencies
        input_specs['additional_inputs'] = additional_inputs

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        output_specs['outputs'] = {
            'FeaturesPyradiomics': {'possible_sequences': [], 'format': '.txt',
                                    'multiplicity': 'all',
                                    'composite': None}}

        return output_specs

    def datasource(self, create_database=True,
                   dict_sequences=None, check_dependencies=True):
        BaseWorkflow.datasource(self, create_database=create_database,
                                dict_sequences=dict_sequences,
                                check_dependencies=check_dependencies,
                                possible_sequences=NEEDED_SEQUENCES)

    def workflow(self):

        images = self.images
        rois = self.rois
        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id

        toextract = {**dict_sequences['MR-RT'], **dict_sequences['OT']}
        workflow = nipype.Workflow('features_extraction_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]

        for key in toextract:
            session = toextract[key]
            if session['scans'] is not None:
                scans = session['scans']
                reg_scans = [x for x in scans if x.endswith('_reg')]
                segmented_masks = [x for x in scans if x in ['GTVPredicted',
                                                             'TumorPredicted',
                                                             'GTVPredicted-2modalities']]
                add_scans = [x for x in scans if x in images]
                add_masks = [x for x in scans if x in rois]
                
                for image in reg_scans:
                    for roi in segmented_masks:
                        image_name = '{}_{}_reg'.format(key, image.split('_')[0])
                        roi_name = '{}_{}'.format(key, roi.split('.nii.gz')[0])
                        features = nipype.Node(
                            interface=FeatureExtraction(),
                            name='features_extraction_{}{}'.format(image_name, roi_name))
                        features.inputs.parameter_file = '/home/fsforazz/git/core/resources/Params_MR.yaml'
                        workflow.connect(datasource, image_name, features, 'input_image')
                        workflow.connect(datasource, roi_name, features, 'rois')
                        workflow.connect(features, 'feature_files', datasink,
                                         'results.subid.{0}.@csv_file_{1}{2}'.format(
                                             key, image_name, roi_name))
                for image in add_scans:
                    for roi in add_masks:
                        image_name = '{}_{}'.format(key, image)
                        roi_name = '{}_{}'.format(key, roi.split('.nii.gz')[0])
                        features = nipype.Node(
                            interface=FeatureExtraction(),
                            name='features_extraction_{}{}'.format(image_name, roi_name))
                        features.inputs.parameter_file = '/home/fsforazz/git/core/resources/Params_MR.yaml'
                        workflow.connect(datasource, image_name, features, 'input_image')
                        workflow.connect(datasource, roi_name, features, 'rois')
                        workflow.connect(features, 'feature_files', datasink,
                                         'results.subid.{0}.@csv_file_{1}{2}'.format(
                                             key, image_name, roi_name))

        datasink.inputs.substitutions = substitutions

        return workflow
