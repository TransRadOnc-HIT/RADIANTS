"Lung segmentation workflows"
import nipype
from core.workflows.base import BaseWorkflow
from radiants.interfaces.custom_preproc import LungSegmentationPreproc
from radiants.interfaces.custom import LungSegmentationInference


class LungSegmentation(BaseWorkflow):
    
    def __init__(self, network_weights, new_spacing=[0.35, 0.35, 0.35],
                 **kwargs):
        
        super().__init__(**kwargs)
        self.network_weights = network_weights
        self.new_spacing = new_spacing

    @staticmethod
    def workflow_inputspecs(additional_inputs=None):

        input_specs = {}

        input_specs['format'] = '.nii.gz'
        input_specs['inputs'] = {
            '': {'mandatory': True, 'format': '.nii.gz', 'dependency': None,
                     'possible_sequences': [], 'multiplicity': 'all',
                     'composite': None}}
        input_specs['data_formats'] = {'': '.nii.gz'}
        input_specs['dependencies'] = {}
        input_specs['additional_inputs'] = additional_inputs

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        output_specs['outputs'] = {'_lung_segmented': {'possible_sequences': [], 'format': '.nii.gz',
                                                       'multiplicity': 'all',
                                                       'composite': None}}

        return output_specs

    def datasource(self, create_database=True,
                   dict_sequences=None, check_dependencies=True):
        BaseWorkflow.datasource(self, create_database=create_database,
                                dict_sequences=dict_sequences,
                                check_dependencies=check_dependencies)

    def workflow(self):

        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id

        toseg = {**dict_sequences['OT']}
        workflow = nipype.Workflow('lung_segmentation_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]
        substitutions += [('_preproc_corrected.', '_preproc.')]
        datasink.inputs.substitutions = substitutions

        for key in toseg:
            files = []
#             if tobet[key]['ref'] is not None:
#                 files.append(tobet[key]['ref'])
            if toseg[key]['scans'] is not None:
                files = files + toseg[key]['scans']
            for el in files:
                el = el.strip(self.extention)
                node_name = '{0}_{1}'.format(key, el)
                preproc = nipype.Node(
                    interface=LungSegmentationPreproc(),
                    name='{}_ls_preproc'.format(node_name))
                preproc.inputs.new_spacing = self.new_spacing
                lung_seg = nipype.Node(
                    interface=LungSegmentationInference(),
                    name='{}_ls'.format(node_name))
                lung_seg.inputs.weights = self.network_weights

                workflow.connect(datasource, node_name, preproc, 'in_file')
                workflow.connect(preproc, 'tensor', lung_seg, 'tensor')
                workflow.connect(preproc, 'image_info', lung_seg, 'image_info')
                workflow.connect(lung_seg, 'segmented_lungs', datasink,
                     'results.subid.{0}.@{1}_segmented_lungs'.format(key, el))

        return workflow
