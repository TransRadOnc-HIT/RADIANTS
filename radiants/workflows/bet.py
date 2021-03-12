"Brain extraction workflow"
import nipype
from radiants.interfaces.mic import HDBet
from nipype.interfaces.fsl.utils import Reorient2Std
from core.workflows.base import BaseWorkflow
from nipype.interfaces.ants import N4BiasFieldCorrection


TON4 = ['T1KM', 'T1']
TOBET = ['T1KM', 'T1', 'FLAIR', 'SWI', 'T2', 'ADC']


class BETWorkflow(BaseWorkflow):

    @staticmethod
    def workflow_inputspecs(additional_inputs=None):

        input_specs = {}
        input_specs['format'] = '.nii.gz'
        input_specs['inputs'] = {
            '': {'mandatory': True, 'format': '.nii.gz', 'dependency': None,
                 'possible_sequences': TOBET, 'multiplicity': 'all', 'composite': None}}
        input_specs['dependencies'] = {}
        input_specs['suffix'] = ['']
        input_specs['prefix'] = []
        input_specs['data_formats'] = {'': '.nii.gz'}
        input_specs['additional_inputs'] = additional_inputs

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        dict_outputs = {'_preproc': {
            'possible_sequences': TOBET, 'format': '.nii.gz',
            'multiplicity': 'all', 'composite': None}}
        output_specs['outputs'] = dict_outputs

        return output_specs

    def workflow(self):

#         self.datasource()

        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id

        tobet = {**dict_sequences['MR-RT'], **dict_sequences['OT']}
        workflow = nipype.Workflow('brain_extraction_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]
        substitutions += [('_preproc_corrected.', '_preproc.')]
        datasink.inputs.substitutions = substitutions

        for key in tobet:
            files = []
#             if tobet[key]['ref'] is not None:
#                 files.append(tobet[key]['ref'])
            if tobet[key]['scans'] is not None:
                files = files + tobet[key]['scans']
            for el in files:
                el = el.strip(self.extention)
                node_name = '{0}_{1}'.format(key, el)
                bet = nipype.Node(
                    interface=HDBet(),
                    name='{}_bet'.format(node_name), serial=True)
                bet.inputs.save_mask = 1
                bet.inputs.out_file = '{}_preproc'.format(el)
                reorient = nipype.Node(
                    interface=Reorient2Std(),
                    name='{}_reorient'.format(node_name))
                if el in TON4:
                    n4 = nipype.Node(
                        interface=N4BiasFieldCorrection(),
                        name='{}_n4'.format(node_name))
                    workflow.connect(bet, 'out_file', n4, 'input_image')
                    workflow.connect(bet, 'out_mask', n4, 'mask_image')
                    workflow.connect(n4, 'output_image', datasink,
                     'results.subid.{0}.@{1}_preproc'.format(key, el))
                else:
                    workflow.connect(bet, 'out_file', datasink,
                     'results.subid.{0}.@{1}_preproc'.format(key, el))
                workflow.connect(bet, 'out_mask', datasink,
                     'results.subid.{0}.@{1}_preproc_mask'.format(key, el))
                workflow.connect(reorient, 'out_file', bet, 'input_file')
                workflow.connect(datasource, node_name, reorient, 'in_file')

        return workflow

    def workflow_setup(self, create_database=True, dict_sequences=None, **kwargs):

        return BaseWorkflow.workflow_setup(
            self, create_database=create_database,
            dict_sequences=dict_sequences, **kwargs)
