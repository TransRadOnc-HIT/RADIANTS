"Brain extraction workflow"
import nipype
from radiants.interfaces.mic import HDBet
from nipype.interfaces.fsl.utils import Reorient2Std
from core.workflows.base import BaseWorkflow
from nipype.interfaces.ants import N4BiasFieldCorrection
from pycurt.workflows.curation import DataCuration


TON4 = ['T1KM', 'T1']
TOBET = ['T1KM', 'T1', 'FLAIR', 'SWI', 'T2', 'ADC']


class BETWorkflow(DataCuration):
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        input_specs = {x: {'format': 'NIFTI_GZ', 'processed': DataCuration}
                       for x in TOBET}
        output_specs = {x+'_preproc': {'format': 'NIFTI_GZ', 'processed': BETWorkflow}
                        for x in TOBET}
        output_specs.update({x+'_preproc_mask': {'format': 'NIFTI_GZ',
                                                 'processed': BETWorkflow}
                            for x in TOBET})
        self.input_specs = input_specs
        self.output_specs.update(output_specs)
#         self.create_input_specs()

    def workflow_inputspecs(self):

        self.input_specs = {}
        self.input_specs['format'] = 'NIFTI_GZ'
    
    def workflow_outputspecs(self):

        self.output_specs = {}
        self.output_specs['format'] = 'NIFTI_GZ'
        self.output_specs['bet_image'] = '_preproc'
        self.output_specs['bet_image'] = '_preproc_mask'

    def workflow(self):

        self.datasource()

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
            if tobet[key]['ref'] is not None:
                files.append(tobet[key]['ref'])
            if tobet[key]['other'] is not None:
                files = files + tobet[key]['other']
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
    
    def workflow_setup(self):
        return BaseWorkflow.workflow_setup(self)
