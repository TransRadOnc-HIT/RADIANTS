"Registration workflows"
import nipype
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.utility import Merge
from radiants.interfaces.ants import AntsRegSyn
from core.workflows.base import BaseWorkflow
from radiants.workflows.bet import BETWorkflow


TOREG = ['T1KM', 'T1', 'FLAIR', 'SWI', 'T2', 'ADC']
POSSIBLE_REF = ['T1KM', 'T1'] # order is important! 


class RegistrationWorkflow(BaseWorkflow):

    @staticmethod
    def workflow_inputspecs():

        input_specs = {}

        input_specs['format'] = '.nii.gz'
        dependencies = {}
        dependencies[BETWorkflow] = [['_preproc', '.nii.gz', TOREG, 'all']]
        formats = {}
        for k in dependencies:
            for entry in dependencies[k]:
                formats[entry[0]] = entry[1]
        input_specs['data_formats'] = formats
        input_specs['suffix'] = ['_preproc']
        input_specs['prefix'] = []
        input_specs['dependencies'] = dependencies

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        output_specs['format'] = '.nii.gz'
        output_specs['suffix'] = ['_reg', '_reg2MR_RT', '_reg2RTCT']
        output_specs['prefix'] = []

        return output_specs
    
    def datasource(self, create_database=True,
                   dict_sequences=None, check_dependencies=True):
        BaseWorkflow.datasource(self, create_database=create_database,
                                dict_sequences=dict_sequences,
                                check_dependencies=check_dependencies,
                                possible_sequences=TOREG)

    def workflow(self):

        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id

        toreg = {**dict_sequences['MR-RT'], **dict_sequences['OT']}
        workflow = nipype.Workflow('registration_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]

        mr_rt_ref = None
        rtct = None

        if dict_sequences['MR-RT'] and self.normilize_mr_rt:
            ref_session = list(dict_sequences['MR-RT'].keys())[0]
            ref_scans = dict_sequences['MR-RT'][ref_session]['scans']
            for pr in POSSIBLE_REF:
                for scan in ref_scans:
                    if pr in scan.split('_')[0]:
                        mr_rt_ref = '{0}_{1}_preproc'.format(
                            ref_session, scan.split('_')[0])
                        mr_rt_ref_name = '{}_preproc'.format(
                            scan.split('_')[0])
                        break
                else:
                    continue
                break
        if dict_sequences['RT'] and self.normilize_rtct:
            rt_session = list(dict_sequences['RT'].keys())[0]
            ct_name = dict_sequences['RT'][rt_session]['rtct']
            if ct_name is not None and mr_rt_ref is not None:
                rtct = '{0}_rtct'.format(rt_session, ct_name)
                reg_mr2ct = nipype.Node(interface=AntsRegSyn(),
                                        name='{}_lin_reg'.format(rt_session))
                reg_mr2ct.inputs.transformation = 'r'
                reg_mr2ct.inputs.num_dimensions = 3
                reg_mr2ct.inputs.num_threads = 4
                reg_mr2ct.inputs.out_prefix = '{}_reg2RTCT'.format(mr_rt_ref_name)
                reg_mr2ct.inputs.interpolation = 'BSpline'
                workflow.connect(datasource, mr_rt_ref, reg_mr2ct, 'input_file')
                workflow.connect(datasource, rtct, reg_mr2ct, 'ref_file')
                workflow.connect(
                    reg_mr2ct, 'regmat', datasink,
                    'results.subid.{0}.@{1}_reg2RTCT_mat'.format(
                    ref_session, mr_rt_ref_name))
                workflow.connect(
                    reg_mr2ct, 'reg_file', datasink,
                    'results.subid.{0}.@{1}_reg2RTCT'.format(
                    ref_session, mr_rt_ref_name))
                substitutions += [('{}_reg2RTCTWarped.nii.gz'.format(mr_rt_ref_name),
                                   '{}_reg2RTCT.nii.gz'.format(mr_rt_ref_name))]
                substitutions += [('{}_reg2RTCT0GenericAffine.mat'.format(mr_rt_ref_name),
                                   '{}_reg2RTCT_linear_mat.mat'.format(mr_rt_ref_name))]

        for key in toreg:
            session = toreg[key]
            if session['scans'] is not None:
                scans = session['scans']
                scans = [x for x in scans if 'mask' not in x]
                ref = None
                for pr in POSSIBLE_REF:
                    for scan in scans:
                        if pr in scan:
                            ref = '{0}_{1}_preproc'.format(
                                key, scan.split('_')[0])
                            scans.remove('{}_preproc'.format(
                                scan.split('_')[0]))
                            ref_name = scan.split('_')[0]
                            workflow.connect(
                                datasource, ref, datasink,
                                'results.subid.{0}.@{1}_reg'.format(
                                    key, ref_name))
                            substitutions += [
                                ('{}_preproc'.format(scan.split('_')[0]),
                                 '{}_reg'.format(scan.split('_')[0]))]
                            break
                    else:
                        continue
                    break
                if ref is not None:
                    if mr_rt_ref is not None and key != ref_session:
                        reg_mr_rt = nipype.Node(
                            interface=AntsRegSyn(),
                            name='{}_def_reg'.format(key))
                        reg_mr_rt.inputs.transformation = 's'
                        reg_mr_rt.inputs.num_dimensions = 3
                        reg_mr_rt.inputs.num_threads = 6
                        reg_mr_rt.inputs.out_prefix = '{}_reg2MR_RT'.format(ref_name)
                        workflow.connect(datasource, ref, reg_mr_rt, 'input_file')
                        workflow.connect(datasource, mr_rt_ref, reg_mr_rt, 'ref_file')
                        workflow.connect(
                            reg_mr_rt, 'regmat', datasink,
                            'results.subid.{0}.@{1}_reg2MR_RT_linear_mat'.format(
                                    key, ref_name))
                        workflow.connect(
                            reg_mr_rt, 'reg_file', datasink,
                            'results.subid.{0}.@{1}_reg2MR_RT'.format(
                                    key, ref_name))
                        workflow.connect(
                            reg_mr_rt, 'warp_file', datasink,
                            'results.subid.{0}.@{1}_reg2MR_RT_warp'.format(
                                    key, ref_name))
                        substitutions += [('{}_reg2MR_RT0GenericAffine.mat'.format(ref_name),
                                           '{}_reg2MR_RT_linear_mat.mat'.format(ref_name))]
                        substitutions += [('{}_reg2MR_RT1Warp.nii.gz'.format(ref_name),
                                           '{}_reg2MR_RT_warp.nii.gz'.format(ref_name))]
                        substitutions += [('{}_reg2MR_RTWarped.nii.gz'.format(ref_name),
                                           '{}_reg2MR_RT.nii.gz'.format(ref_name))]
                    if rtct is not None and key != ref_session:
                        apply_ts_rt_ref = nipype.Node(
                            interface=ApplyTransforms(),
                            name='{}_norm2RT'.format(ref_name))
                        apply_ts_rt_ref.inputs.output_image = (
                            '{}_reg2RTCT.nii.gz'.format(ref_name))
                        workflow.connect(datasource, ref, apply_ts_rt_ref,
                                         'input_image')
                        workflow.connect(datasource, rtct, apply_ts_rt_ref,
                                         'reference_image')
                        workflow.connect(
                            apply_ts_rt_ref, 'output_image', datasink,
                            'results.subid.{0}.@{1}_reg2RTCT'.format(
                                key, ref_name))
                        merge_rt_ref = nipype.Node(interface=Merge(4),
                                        name='{}_merge_rt'.format(ref_name))
                        merge_rt_ref.inputs.ravel_inputs = True
                        workflow.connect(reg_mr2ct, 'regmat', merge_rt_ref, 'in1')
                        workflow.connect(reg_mr_rt, 'regmat', merge_rt_ref, 'in3')
                        workflow.connect(reg_mr_rt, 'warp_file', merge_rt_ref, 'in2')
                        workflow.connect(merge_rt_ref, 'out', apply_ts_rt_ref, 'transforms')

                    for el in scans:
                        el = el.strip(self.extention)
                        el_name = el.split('_')[0]
                        node_name = '{0}_{1}'.format(key, el)
                        reg = nipype.Node(interface=AntsRegSyn(),
                                          name='{}_lin_reg'.format(node_name))
                        reg.inputs.transformation = 'r'
                        reg.inputs.num_dimensions = 3
                        reg.inputs.num_threads = 4
                        reg.inputs.interpolation = 'BSpline'
                        reg.inputs.out_prefix = '{}_reg'.format(el_name)
                        workflow.connect(datasource, node_name, reg, 'input_file')
                        workflow.connect(datasource, ref, reg, 'ref_file')
                        workflow.connect(
                            reg, 'reg_file', datasink,
                            'results.subid.{0}.@{1}_reg'.format(key, el_name))
                        workflow.connect(
                            reg, 'regmat', datasink,
                            'results.subid.{0}.@{1}_regmat'.format(key, el_name))
                        substitutions += [('{}_regWarped.nii.gz'.format(el_name),
                                           '{}_reg.nii.gz'.format(el_name))]
                        substitutions += [('{}_reg0GenericAffine.mat'.format(el_name),
                                           '{}_linear_regmat.mat'.format(el_name))]
                        if mr_rt_ref is not None and key != ref_session:
                            merge = nipype.Node(interface=Merge(3),
                                                name='{}_merge_MR_RT'.format(node_name))
                            merge.inputs.ravel_inputs = True
                            workflow.connect(reg, 'regmat', merge, 'in3')
                            workflow.connect(reg_mr_rt, 'regmat', merge, 'in2')
                            workflow.connect(reg_mr_rt, 'warp_file', merge, 'in1')
                            apply_ts = nipype.Node(interface=ApplyTransforms(),
                                                   name='{}_norm2MR_RT'.format(node_name))
                            apply_ts.inputs.output_image = '{}_reg2MR_RT.nii.gz'.format(el_name)
                            workflow.connect(merge, 'out', apply_ts, 'transforms')
                            workflow.connect(datasource, node_name, apply_ts,
                                             'input_image')
                            workflow.connect(datasource, mr_rt_ref, apply_ts,
                                             'reference_image')
                            workflow.connect(
                                apply_ts, 'output_image', datasink,
                                'results.subid.{0}.@{1}_reg2MR_RT'.format(
                                    key, el_name))
                        if rtct is not None:
                            apply_ts_rt = nipype.Node(interface=ApplyTransforms(),
                                                   name='{}_norm2RT'.format(node_name))
                            apply_ts_rt.inputs.output_image = '{}_reg2RTCT.nii.gz'.format(el_name)
                            workflow.connect(datasource, node_name, apply_ts_rt,
                                             'input_image')
                            workflow.connect(datasource, rtct, apply_ts_rt,
                                             'reference_image')
                            workflow.connect(
                                apply_ts_rt, 'output_image', datasink,
                                'results.subid.{0}.@{1}_reg2RTCT'.format(
                                    key, el_name))
                            if key != ref_session:
                                merge_rt = nipype.Node(interface=Merge(4),
                                                name='{}_merge_rt'.format(node_name))
                                merge_rt.inputs.ravel_inputs = True
                                workflow.connect(reg_mr2ct, 'regmat', merge_rt, 'in1')
                                workflow.connect(reg, 'regmat', merge_rt, 'in4')
                                workflow.connect(reg_mr_rt, 'regmat', merge_rt, 'in3')
                                workflow.connect(reg_mr_rt, 'warp_file', merge_rt, 'in2')
                                workflow.connect(merge_rt, 'out', apply_ts_rt, 'transforms')
                            else:
                                merge_rt = nipype.Node(interface=Merge(2),
                                                name='{}_merge_rt'.format(node_name))
                                merge_rt.inputs.ravel_inputs = True
                                workflow.connect(reg_mr2ct, 'regmat', merge_rt, 'in1')
                                workflow.connect(reg, 'regmat', merge_rt, 'in2')
                                workflow.connect(merge_rt, 'out', apply_ts_rt, 'transforms')
                                

        datasink.inputs.substitutions = substitutions

        return workflow
