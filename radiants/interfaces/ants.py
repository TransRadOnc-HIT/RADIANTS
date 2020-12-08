from nipype.interfaces.base import (
    TraitedSpec, traits, File, CommandLineInputSpec, CommandLine)
from nipype.interfaces.ants.base import ANTSCommand, ANTSCommandInputSpec
import os.path as op
from nipype.interfaces.base import isdefined
from nipype.interfaces.base.traits_extension import InputMultiPath
import os


BASH_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                         os.pardir, os.pardir, 'bash'))


class AntsRegSynInputSpec(CommandLineInputSpec):

    _trans_types = ['t', 'r', 'a', 's', 'sr', 'so', 'b', 'br', 'bo']
    _precision_types = ['f', 'd']
    _interp_type = ['Linear', 'NearestNeighbor', 'BSpline',
                    'CosineWindowedSinc', 'WelchWindowedSinc',
                    'HammingWindowedSinc']
    input_file = File(mandatory=True, desc='existing input image',
                      argstr='-m %s', exists=True)
    ref_file = File(mandatory=True, desc='existing reference image',
                    argstr='-f %s', exists=True)
    num_dimensions = traits.Int(desc='number of dimension of the input file',
                                argstr='-d %s', mandatory=True)
    out_prefix = traits.Str(
        'antsreg', argstr='-o %s', usedefault=True, mandatory=True,
        desc='A prefix that is prepended to all output files')
    transformation = traits.Enum(
        *_trans_types, argstr='-t %s',
        desc='type of transformation. t:translation, r:rigid, a:rigid+affine,'
        's:rigid+affine+deformable Syn, sr:rigid+deformable Syn, so:'
        'deformable Syn, b:rigid+affine+deformable b-spline Syn, br:'
        'rigid+deformable b-spline Syn, bo:deformable b-spline Syn')
    num_threads = traits.Int(desc='number of threads', argstr='-n %s')
    radius = traits.Float(
        desc='radius for cross correlation metric used during SyN stage'
        ' (default = 4)', argstr='-r %f')
    spline_dist = traits.Float(
        desc='spline distance for deformable B-spline SyN transform'
        ' (default = 26)', argstr='-s %f')
    ref_mask = File(
        desc='mask for the fixed image space', exists=True, argstr='-x %s')
    precision_type = traits.List(
        traits.Enum(*_precision_types), argstr='-p %s', desc='precision type '
        '(default = d). f:float, d:double')
    use_histo_match = traits.Int(desc='use histogram matching (default = 0).'
                                 '0: False, 1:True', argstr='-j %s')
    interpolation = traits.Enum(
        *_interp_type, argstr='-l %s',
        desc='type of interpolation. Linear, NearestNeighbor, BSpline, '
             'CosineWindowedSinc, WelchWindowedSinc, HammingWindowedSinc')


class AntsRegSynOutputSpec(TraitedSpec):
    regmat = File(desc="Linear transformation matrix")
    reg_file = File(desc="Registered image")
    warp_file = File(desc="non-linear warp file")
    inv_warp = File(desc='invert of the warp file')


class AntsRegSyn(CommandLine):

#     _cmd = os.path.join(BASH_PATH, 'antsRegistrationSyN.sh')
    _cmd = 'antsRegistrationSyN1.sh'
    input_spec = AntsRegSynInputSpec
    output_spec = AntsRegSynOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['regmat'] = op.abspath(self.inputs.out_prefix +
                                       '0GenericAffine.mat')
        outputs['reg_file'] = op.abspath(self.inputs.out_prefix +
                                         'Warped.nii.gz')
        if isdefined(self.inputs.transformation and
                     (self.inputs.transformation != 's' or
                      self.inputs.transformation != 'a' or
                      self.inputs.transformation != 'b')):
            outputs['warp_file'] = op.abspath(
                self.inputs.out_prefix + '1Warp.nii.gz')
            outputs['inv_warp'] = op.abspath(
                self.inputs.out_prefix + '1InverseWarp.nii.gz')

        return outputs


class ResampleImageInputSpec(ANTSCommandInputSpec):

    dimensions = traits.Enum(3, 2, argstr='%d', usedefault=True, position=0,
                             desc='image dimension (2 or 3)')
    in_file = File(exists=True, mandatory=True, desc='Image to resample',
                   position=1, argstr='%s')
    out_file = File(exists=True, desc='Name of the output file.', name_source=['in_file'],
                    name_template='%s_resampled', keep_extension=True, position=2,
                    argstr='%s')
    new_size = traits.Str(argstr='%s', position=3, mandatory=True)
    mode = traits.Enum(0, 1, argstr='%d', position=4, usedefault=True,
                       desc='0 is size was specified, 1 if voxels were specified.')
    interpolation = traits.Enum(0, 1, 2, 3, 4, argstr='%d', position=5,
                                desc='Interpolation to use', usedefault=True)


class ResampleImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Resampled image')


class ResampleImage(ANTSCommand):
    
    _cmd = 'ResampleImage'
    input_spec = ResampleImageInputSpec
    output_spec = ResampleImageOutputSpec
