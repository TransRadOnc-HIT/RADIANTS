from nipype.interfaces.base import (
    TraitedSpec, traits, File, CommandLineInputSpec, CommandLine,
    BaseInterfaceInputSpec, Directory, BaseInterface)
import os
import glob
from nipype.interfaces.base import isdefined
from core.utils.filemanip import split_filename
try:
    from nnunet.inference.predict import predict_from_folder
    import torch
except ModuleNotFoundError:
    print('Cannot find import nnUNet, no brain extraction or tumor '
          'segmentation can be performed!')


BASH_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                         os.pardir, os.pardir, 'bash'))


class HDBetInputSpec(CommandLineInputSpec):
    
    _mode_types = ['accurate', 'fast']
    input_file = File(mandatory=True, desc='existing input image',
                      argstr='-i %s', exists=True)
    out_file = traits.Str(argstr='-o %s', desc='output file (or folder) name.')
    mode = traits.Enum(*_mode_types, argstr='-mode %s',
                       desc='Fast will use only one set of parameters whereas '
                             'accurate will use the five sets of parameters '
                             'that resulted from our cross-validation as an '
                             'ensemble. Default: accurate')
    device = traits.Str(argstr='-device %s',
                        desc='Used to set on which device the prediction will run.'
                             'Must be either int or str. Use int for GPU id or "cpu" '
                             'to run on CPU. When using CPU you should consider '
                             'disabling tta. Default for -device is: 0')
    tta = traits.Int(argstr='-tta %i',
                     desc='Whether to use test time data augmentation '
                          '(mirroring). 1= True, 0=False. Disable this if you are '
                          'using CPU to speed things up! Default: 1')
    post_processing = traits.Int(argstr='-pp %i',
                                 desc='Set to 0 to disabe postprocessing '
                                      '(remove all but the largest connected '
                                      'component in the prediction. Default: 1')
    save_mask = traits.Int(argstr='-s %i',
                           desc='If set to 0 the segmentation mask will not be saved')
    overwrite_existing = traits.Int(argstr='--overwrite_existing %i',
                                    desc='Set this to 0 if you do not want to'
                                    ' overwrite existing predictions')


class HDBetOutputSpec(TraitedSpec):

    out_file = File(desc='Brain extracted image.')
    out_mask = File(desc='Brain mask.')


class HDBet(CommandLine):

    _cmd = 'hd-bet'
    input_spec = HDBetInputSpec
    output_spec = HDBetOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename('out_file')
        if isdefined(self.inputs.save_mask and self.inputs.save_mask != 0):
            outputs['out_mask'] = self._gen_outfilename('out_mask')

        return outputs

    def _gen_outfilename(self, name):
        if name == 'out_file':
            out_file = self.inputs.out_file
            if isdefined(out_file) and isdefined(self.inputs.input_file):
                _, _, ext = split_filename(self.inputs.input_file)
                out_file = self.inputs.out_file+ext
            if not isdefined(out_file) and isdefined(self.inputs.input_file):
                pth, fname, ext = split_filename(self.inputs.input_file)
                print(pth, fname, ext)
                out_file = os.path.join(pth, fname+'_bet'+ext)
        elif name == 'out_mask':
            out_file = self.inputs.out_file
            if isdefined(out_file) and isdefined(self.inputs.input_file):
                _, _, ext = split_filename(self.inputs.input_file)
                out_file = self.inputs.out_file+'_mask'+ext
#             if isdefined(out_file):
#                 pth, fname, ext = split_filename(out_file)
#                 out_file = os.path.join(pth, fname+'_bet_mask'+ext)
            elif not isdefined(out_file) and isdefined(self.inputs.input_file):
                pth, fname, ext = split_filename(self.inputs.input_file)
                print(pth, fname, ext)
                out_file = os.path.join(pth, fname+'_bet_mask'+ext)

        return os.path.abspath(out_file)

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename('out_file')
        elif name == 'out_mask':
            return self._gen_outfilename('out_mask')
        return None


class HDGlioPredictInputSpec(CommandLineInputSpec):

    t1 = traits.File(mandatory=True, exists=True, argstr='-t1 %s',
                     desc='T1 weighted image')
    ct1 = traits.File(mandatory=True, exists=True, argstr='-t1c %s',
                      desc='T1 weighted image')
    t2 = traits.File(mandatory=True, exists=True, argstr='-t2 %s',
                     desc='T1 weighted image')
    flair = traits.File(mandatory=True, exists=True, argstr='-flair %s',
                        desc='T1 weighted image')
    out_file = traits.Str(argstr='-o %s', desc='output file (or folder) name.')


class HDGlioPredictOutputSpec(TraitedSpec):

    out_file = File(desc='Brain extracted image.')


class HDGlioPredict(CommandLine):

    _cmd = 'hd_glio_predict'
    input_spec = HDGlioPredictInputSpec
    output_spec = HDGlioPredictOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()

        return outputs

    def _gen_outfilename(self):

        out_file = self.inputs.out_file
        if isdefined(out_file) and isdefined(self.inputs.t1):
            _, _, ext = split_filename(self.inputs.t1)
            out_file = self.inputs.out_file+ext
        if not isdefined(out_file) and isdefined(self.inputs.t1):
            pth, _, ext = split_filename(self.inputs.t1)
            print(pth, ext)
            out_file = os.path.join(pth, 'segmentation'+ext)

        return os.path.abspath(out_file)


class NNUnetInferenceInputSpec(CommandLineInputSpec):

    input_folder = Directory(exist=True, mandatory=True,
                             desc='Input directory', argstr='-i %s')
    output_folder = Directory(genfile=True,
                                desc='Output directory', argstr='-o %s')
    model_folder = Directory(mandatory=True, exist=True,
                             desc='Folder with the results of the nnUnet'
                             'training.', argstr='-m %s')
    prefix = traits.Str()


class NNUnetInferenceOutputSpec(TraitedSpec):

    output_folder = Directory(exist=True, desc='Output directory')
    output_file = File(exists=True, desc='First nifti file inside the'
                       ' output folder.')


class NNUnetInference(CommandLine):

    _cmd = 'predict_simple.py'
    input_spec = NNUnetInferenceInputSpec
    output_spec = NNUnetInferenceOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        output_folder = self._gen_outfilename()
        outputs['output_folder'] = output_folder
        outputs['output_file'] = sorted(glob.glob(
            os.path.join(output_folder, '*.nii.gz')))[0]

        return outputs
    
    def _gen_outfilename(self):
        output_folder = self.inputs.output_folder
        if not isdefined(output_folder) and isdefined(self.inputs.input_folder):
            basepath = '/'.join(self.inputs.input_folder.split('/')[:-1])
            outname = 'nnunet_inference_{}'.format(self.inputs.prefix)
            output_folder = os.path.join(basepath, outname)
#             output_folder = 'nnunet_inference'
        return os.path.abspath(output_folder)

    def _gen_filename(self, name):
        if name == 'output_folder':
            return self._gen_outfilename()
        return None

