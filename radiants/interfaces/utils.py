"Utility interfaces"
import shutil
import os
from nipype.interfaces.base import (
    BaseInterface, TraitedSpec, Directory,
    traits, BaseInterfaceInputSpec)
from core.utils.filemanip import split_filename


class NNUnetPreparationInputSpec(BaseInterfaceInputSpec):

    images = traits.List(mandatory=True, desc='List of images to be prepared before'
                         ' running the nnUNet inference.')


class NNUnetPreparationOutputSpec(TraitedSpec):

    output_folder = Directory(exists=True, desc='Output folder prepared for nnUNet.')


class NNUnetPreparation(BaseInterface):

    input_spec = NNUnetPreparationInputSpec
    output_spec = NNUnetPreparationOutputSpec

    def _run_interface(self, runtime):

        images = self.inputs.images
        if images:
            new_dir = os.path.abspath('data_prepared')
            os.mkdir(os.path.abspath('data_prepared'))
            for i, image in enumerate(images):
                _, _, ext = split_filename(image)
                shutil.copy2(image, os.path.join(
                    new_dir,'subject1_{}'.format(str(i).zfill(4))+ext))
        else:
            raise Exception('No images provided!Please check.')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_folder'] = os.path.abspath('data_prepared')

        return outputs
