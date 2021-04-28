import os
import nibabel as nib
import nrrd
import numpy as np
from nipype.interfaces.base import (
    BaseInterface, TraitedSpec, Directory,
    BaseInterfaceInputSpec, traits, InputMultiPath)
from nipype.interfaces.base import isdefined
from skimage.transform import resize
from core.utils.filemanip import split_filename
from radiants.utils.dataloader import load_data_2D


class LungSegmentationPreprocInputSpec(BaseInterfaceInputSpec):
    
    in_file = traits.File(exists=True, desc=(
        'Cropped image (from PyCURT) to be preprocessed.'))
    new_spacing = InputMultiPath([0.35, 0.35, 0.35], desc=(
        'List of 3 Floats to be used to resample the input image.'))
    outdir = Directory('preproc', usedefault=True,
                        desc='Folder to store the preprocessing results.')


class LungSegmentationPreprocOutputSpec(TraitedSpec):
    
    preproc_image = traits.File(exists=True, desc='Preprocessed image')
    tensor = traits.Array(desc='Tensor to be fed to the network.')
    image_info = traits.Dict(desc='Dictionary with information about the image.')


class LungSegmentationPreproc(BaseInterface):
    
    input_spec = LungSegmentationPreprocInputSpec
    output_spec = LungSegmentationPreprocOutputSpec
    
    def _run_interface(self, runtime):
        
        new_spacing = self.inputs.new_spacing
        image = self.inputs.in_file
        outdir = os.path.abspath(self.inputs.outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.image_info = {}

        _, _, img_path, orig_size = self.resize_image(image, new_spacing=new_spacing,
                                                      outdir=outdir)
        self.image_info[img_path] = {}
        self.image_info[img_path]['orig_size'] = orig_size
        self.image_info[img_path]['orig_image'] = image
        self.create_tensors(img_path)
        self.img_path = img_path
        
        return runtime
    
    def resize_image(self, image, order=0, new_spacing=(0.1, 0.1, 0.1),
                     save2file=True, outdir=None):

        if outdir is None:
            outdir, fname, ext = split_filename(image)
        else:
            _, fname, ext = split_filename(image)
        outname = os.path.join(outdir, fname+'_resampled'+ext)
        if ext == '.nrrd':
            image, hd = nrrd.read(image)
            space_x = np.abs(hd['space directions'][0, 0])
            space_y = np.abs(hd['space directions'][1, 1])
            space_z = np.abs(hd['space directions'][2, 2])
        elif ext == '.nii.gz' or ext == '.nii':
            hd = nib.load(image).header
            affine = nib.load(image).affine
            image = nib.load(image).get_data()
            space_x, space_y, space_z = hd.get_zooms()
    
        resampling_factor = (new_spacing[0]/space_x, new_spacing[1]/space_y, new_spacing[2]/space_z)
        new_shape = (image.shape[0]//resampling_factor[0], image.shape[1]//resampling_factor[1],
                     image.shape[2]//resampling_factor[2])
        new_image = resize(image.astype(np.float64), new_shape, order=order, mode='edge',
                           cval=0, anti_aliasing=False)
        if save2file:
            if ext == '.nrrd':
                hd['sizes'] = np.array(new_image.shape)
                hd['space directions'][0, 0] = new_spacing[0]
                hd['space directions'][1, 1] = new_spacing[1]
                hd['space directions'][2, 2] = new_spacing[2]
                nrrd.write(outname, new_image, header=hd)
            elif ext == '.nii.gz' or ext == '.nii':
                im2save = nib.Nifti1Image(new_image, affine=affine)
                nib.save(im2save, outname)
        return new_image, tuple(map(int, new_shape)), outname, image.shape

    def create_tensors(self, image, patch_size=(96, 96)):
        "Function to create the 2D tensor from the 3D images"
        image_tensor = []
        im_base, im_name, ext = split_filename(image)
        im_path = os.path.join(im_base, im_name)
        if ext == '.nrrd':
            image, _ = nrrd.read(image)
        elif ext == '.nii.gz' or ext == '.nii':
            image = nib.load(image).get_fdata()
        im_size = image.shape[:2]
        for n_slice in range(image.shape[2]):
            im_array, info_dict = load_data_2D(
                '', '', array=image[:, :, n_slice], img_size=im_size,
                patch_size=patch_size, binarize=False, normalization=True,
                mb=[], prediction=True)
            for j in range(im_array.shape[0]):
                image_tensor.append(im_array[j, :])

        if info_dict is not None:
            im_name = im_path+ext
            self.image_info[im_name]['slices'] = n_slice+1
            for k in info_dict[0].keys():
                self.image_info[im_name][k] = info_dict[0][k]
        if image_tensor:
            self.image_tensor = (np.asarray(image_tensor).reshape(
                -1, im_array.shape[1], im_array.shape[2], 1))
        else:
            self.image_tensor = np.zeros()

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['preproc_image'] = self.img_path
        outputs['tensor'] = self.image_tensor
        outputs['image_info'] = self.image_info

        return outputs
