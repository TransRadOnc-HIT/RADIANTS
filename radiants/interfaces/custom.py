import nibabel as nib
import nrrd
import os
import numpy as np
from nipype.interfaces.base import (
    BaseInterface, TraitedSpec,
    BaseInterfaceInputSpec, traits,
    Directory)
from core.utils.filemanip import split_filename
from skimage.transform import resize
from skimage.filters.thresholding import threshold_otsu
from radiants.utils.networks import unet_lung


class LungSegmentationInferenceInputSpec(BaseInterfaceInputSpec):
    
    tensor = traits.Array(desc='Tensor to be fed to the network.')
    image_info = traits.Dict(desc='Dictionary with information about the image.')
    weights = traits.List(desc='List of network weights.')
    outdir = Directory('segmented', usedefault=True,
                       desc='Folder to store the preprocessing results.')


class LungSegmentationInferenceOutputSpec(TraitedSpec):
    
    segmented_lungs = traits.File(exists=True, desc='Segmented lungs')


class LungSegmentationInference(BaseInterface):
    
    input_spec = LungSegmentationInferenceInputSpec
    output_spec = LungSegmentationInferenceOutputSpec
    
    def _run_interface(self, runtime):
        "Function to run the CNN inference"
        self.image_info = self.inputs.image_info
        outdir = os.path.abspath(self.inputs.outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        test_set = np.asarray(self.inputs.tensor)
        predictions = []
        model = unet_lung()
        for i, weight in enumerate(self.inputs.weights):
            print('Segmentation inference fold {}.'.format(i+1))
            model.load_weights(weight)
            predictions.append(model.predict(test_set))

        predictions = np.asarray(predictions, dtype=np.float16)
        self.prediction = np.mean(predictions, axis=0)
        self.segmentation = self.save_inference(outdir)

        return runtime

    def save_inference(self, outdir, binarize=True):
        "Function to save the segmented masks"
        prediction = self.prediction
        z0 = 0
        for i, image in enumerate(self.image_info):
            try:
                _, basename, ext = split_filename(image)
                patches = self.image_info[image]['patches']
                slices = self.image_info[image]['slices']
                resampled_image_dim = self.image_info[image]['image_dim']
                indexes = self.image_info[image]['indexes']
                deltas = self.image_info[image]['deltas']
                original_image_dim = self.image_info[image]['orig_size']
                im = prediction[z0:z0+(slices*patches), :, :, 0]
                final_prediction = self.inference_reshaping(
                    im, patches, slices, resampled_image_dim, indexes, deltas,
                    original_image_dim, binarize=binarize)
                outname = os.path.join(outdir, basename.split(
                    '_resampled')[0]+'_lung_segmented{}'.format(ext))
                reference = self.image_info[image]['orig_image']
                if ext == '.nrrd':
                    _, hd = nrrd.read(reference)
                    nrrd.write(outname, final_prediction, header=hd)
                elif ext == '.nii.gz' or ext == '.nii':
                    ref = nib.load(reference)
                    im2save = nib.Nifti1Image(final_prediction, affine=ref.affine)
                    nib.save(im2save, outname)
                z0 = z0+(slices*patches)
            except:
                continue
        return outname

    def inference_reshaping(self, generated_images, patches, slices,
                            dims, indexes, deltas, original_size,
                            binarize=False):
        "Function to reshape the predictions"
        if patches > 1:
            sl = 0
            final_image = np.zeros((slices, dims[0], dims[1], patches),
                                   dtype=np.float32)-2
            for n in range(0, generated_images.shape[0], patches):
                k = 0
                for j in indexes[1]:
                    for i in indexes[0]:
                        final_image[sl, i[0]:i[1], j[0]:j[1], k] = (
                            generated_images[n+k, deltas[0]:, deltas[1]:])
                        k += 1
                sl = sl + 1
            final_image[final_image==-2] = np.nan
            final_image = np.nanmean(final_image, axis=-1)
            final_image[np.isnan(final_image)] = 0
        else:
            final_image = generated_images[:, deltas[0]:, deltas[1]:]

        final_image = np.swapaxes(final_image, 0, 2)
        final_image = np.swapaxes(final_image, 0, 1)
        if final_image.shape != original_size:
            final_image = resize(final_image.astype(np.float64), original_size, order=0,
                                 mode='edge', cval=0, anti_aliasing=False)
        if binarize:
            final_image = self.binarization(final_image)

        return final_image

    @staticmethod
    def binarization(image):

        th = threshold_otsu(image)
        image[image>=th] = 1
        image[image!=1] = 0
    
        return image

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['segmented_lungs'] = self.segmentation

        return outputs
