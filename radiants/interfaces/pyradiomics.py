from nipype.interfaces.base import (
    BaseInterface, TraitedSpec, Directory, File,
    traits, BaseInterfaceInputSpec, InputMultiPath)
from radiomics import featureextractor
import csv
import os.path as op


class FeatureExtractionInputSpec(BaseInterfaceInputSpec):
    
    parameter_file = File(exists=True, desc='File with all the '
                          'parameters to be used for feature extraction.')
    rois = InputMultiPath(File(exists=True), desc='List of roi to extract the features from.')
    input_image = File(exists=True, desc='Input image.')
    outname = traits.Str('Features_pyradiomics', usedefault=True,
                         desc='Output file name.')


class FeatureExtractionOutputSpec(TraitedSpec):
    
    feature_files = InputMultiPath(File(exists=True), desc='CSV file with the radiomics features.')

class FeatureExtraction(BaseInterface):

    input_spec = FeatureExtractionInputSpec
    output_spec = FeatureExtractionOutputSpec

    def _run_interface(self, runtime):
        
        rois = self.inputs.rois
        parameter_file = self.inputs.parameter_file
        image = self.inputs.input_image
        image_name = image.split('/')[-1].split('.nii')[0]
        out_basename = self.inputs.outname
        self.outfiles = []
        extractor = featureextractor.RadiomicsFeatureExtractor(parameter_file)

        for roi in rois:
            roi_name = roi.split('/')[-1].split('.nii')[0]
            keys = ['Subject', 'Mask']
            values = [image, roi]
            outname = op.abspath(out_basename+'_'+image_name+'_'+roi_name+'.csv')
            result = extractor.execute(image, roi)
            
            for k, value in result.items():
                keys.append(k)
                values.append(value)
            
            with open(outname, 'w') as outfile:
                csvwriter = csv.writer(outfile)
                csvwriter.writerow(keys)
                csvwriter.writerow(values)
            self.outfiles.append(outname)
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['feature_files'] = self.outfiles

        return outputs
