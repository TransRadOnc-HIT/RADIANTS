from setuptools import setup, find_packages

setup(name='radiants',
      version='1.0',
      description='Radiomics analysis for Radiotherpy studies',
      url='https://github.com/TransRadOnc-HIT/RADIANTS.git',
      python_requires='>=3.5',
      author='Francesco Sforazzini',
      author_email='f.sforazzini@dkfz.de',
      license='Apache 2.0',
      zip_safe=False,
      include_package_data=True,
      install_requires=[
      'matplotlib',
      'nibabel',
      'pandas',
      'pydicom',
      'pynrrd',
      'scikit-image',
      'opencv-python',
      'requests',
      'SimpleITK',
      'pyradiomics',
      'core',
      'nipype',
      'hd-bet',
      'nnunet',
      'h5py==2.10.0',
      'tensorflow==1.13.1',
      'keras==2.2.4'],
      dependency_links=['git+https://github.com/TransRadOnc-HIT/core.git#egg=core',
                        'git+https://github.com/MIC-DKFZ/HD-BET#egg=hd-bet',
			'git+https://github.com/TransRadOnc-HIT/nipype.git@c453eac5d7efdd4e19a9bcc8a7f3d800026cc125#egg=nipype-9876543210'],
      entry_points={
          'console_scripts': ['run_workflow = scripts.run_workflows:main']},
      packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ],
      scripts=['bash/predict_simple.py', 'bash/antsRegistrationSyN1.sh']
      )
