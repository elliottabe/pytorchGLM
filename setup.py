from setuptools import find_packages, setup

setup(
    name='pytorchGLM',
    version='0.1.0',    
    description='Code to train GLM models with pytorch',
    url='https://github.com/elliottabe/pytorchGLM',
    author='Elliott T. T. Abe',
    author_email='elliottabe@gmail.com',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['pandas',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
