from setuptools import find_packages, setup

package_name = 'intro_to_robotics_final_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chenjiaz',
    maintainer_email='chenjiaz2022@gmail.com',
    description='Package for Final Project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'final-project = intro_to_robotics_final_project.final_project:main',
            'gesture = intro_to_robotics_final_project.gesture:main',
            'gesture-training = intro_to_robotics_final_project.gestureTraining:main',
        ],
    },
)
