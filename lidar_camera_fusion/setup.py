from setuptools import find_packages, setup

package_name = 'lidar_camera_fusion'

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
    maintainer='usman',
    maintainer_email='usman.a1282@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_reflectivity_processor=lidar_camera_fusion.tests.test_reflectivity_processor:main',
            'test_orb_extractor=lidar_camera_fusion.tests.test_orb_extractor:main',
            'visual_augmentor=lidar_camera_fusion.modules.visual_augmentor:main',
            'debug_visual_augmentor=lidar_camera_fusion.modules.debug_visual_augmentor:main',
            'visual_augmentor_debug=lidar_camera_fusion.modules.visual_augmentor_debug:main',
            'visual_augmentor_copy=lidar_camera_fusion.modules.visual_augmentor_copy:main',
            'lidar_overlay=lidar_camera_fusion.modules.lidar_overlay_test:main'
        ],
    },
)
