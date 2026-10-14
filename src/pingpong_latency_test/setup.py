from setuptools import setup
import os
from glob import glob

package_name = 'pingpong_latency_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hanlin',
    maintainer_email='hanlin@example.com',
    description='Latency testing tools for ping-pong ball detection system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flash_detector = pingpong_latency_test.flash_detector_node:main',
            'local_flasher = pingpong_latency_test.local_flasher:main',
            'pipeline_profiler = pingpong_latency_test.pipeline_profiler:main',
            'latency_analyzer = pingpong_latency_test.latency_analyzer:main',
            'roi_calibrator = pingpong_latency_test.roi_calibrator:main',
            'clock_sync = pingpong_latency_test.clock_sync:main',
        ],
    },
)
