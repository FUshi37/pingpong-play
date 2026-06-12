from setuptools import find_packages, setup

package_name = 'pingpong_replay'

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
    maintainer='jacky',
    maintainer_email='yjk2261123518@163.com',
    description='Replay recorded RL right-arm trajectory on the real robot '
                'via /mx_mix_command (action/qdd/qvel/angle input modes).',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'replay_node = pingpong_replay.replay_node:main',
            'plot_analysis = pingpong_replay.plot_analysis:main',
        ],
    },
)
