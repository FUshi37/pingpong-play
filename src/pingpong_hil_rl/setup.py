from setuptools import find_packages, setup

package_name = "pingpong_hil_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/hil_flashsac.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "gymnasium", "mujoco", "omegaconf", "tensorboard"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "hil_live_probe = pingpong_hil_rl.live_probe:main",
            "hil_train_flashsac = pingpong_hil_rl.train_flashsac:main",
            "hil_replay_equivalence = pingpong_hil_rl.replay_equivalence:main",
        ]
    },
)
