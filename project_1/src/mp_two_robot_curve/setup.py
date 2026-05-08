from setuptools import find_packages, setup

package_name = 'mp_two_robot_curve'

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
    maintainer='pedro',
    maintainer_email='pedro55562@ufmg.br',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'two_robots_curve = mp_two_robot_curve.two_robots_curve_node:main',
        'multi_robot_avoidance_node = mp_two_robot_curve.multi_robot_avoidance_node:main',
        ],
    },
)
