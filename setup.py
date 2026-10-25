from setuptools import find_packages, setup

package_name = 'arena_assets'

setup(
    name=package_name,
    packages=find_packages(where='src', include=[f'{package_name}*']),
    package_dir={'': 'src'},
    package_data={package_name: ['semantic/asa.yaml']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    extras_require={
        'test': ['pytest>=8'],
    },
    zip_safe=True,
    maintainer='Vova Shcherbyna',
    maintainer_email='dev@voshch.dev',
    description='Build, query, and manage 3D asset databases',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            f'{package_name} = {package_name}.__main__:app',
        ],
    },
)
