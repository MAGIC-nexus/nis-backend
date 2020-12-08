# ONE TIME  -----------------------------------------------------------------------------
#
# pip install --upgrade setuptools wheel twine
#
# Create account:
# PyPI test: https://test.pypi.org/account/register/
# or PyPI  : https://pypi.org/account/register/
#
# EACH TIME -----------------------------------------------------------------------------
#
# Modify version code in "setup.py" (this file)
#
# Build (cd to directory where "setup.py" is)
# python3 setup.py sdist bdist_wheel
#
# Upload:
# PyPI test: twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*
# or PyPI  : twine upload --skip-existing dist/*
#
# INSTALL   ------------------------------------------------------------------------------
#
# PyPI test: pip install --index-url https://test.pypi.org/simple/ --upgrade nexinfosys
# PyPI     : pip install --upgrade nexinfosys
# No PyPI  : pip install -e <local path where "setup.py" (this file) is located>
#
# EXECUTION EXAMPLE ("gunicorn" must be installed: "pip install gunicorn")
#
# gunicorn --workers=1 --log-level=debug --timeout=2000 --bind 0.0.0.0:8081 nexinfosys.restful_service.service_main:app
#
from os import path

from setuptools import setup
from pkg_resources import yield_lines
# from distutils.extension import Extension
from Cython.Build import cythonize
# from Cython.Distutils import build_ext

package_name = 'nexinfosys'
version = '0.40'


def parse_requirements(strs):
    """Yield ``Requirement`` objects for each specification in `strs`

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(strs))

    ret = []
    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if ' #' in line:
            line = line[:line.find(' #')]
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith('\\'):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return
        ret.append(line)

    return ret


with open('requirements-as-package.txt') as f:
    required = f.read().splitlines()

install_reqs = parse_requirements(required)
print(install_reqs)

# ext_modules = [
#     Extension("helper_accel", ["nexinfosys/common/helper_accel.pyx"]),
#     Extension("parser_spreadsheet_utils_accel", ["nexinfosys/command_generators/parser_spreadsheet_utils_accel.pyx"])
# ]

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=package_name,
    version=version,
    install_requires=install_reqs,
    packages=['nexinfosys', 'nexinfosys.common', 'nexinfosys.models', 'nexinfosys.models.experiments', 'nexinfosys.solving',
              'nexinfosys.solving.graph', 'nexinfosys.ie_exports', 'nexinfosys.ie_imports', 'nexinfosys.ie_imports.data_sources',
              'nexinfosys.ie_imports.experimental', 'nexinfosys.authentication', 'nexinfosys.model_services',
              'nexinfosys.command_executors', 'nexinfosys.command_executors.misc', 'nexinfosys.command_executors.solving',
              'nexinfosys.command_executors.analysis', 'nexinfosys.command_executors.version2',
              'nexinfosys.command_executors.read_query', 'nexinfosys.command_executors.external_data',
              'nexinfosys.command_executors.specification', 'nexinfosys.command_generators',
              'nexinfosys.command_generators.spreadsheet_command_parsers',
              'nexinfosys.command_generators.spreadsheet_command_parsers.analysis',
              'nexinfosys.command_generators.spreadsheet_command_parsers.external_data',
              'nexinfosys.command_generators.spreadsheet_command_parsers.specification',
              # 'nexinfosys.magic_specific_integrations',
              'nexinfosys.command_generators.spreadsheet_command_parsers_v2',
              ],
    # See files to pack in "MANIFEST.in" file ("frontend" currently disabled)
    include_package_data=True,
    # cmdclass={'build_ext': build_ext},
    # ext_modules=cythonize(["nexinfosys/common/helper_accel.pyx", "nexinfosys/command_generators/parser_spreadsheet_utils_accel.pyx"], language_level="3"),
    url='https://github.com/MAGIC-nexus/nis-backend',
    license='BSD-3',
    author=['Rafael Nebot', 'Marco Galluzzi'],
    author_email='rnebot@itccanarias.org',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Formal and executable MuSIASEM multi-system Nexus models for Sustainable Development Analysis'
)
