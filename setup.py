from pkg_resources import yield_lines
from setuptools import setup


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


with open('requirements.txt') as f:
    required = f.read().splitlines()

install_reqs = parse_requirements(required)
print(install_reqs)

setup(
    name='nexinfosys-backend',
    version='1.0',
    install_requires=install_reqs,
    packages=['backend', 'backend.common', 'backend.models', 'backend.models.experiments', 'backend.solving',
              'backend.solving.graph', 'backend.ie_exports', 'backend.ie_imports', 'backend.ie_imports.data_sources',
              'backend.ie_imports.experimental', 'backend.authentication', 'backend.model_services',
              'backend.restful_service', 'backend.restful_service.gunicorn', 'backend.restful_service.mod_wsgi',
              'backend.command_executors', 'backend.command_executors.misc', 'backend.command_executors.solving',
              'backend.command_executors.analysis', 'backend.command_executors.version2',
              'backend.command_executors.read_query', 'backend.command_executors.external_data',
              'backend.command_executors.specification', 'backend.command_generators',
              'backend.command_generators.spreadsheet_command_parsers',
              'backend.command_generators.spreadsheet_command_parsers.analysis',
              'backend.command_generators.spreadsheet_command_parsers.external_data',
              'backend.command_generators.spreadsheet_command_parsers.specification',
              'backend.command_generators.spreadsheet_command_parsers_v2', 'backend.magic_specific_integrations'],
    include_package_data=True,
    url='https://github.com/MAGIC-nexus/nis-backend',
    license='BSD-3',
    author='rnebot',
    author_email='rnebot@itccanarias.org',
    description='A package supporting MuSIASEM formalism and methodology'
)
