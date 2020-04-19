from os.path import dirname

import pandasdmx

# FIRST, ensure "pip freeze" output matches versions in "requirements.txt"

# Several installers supported
# * pip
# * pyinstaller
# * cx_freeze

# TODO
# Cython files:
# * How?: Just invoke "cythonize -i **/*.pyx" before the "pyinstaller" command
#
# configuration file. Generate one dynamically if not specified, with default data folder for windows, default DATA directory for Linux
#  directory for database files
#  directory for sessions
#  directory for datasets: FAO and FADN
#  directory for temporary: Eurostat and OECD
# credentials files for Google Drive
#
#
# WINDOWS...
# waitress (optional, start with default server)


def elaborate_pyinstaller_command_line(system, output_name, output_type, config_file_name):
    """
    To be executed in the target system

    :param system: "linux", "windows"
    :param output_name: None, "nis-backend", ...
    :param output_type: "directory" or "single-file"
    :param config_file_name: configuration for runtime
    :return: string with the command to execute
    """

    if not system:
        import platform
        system = "linux" if platform.system() == "Linux" else "windows" if platform.system() == "Windows" else "macosx"

    lsep = "\\" if system in ["linux", "macosx"] else "^"
    sep = "/" if system in ["linux", "macosx"] else "\\"
    set_var = "export" if system in ["linux", "macosx"] else "set"
    two_paths_sep = ":" if system in ["linux", "macosx"] else ";"
    pandasdmx_path = dirname(pandasdmx.__file__)
    if not output_name:
        output_name = "service_main"
    if output_type == "onedir" or output_type == "onefile":
        output_type_option = "--" + output_type
    else:
        output_type_option = ""

    cmd = f"""
<In Windows, install Visual Studio Community>
<Install a Python environment (tested with Anaconda)>
<In Windows, execute the following commands in a special BASH shell, available after installing "gitforwindows">
git clone https://github.com/MAGIC-nexus/nis-backend
<In Windows, execute the following commands in a special BASH shell, available after installing "gitforwindows". Run as Administrator>
<Execute the following two "pip" lines IF it is the first time OR if there is an update in requirements.txt or in "pyinstaller">
<In Windows: conda install -c conda-forge fiona> (the first time)>
pip install -r requirements.txt
pip install pyinstaller

<cd to the "nis-backend" directory and execute:>
cythonize -i **/*.pyx

<In Windows, execute in a normal CMD.EXE shell (not PowerShell):>

<Pack the following command in a shell file (.bat for Windows, .sh for Linux and Mac OS)>

pyinstaller -n {output_name} {output_type_option} {lsep}
--exclude-module matplotlib --exclude-module _tkinter --exclude-module PyQt4 --exclude-module PyQt5 --exclude-module IPython {lsep}
--hidden-import nexinfosys.command_executors.version2 {lsep}
--hidden-import nexinfosys.command_executors.version2.attribute_sets_command {lsep}
--hidden-import nexinfosys.command_executors.version2.dataset_data_command {lsep}
--hidden-import nexinfosys.command_executors.version2.dataset_definition_command {lsep}
--hidden-import nexinfosys.command_executors.version2.dataset_query_command {lsep}
--hidden-import nexinfosys.command_executors.version2.hierarchy_categories_command {lsep}
--hidden-import nexinfosys.command_executors.version2.hierarchy_mapping_command {lsep}
--hidden-import nexinfosys.command_executors.version2.interfaces_command {lsep}
--hidden-import nexinfosys.command_executors.version2.interface_types_command {lsep}
--hidden-import nexinfosys.command_executors.version2.matrix_indicators_command {lsep}
--hidden-import nexinfosys.command_executors.version2.nested_commands_command {lsep}
--hidden-import nexinfosys.command_executors.version2.pedigree_matrices_command {lsep}
--hidden-import nexinfosys.command_executors.version2.problem_statement_command {lsep}
--hidden-import nexinfosys.command_executors.version2.processor_scalings_command {lsep}
--hidden-import nexinfosys.command_executors.version2.processors_command {lsep}
--hidden-import nexinfosys.command_executors.version2.references_v2_command {lsep}
--hidden-import nexinfosys.command_executors.version2.relationships_command {lsep}
--hidden-import nexinfosys.command_executors.version2.scalar_indicator_benchmarks_command {lsep}
--hidden-import nexinfosys.command_executors.version2.scalar_indicators_command {lsep}
--hidden-import nexinfosys.command_executors.version2.scale_conversion_v2_command {lsep}
--hidden-import nexinfosys.command_executors.specification {lsep}
--hidden-import nexinfosys.command_executors.specification.data_input_command {lsep}
--hidden-import nexinfosys.command_executors.specification.hierarchy_command {lsep}
--hidden-import nexinfosys.command_executors.specification.metadata_command {lsep}
--hidden-import nexinfosys.command_executors.specification.pedigree_matrix_command {lsep}
--hidden-import nexinfosys.command_executors.specification.references_command {lsep}
--hidden-import nexinfosys.command_executors.specification.scale_conversion_command {lsep}
--hidden-import nexinfosys.command_executors.specification.structure_command {lsep}
--hidden-import nexinfosys.command_executors.specification.upscale_command {lsep}
--hidden-import nexinfosys.command_executors.external_data {lsep}
--hidden-import nexinfosys.command_executors.external_data.etl_external_dataset_command {lsep}
--hidden-import nexinfosys.command_executors.external_data.mapping_command {lsep}
--hidden-import nexinfosys.command_executors.external_data.parameters_command {lsep}
--add-data nexinfosys{sep}frontend{sep}{two_paths_sep}.{sep}nexinfosys{sep}frontend {lsep}
--add-data {pandasdmx_path}{sep}agencies.json{two_paths_sep}pandasdmx {lsep}
nexinfosys/restful_service/service_main.py

<Modify config file. Different in Windows and Linux>

{set_var} MAGIC_NIS_SERVICE_CONFIG_FILE={config_file_name.replace("/", sep)} {lsep}
dist{sep}{output_name}{sep+output_name if output_type == "onedir" else ""}
"""
    return cmd


if __name__ == '__main__':
    cfg_file = "/home/rnebot/Dropbox/nis-backend-config/nis_local.conf"
    system_type = "windows"  # "linux", "windows", "macosx", None (autodetect)
    dist_type = "onefile"  # "onedir", "onefile"
    print(elaborate_pyinstaller_command_line(system_type, "nis-backend", dist_type, cfg_file))
