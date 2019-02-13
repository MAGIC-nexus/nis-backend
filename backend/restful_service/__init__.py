from flask import Flask

import backend
from backend.ie_imports.data_source_manager import DataSourceManager
from backend.ie_imports.data_sources.eurostat_bulk import Eurostat
from backend.ie_imports.data_sources.fadn import FADN
from backend.ie_imports.data_sources.faostat import FAOSTAT
from backend.ie_imports.data_sources.oecd import OECD
from backend.models.musiasem_methodology_support import DBSession

nis_api_base = "/nis_api"  # Base for all RESTful calls
nis_client_base = "/nis_client"  # Base for the Angular2 client
nis_external_client_base = "/nis_external"  # Base for the Angular2 client called from outside

app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = '/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Initialize configuration
try:
    app.config.from_envvar('MAGIC_NIS_SERVICE_CONFIG_FILE')
except Exception:
    print("MAGIC_NIS_SERVICE_CONFIG_FILE environment variable not defined!")

tm_permissions = {  # PermissionType
    "f19ad19f-0a74-44e8-bd4e-4762404a35aa": "read",
    "04cac7ca-a90b-4d12-a966-d8b0c77fca70": "annotate",
    "d0924822-32fa-4456-8143-0fd48da33fd7": "contribute",
    "83d837ab-01b2-4260-821b-8c4a3c52e9ab": "share",
    "d3137471-84a0-4bcf-8dd8-16387ea46a30": "delete"
}

tm_default_users = {  # Users
    "f3848599-4aa3-4964-b7e1-415d478560be": "admin",
    "2a512320-cef7-41c6-a141-8380d900761b": "_anonymous"
}

tm_object_types = {  # ObjectType
    "d6649a54-3538-4ee6-a4fc-8a67b74ed21f": "processor",
    "7eca7cfb-3eea-475f-9950-2d1093099ccb": "flow/fund",
    "3cc2582b-5142-4c8f-b484-a4693ba267cf": "flow",
    "c18ca5ab-471d-4ab4-9c0d-9563078a1c18": "fund",
    "5c9ef31b-399f-4007-824f-f2251e29bfdc": "ff_in_processor",
    "6e187787-adf2-4bdf-a0af-ab2801a6be42": "hierarchy",
    "91a7e3c2-115d-4166-a893-db6a19224154": "pedigree_matrix",
    "659f95f3-bc48-47e5-8584-64bd4477f3f2": "case_study"
}

tm_authenticators = {  # Authenticator
    "b33193c3-63b9-49f7-b888-ceba619d2812": "google",
    "c09fa36b-62a3-4904-9600-e3bb5028d809": "facebook",
    "f510cb30-7a44-4cb1-86f5-1b112e43293a": "firebase",
    "5f32a593-306f-4b69-983c-0a5680556fae": "local",
}

tm_case_study_version_statuses = {  # CaseStudyStatus
    "ee436cb7-0237-4b40-bd77-25acfade0f9b": "in_elaboration",
    "eef5c756-56ac-47a0-96cf-b7bebd73d392": "finished",
    "7d2cd2ae-0f4e-4962-b73c-4e40c553f533": "finished_and_published",
    "e5f3d8e6-accf-4175-9e74-5ad1b9a6faf5": "stopped"
}

# "22bc2577-883a-4408-bba9-fcace20c0fc8":
# "e80a7d27-3ec8-4aa1-b49c-5498e0f85bee":
# "d30120f0-28df-4bca-90e4-4f0676d1c874":
# "83084df6-7ad0-45d7-b3f1-6de594c78611":
# "7e23991b-24a0-4da1-8251-c3c3434dfb87":
# "bfc0c9fe-631f-44d0-8e96-e22d01ffb1ed":
# "dec7e901-b3f4-4343-b3d1-4fa5fbf3804e":
# "013b2f3b-4b2f-4b6c-8f5f-425132aea74b":
# "3eef41be-fde1-4ad4-92d0-fe795158b41d":
# "0fba3591-4ffc-4a88-977a-6e1d922f0735":
# "a61fc587-1272-4d46-bdd0-027cde1b8a78":
# "600397ef-0102-486e-a6f7-d43b0f8ce4b9":
# "763e57b7-2636-4c04-9861-d865fe0bb5ab":
# "788065a7-d9f5-46fa-b8ba-8bc223d09331":
# "38fb34f7-a952-4036-9b0b-4d6c59e8f8d4":
# "0292821a-dd33-450a-bdd8-813b2b95c456":


def register_external_datasources(cfg):
    dsm2 = DataSourceManager(session_factory=DBSession)

    # Eurostat
    dsm2.register_datasource_manager(Eurostat())

    # FAO
    if 'FAO_DATASETS_DIR' in cfg:
        fao_dir = cfg['FAO_DATASETS_DIR']
    else:
        fao_dir = "/home/rnebot/DATOS/FAOSTAT/"
    dsm2.register_datasource_manager(FAOSTAT(datasets_directory=fao_dir,
                                             metadata_session_factory=DBSession,
                                             data_engine=backend.data_engine))

    # OECD
    dsm2.register_datasource_manager(OECD())

    # FADN
    dsm2.register_datasource_manager(FADN(metadata_session_factory=DBSession,
                                          data_engine=backend.data_engine))
    # sources = dsm2.get_supported_sources()
    return dsm2
