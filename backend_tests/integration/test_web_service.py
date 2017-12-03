import os
import io
import unittest
import requests
import json

"""
* CORS. Probar a hacer las dos primeras llamadas (/isession y /isession/identify?user=test_user)
* Llamadas asíncronas. Hacer un componente que muestre el contenido del JSON
* Token Firebase. ¿Cómo comprobar la validez?

* Estudiar un poco Angular2
* ¿Cómo incluir compilado de Angular en Proyecto? (o poner en proyecto aparte)
* Revisar cada uno de los servicios
* Llamadas desde cliente ¿cómo llegan?
* Creación de casos de uso FIJOS. BDD local
* Autenticación. Bearer

* Crear usuarios a mano: asosa, rnebot

* Otros servicios:

 
"""

os.environ["MAGIC_NIS_SERVICE_CONFIG_FILE"] = "./nis_unittests.conf"
import backend.restful_service
backend.restful_service.app.config["TESTING"] = "True"
import backend.restful_service.service_main


def to_str(resp_data):
    import flask.wrappers
    if isinstance(resp_data, flask.wrappers.Response):
        return resp_data.data.decode("utf-8")
    elif isinstance(resp_data, requests.Response):
            return resp_data.text
    else:
        return resp_data.decode("utf-8")


def to_json(resp_data):
    import flask.wrappers
    if isinstance(resp_data, flask.wrappers.Response):
        return json.loads(resp_data.data.decode("utf-8"))
    elif isinstance(resp_data, requests.Response):
            return resp_data.json()
    else:
        return json.loads(resp_data.decode("utf-8"))


class RequestsClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.cookies = None

    def get(self, get_action):
        if get_action.startswith("http"):
            url = get_action
        else:
            url = self.base_url + get_action
        r = requests.get(url, cookies=self.cookies)
        if len(r.cookies.list_domains()) > 0:
            dom1 = r.cookies.list_domains()[0]
            self.cookies = r.cookies.get_dict()
        r.data = r.content
        return r

    def post(self, post_action, data: dict):
        if post_action.startswith("http"):
            url = post_action
        else:
            url = self.base_url + post_action
        r = requests.post(url, data, cookies=self.cookies)
        if len(r.cookies.list_domains()) > 0:
            dom1 = r.cookies.list_domains()[0]
            self.cookies = r.cookies.get_dict()
        r.data = r.content
        return r


def setUpModule():
    print('In setUpModule()')


def tearDownModule():
    print('In tearDownModule()')
    del backend.restful_service


class HighLevelUseCases(unittest.TestCase):

    # ## Called before and after methods in the class are to be executed ###
    @classmethod
    def setUpClass(cls):
        print('In setUpClass()')
        # Alternative:
        # HighLevelUseCases.app = RequestsClient("http://localhost:5000")
        HighLevelUseCases.app = backend.restful_service.app.test_client()

    @classmethod
    def tearDownClass(cls):
        print('In tearDownClass()')

    # ######## Called before and after each method ##############

    def setUp(self):
        super().setUp()
        print('\nIn setUp()')

    def tearDown(self):
        print('In tearDown()')
        super().tearDown()

    # ###########################################################

    def test_001_new_case_study_with_only_metadata_command(self):
        a = HighLevelUseCases.app
        # Reset DB
        r = a.post("/resetdb")
        self.assertEqual(r.status_code, 204)

        # An interactive session
        r = a.post("/isession")
        self.assertEqual(r.status_code, 204)
        r = a.put("/isession/identity?user=test_user")
        self.assertEqual(r.status_code, 200)

        # Create reproducible session, prepared to create a new case study
        r = a.post("/isession/rsession", data={"create_new": "case_study", "allow_saving": "True"})
        self.assertEqual(r.status_code, 204)
        var_name = "a_var"
        var_value = "the_value"
        # Add a command to the session
        r = a.post("/isession/rsession/command?execute=True&register=True",
                   data=json.dumps({"command": "dummy", "content": {"name": var_name, "description": var_value}}),
                   headers={"Content-Type": "text/json"})
        self.assertEqual(r.status_code, 204)

        # Close the reproducible session
        r = a.delete("/isession/rsession?save_before_close=True")
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        uuid2 = d.get("session_uuid")
        v_uuid = d.get("version_uuid")
        cs_uuid = d.get("case_study_uuid")
        self.assertIsNotNone(uuid2)
        self.assertIsNotNone(v_uuid)
        self.assertIsNotNone(cs_uuid)

        # TODO Change permissions. Automatically the creator has access (no need to be in the ACL). Add another user
        # Logout
        a.delete("/isession")

        # List case studies (error, logged out)
        r = a.get("/case_studies/")
        self.assertEqual(r.status_code, 400)

        # Start interactive session AGAIN
        a.post("/isession")

        # List case studies of ANONYMOUS user
        r = a.get("/case_studies/")
        lst = json.loads(r.data)
        self.assertEqual(len(lst), 1)

        # List case studies of TEST_USER
        a.put("/isession/identity?user=test_user")
        r = a.get("/case_studies/")
        lst = json.loads(r.data)
        self.assertEqual(len(lst), 1)
        # List case study versions
        uuid2 = lst[0]["uuid"]
        r = a.get(lst[0]["versions"])  # Use the navigation URL
        d = json.loads(r.data)
        self.assertIsNotNone(d["uuid"])
        # List case study version sessions
        # Open a reproducible session, add another command

        # Get a listing of all variables
        r = a.get("/case_studies/"+cs_uuid+"/versions/"+v_uuid+"/variables/")
        d = json.loads(r.data)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["name"], var_name)

        # Get some variable from version
        r = a.get("/case_studies/"+cs_uuid+"/versions/"+v_uuid+"/variables/"+var_name)
        d = json.loads(r.data)
        self.assertEqual(d[var_name], var_value)

    def test_002_new_case_study_send_xlsx_file(self):
        a = HighLevelUseCases.app
        # Reset DB
        r = a.post("/resetdb")
        self.assertEqual(r.status_code, 204)
        # An interactive session
        r = a.post("/isession")
        self.assertEqual(r.status_code, 204)
        r = a.put("/isession/identity?user=test_user")
        self.assertEqual(r.status_code, 200)

        # Create reproducible session, prepared to create a new case study
        r = a.post("/isession/rsession", data={"create_new": "case_study", "allow_saving": "True"})
        self.assertEqual(r.status_code, 204)
        # Send Excel file: store and execute it.
        with open("/home/rnebot/input_file.xlsx", "rb") as f:
            b = f.read()
        r = a.post("/isession/rsession/generator?execute=True&register=True",
                   data={'file': (io.BytesIO(b), "input_file.xlsx")},
                   headers={"Content-Type": "multipart/form-data"})  # application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        self.assertEqual(r.status_code, 204)
        # Close the reproducible session
        r = a.delete("/isession/rsession?save_before_close=True")
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        uuid2 = d.get("session_uuid")
        self.assertIsNotNone(uuid2)
        # Logout
        a.delete("/isession")

        # Login, download the Excel file, compare it with "b"
        # An interactive session
        r = a.post("/isession")
        self.assertEqual(r.status_code, 204)
        r = a.put("/isession/identity?user=test_user")
        self.assertEqual(r.status_code, 200)
        # List session commands
        # r = a.get("/case_studies/dummy/versions/dummy/sessions/"+uuid2)
        # Get the content of some command
        r = a.get("/case_studies/dummy/versions/dummy/sessions/" + uuid2 + "/0")
        self.assertEqual(r.data, b)

        # TODO Same isession, open "uuid2", get some variable from state modified by the Excel file
        # Open previous reproducible session for reading
        r = a.post("/isession/rsession", data={"uuid": uuid2, "read_version_state": "True"})
        self.assertEqual(r.status_code, 204)

    def test_003_manage_users(self):
        # TODO Login as ADMIN
        # TODO Create user1
        # TODO Create user2

        # TODO Login as user1
        # TODO Create case study
        # TODO Login as user2
        # TODO List case studies
        # TODO The list must be length 0
        # TODO Try to access directly to the case study
        # TODO An error should be returned
        # TODO Login as user1
        # TODO Allow read access to user2
        # TODO Login as user2
        # TODO List case studies
        # TODO The list must be length 1
        # TODO Try to access directly the case study
        # TODO It should be returned correctly
        # TODO Try to add a session to the case study, An error should be thrown
        # TODO Try to create a version, an error should be thrown
        # TODO Try to create a new case study, it should be OK
        # TODO Login as user1
        # TODO Allow contribute access to user2
        # TODO Login as user2
        # TODO Try to add a session to the case study, it should be OK
        # TODO Try to add a version, it should be OK
        # TODO Delete user2. ¿disable it?
        pass

    def test_submit_worksheet_new_version(self):
        pass

    def test_submit_worksheet_evolve_version(self):
        pass

    def test_submit_r_script_new_case_study(self):
        pass

    def test_delete_case_study_version(self):
        pass

    def test_delete_case_study(self):
        pass

    def test_list_case_studies_1(self):
        pass

    def test_list_case_studies_2(self):
        pass

    def test_list_case_studies_3(self):
        pass
