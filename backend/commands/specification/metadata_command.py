import json

from backend.domain import IExecutableCommand


class MetadataCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._metadata_dictionary = {}

    def execute(self, state: "State"):
        """ The execution creates an instance of a Metadata object, and assigns the name "metadata" to the variable,
            inserting it into "State" 
        """
        state.set("metadata", self._metadata_dictionary)
        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return json.dumps(self._metadata_dictionary)

    def json_deserialize(self, json_input):
        # TODO Read and check keys validity
        valid_keys = ["CaseStudyName"]
        metadata = {"CaseStudyName": "A test",
                    "CaseStudyCode": None,
                    "Title": "Case study for test only",
                    "SubjectTopicKeywords": ["Desalination", "Renewables", "Sustainability"],
                    "Description": "A single command case study",
                    "Level": "Local",
                    "Dimensions": ["Water", "Energy", "Food"],
                    "ReferenceDocumentation": ["reference 1", "reference 2"],
                    "Authors": ["Author 1", "Author 2"],
                    "DateOfElaboration": "2017-11-01",
                    "TemporalSituation": "Years 2015 and 2016",
                    "GeographicalLocation": "South",
                    "DOI": None,
                    "Language": "English"
                    }
        if isinstance(json_input, dict):
            self._metadata_dictionary = json_input
        else:
            self._metadata_dictionary = json.loads(json_input)
