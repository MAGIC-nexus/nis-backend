import json
import pandas as pd
import numpy as np
from openpyxl.chart import reference

from backend.model_services import IExecutableCommand, State, get_case_study_registry_objects
from backend.model.memory.musiasem_concepts import Observer, Processor, Factor, \
    ProcessorsRelationUndirectedFlowObservation, ProcessorsRelationPartOfObservation, \
    ProcessorsRelationUpscaleObservation, \
    FactorsRelationDirectedFlowObservation, FactorQuantitativeObservation


class UpscaleCommand(IExecutableCommand):
    """
    Serves to instantiate processor templates inside another processor
    See Almeria case study for the base example of this
    """
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """
        For each parent processor clone all the child processors.
        The cloning process may pass some factor observation, that may result in
        """
        some_error = False
        issues = []

        parent_processor_type = self._content["parent_processor_type"]
        child_processor_type = self._content["child_processor_type"]
        scaled_factor = self._content["scaled_factor"]
        source = self._content["source"]
        column_headers = self._content["column_headers"]
        row_headers = self._content["row_headers"]
        scales = self._content["scales"]

        # Find processor sets, for parent and child
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)
        if parent_processor_type not in p_sets:
            some_error = True
            issues.append((3, "The processor type '"+parent_processor_type +
                           "' (appointed for parent) has not been found in the commands execute so far"))

        if child_processor_type not in p_sets:
            some_error = True
            issues.append((3, "The processor type '"+child_processor_type +
                           "' (should be child processor) has not been found in the commands execute so far"))

        if some_error:
            return issues, None

        # Processor Sets have associated attributes, and each of them has a code list
        parent = p_sets[parent_processor_type]
        child = p_sets[child_processor_type]

        for c in column_headers:  # A list in vertical (top of the matrix)
            for i, cc in enumerate(c):
                found = False
                for k in parent.attributes:
                    if cc in k:
                        found = True
                        ("column", i, k, parent)
                for k in child.attributes:
                    if cc in k:
                        found = True
                        ("column", i, k, child)

        parent.attributes

        # Analyze and Locate taxa mentioned in rows and columns

        # Build a DataFrame
        df = pd.DataFrame(data=np.array(scales).reshape((len(row_headers), len(column_headers))))

        # From each key identify taxa containing, then which processor type uses the taxa

        lst_taxa = []

        # columns = [('c', 'a'), ('c', 'b')]
        # df.columns = pd.MultiIndex.from_tuples(columns)

        # CREATE the Observer of the Upscaling
        oer = glb_idx.get(Observer.partial_key(source, registry=glb_idx))
        if not oer:
            oer = Observer(source)
            glb_idx.put(oer.key(glb_idx), oer)
        else:
            oer = oer[0]

        # TODO Obtain the parent and child processors
        parent = None
        child = None
        quantity = None

        # Clone the child processor.
        cloned_child = Processor.clone(child, reference=True)

        # Elaborate the Observations: relational and quantitative
        o1 = ProcessorsRelationPartOfObservation.create_and_append(parent, cloned_child, oer)  # Part-of
        glb_idx.put(o1.key(glb_idx), o1)
        o2 = ProcessorsRelationUndirectedFlowObservation.create_and_append(parent, cloned_child, oer)  # Flow
        glb_idx.put(o2.key(glb_idx), o2)
        o3 = ProcessorsRelationUpscaleObservation.create_and_append(parent, cloned_child, scaled_factor, quantity)  # Upscale
        glb_idx.put(o3.key(glb_idx), o3)

        return issues, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the content
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)
        return issues
