import json

from backend.model.memory.musiasem_concepts import Indicator
from backend.model_services import IExecutableCommand, get_case_study_registry_objects


class IndicatorsCommand(IExecutableCommand):
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """

        :param state:
        :return:
        """
        # TODO At SOLVE time (not here, which is only in charge of creating structure, Indicator objects and Benchmark objects)

        # TODO For each indicator
        # TODO Parse it, and return the number of referenced Processors:
        # TODO If referenced Factors are all inside a Processor, it returns ONE -> IndicatorCategories.factors_expression
        # TODO If Factors are from different Processors, it returns > ONE -> IndicatorCategories.case_study
        # TODO If everything is FactorTypes, it returns ZERO -> IndicatorCategories.factor_types_expression
        # TODO This last case will serve at solve time to potentially generate an Indicator per Processor matching the FactorTypes mentioned in the indicator (first the FactorTypes have to be resolved)

        # TODO Only the factor_types_expression can be validated at this EXECUTION time, all FactorTypes MUST exist
        # TODO The other two types "factors_expression" and "case_study" can mention FactorTypes which are not currently expanded (they are expanded at reasoning/solving time)
        # Obtain global variables in state
        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

        for indicator in self._content:
            name = indicator["name"]
            formula = indicator["formula"]
            indi = Indicator(name, formula, from_indicator=None, benchmark=None, indicator_category=None)
            if "description" in indicator:
                indi._description = indicator["description"]
            glb_idx.put(indi.key(), indi)

        return None, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the metadata dictionary
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)

        return issues