from typing import Dict, Any, NoReturn

from backend.command_executors import BasicCommand
from backend.command_field_definitions import get_command_fields_from_class
from backend.command_generators import IType
from backend.common.helper import strcmp
from backend.models.musiasem_concepts import Indicator, IndicatorCategories, Benchmark


class ScalarIndicatorsCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, fields: Dict[str, Any]) -> NoReturn:
        """
        Create and register Indicator object

        :param fields:
        """
        benchmark_name = fields["benchmark"]
        if benchmark_name:
            benchmark = self._glb_idx.get(Benchmark.partial_key(benchmark_name))
            if len(benchmark) == 1:
                benchmark = benchmark[0]
            elif len(benchmark) == 0:
                self._add_issue(IType.ERROR,
                                f"Benchmark {benchmark_name} does not exist (it must be declared previously in a "
                                "ScalarBenchmark command worksheet")
                return
            elif len(benchmark) > 1:
                self._add_issue(IType.ERROR,
                                f"Benchmark {benchmark_name} exists {len(benchmark)} times."
                                " Only one occurrence is allowed.")
                return
        else:
            benchmark = None
        indicator = Indicator(fields["indicator_name"],
                              fields["formula"],
                              None,
                              fields.get("processors_selector"),
                              benchmark,
                              IndicatorCategories.factors_expression if strcmp(fields.get("local"), "Yes")
                              else IndicatorCategories.case_study,
                              fields.get("description"))
        self._glb_idx.put(indicator.key(), indicator)
