from typing import Dict, Any

from backend.command_executors import BasicCommand
from backend.command_field_definitions import get_command_fields_from_class
from backend.command_generators import IType
from backend.common.helper import create_dictionary
from backend.models.musiasem_concepts import Benchmark


class ScalarIndicatorBenchmarksCommand(BasicCommand):
    def __init__(self, name: str):
        BasicCommand.__init__(self, name, get_command_fields_from_class(self.__class__))

    def _process_row(self, fields: Dict[str, Any]) -> None:
        """
        Create and register Benchmark object

        :param fields:
        """
        name = fields["benchmark"]
        benchmark_group = fields["benchmark_group"]
        stakeholders = fields["stakeholders"]
        b = self._glb_idx.get(Benchmark.partial_key(name=name))
        if len(b) == 1:
            b = b[0]
        elif len(b) == 0:
            b = Benchmark(name, benchmark_group, stakeholders.split(",") if stakeholders else [])
            self._glb_idx.put(b.key(), b)
        else:
            self._add_issue(IType.ERROR,
                            f"There are {len(b)} instances of the Benchmark '{name}'")
            return

        # Add range, if not repeated
        category = fields["category"]
        if category not in b.ranges:
            b.ranges[category] = create_dictionary(
                data=dict(range=fields["range"],
                          unit=fields["unit"],
                          category=category,
                          label=fields["label"],
                          description=fields["description"])
            )
        else:
            self._add_issue(IType.WARNING,
                            f"Range with category '{category}' repeated")

