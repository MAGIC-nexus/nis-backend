from enum import Enum
from attr import attrs, attrib, validators
from openpyxl.utils import get_column_letter

import nexinfosys
from nexinfosys.common.helper import create_dictionary


class IType(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3


@attrs
class IssueLocation:
    sheet_name = attrib()
    sheet_number = attrib(default=None)
    row = attrib(default=None)
    column = attrib(default=None)

    def __str__(self):
        return f'(sheet_name="{self.sheet_name}", sheet_no={self.sheet_number}, row={self.row}, ' \
               f'column={get_column_letter(self.column) if self.column else "-"})'


@attrs
class Issue:
    # (1) Info, (2) Warning, (3) Error
    itype = attrib()  # type: IType
    # An english description of what happened
    description = attrib()  # type: str
    # Command type
    ctype = attrib(default=None)  # type: str
    # Where is the issue. The expression of the location depends. For spreadsheet it is sheet name, row and column
    location = attrib(default=None)  # type: IssueLocation

    def __str__(self):
        return f'(level={self.itype.name}, msg="{self.description}", cmd="{self.ctype}", {self.location})'

    @itype.validator
    def check(self, attribute, value):
        if not isinstance(value, nexinfosys.command_generators.IType):
            raise ValueError("itype should be an instance of enum IType")


# List of Global Functions (used by "parser_field_parsers.py" and "parser_ast_evaluators.py")
global_functions = create_dictionary(data={i["name"]: i for i in
                    [{"name": "cos",
                      "full_name": "math.cos",
                      "kwargs": None},
                     {"name": "sin",
                      "full_name": "math.sin",
                      "kwargs": None},
                     {"name": "InterfaceType",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.get_interface_type",
                      "kwargs": None,
                      "special_kwargs": {"PartialRetrievalDictionary": "prd"}},
                     {"name": "NISName",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.get_nis_name",
                      "kwargs": None},
                     {"name": "Processor",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.get_processor",
                      "kwargs": None,
                      "special_kwargs": {"PartialRetrievalDictionary": "prd"}}
                     ]
                    })

global_functions_extended = create_dictionary(data={i["name"]: i for i in
                    [{"name": "sum",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.aggregator_sum",
                      "kwargs": None,
                      "special_kwargs": {"ProcessorsMap": "processors_map", "ProcessorsDOM": "processors_dom",
                                         "DataFrameGroup": "df_group", "IndicatorsDataFrameGroup": "df_indicators_group"}},
                     {"name": "avg",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.aggregator_avg",
                      "kwargs": None,
                      "special_kwargs": {"ProcessorsMap": "processors_map", "ProcessorsDOM": "processors_dom",
                                         "DataFrameGroup": "df_group", "IndicatorsDataFrameGroup": "df_indicators_group"}},
                     {"name": "max",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.aggregator_max",
                      "kwargs": None,
                      "special_kwargs": {"ProcessorsMap": "processors_map", "ProcessorsDOM": "processors_dom",
                                         "DataFrameGroup": "df_group", "IndicatorsDataFrameGroup": "df_indicators_group"}},
                     {"name": "min",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.aggregator_min",
                      "kwargs": None,
                      "special_kwargs": {"ProcessorsMap": "processors_map", "ProcessorsDOM": "processors_dom",
                                         "DataFrameGroup": "df_group", "IndicatorsDataFrameGroup": "df_indicators_group"}},
                     {"name": "count",
                      "full_name": "nexinfosys.command_generators.parser_ast_evaluators.aggregator_count",
                      "kwargs": None,
                      "special_kwargs": {"ProcessorsMap": "processors_map", "ProcessorsDOM": "processors_dom",
                                         "DataFrameGroup": "df_group", "IndicatorsDataFrameGroup": "df_indicators_group"}},
                     ]
                    })
