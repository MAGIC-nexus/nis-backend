from enum import Enum
from attr import attrs, attrib, validators
from openpyxl.utils import get_column_letter


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
    itype = attrib(validator=validators.instance_of(IType))  # type: IType
    # An english description of what happened
    description = attrib()  # type: str
    # Command type
    ctype = attrib(default=None)  # type: str
    # Where is the issue. The expression of the location depends. For spreadsheet it is sheet name, row and column
    location = attrib(default=None)  # type: IssueLocation

    def __str__(self):
        return f'(level={self.itype.name}, msg="{self.description}", cmd="{self.ctype}", {self.location})'
