from attr import attrs, attrib
from openpyxl.utils import get_column_letter


class IType:
    _values = {'INFO': 1, 'WARNING': 2, 'ERROR': 3}
    _names = {v: k for k, v in _values.items()}

    @staticmethod
    def name(value: int):
        return IType._names[value]

    @staticmethod
    def info() -> int:
        return IType._values["INFO"]

    @staticmethod
    def warning() -> int:
        return IType._values["WARNING"]

    @staticmethod
    def error() -> int:
        return IType._values["ERROR"]

    @staticmethod
    def valid_values():
        return ", ".join([f'{value} ({name})' for name, value in IType._values.items()])

    @staticmethod
    def validator(instance, attribute, value):
        if value not in IType._values.values():
            raise ValueError(f"itype '{value}' is not correct. Valid values are: {IType.valid_values()}")


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
    itype = attrib(validator=IType.validator)  # type: int
    # An english description of what happened
    description = attrib()  # type: str
    # Command type
    ctype = attrib(default=None)  # type: str
    # Where is the issue. The expression of the location depends. For spreadsheet it is sheet name, row and column
    location = attrib(default=None)  # type: IssueLocation

    def __str__(self):
        return f'(level={IType.name(self.itype)}, msg="{self.description}", cmd="{self.ctype}", {self.location})'
