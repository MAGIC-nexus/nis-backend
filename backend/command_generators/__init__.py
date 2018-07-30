from attr import attrs, attrib


@attrs
class Issue:
    # (1) Info, (2) Warning, (3) Error
    itype = attrib()  # type: int
    # An english description of what happened
    description = attrib()
    # Where is the issue. The expression of the location depends. For spreadsheet it is sheet name, row and column
    location = attrib()
