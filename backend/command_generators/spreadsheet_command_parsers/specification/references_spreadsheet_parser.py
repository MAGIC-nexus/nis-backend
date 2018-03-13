import openpyxl
import numpy as np

from backend.command_generators.spreadsheet_utils import worksheet_to_numpy_array, obtain_rectangular_submatrices


def parse_references_command(sh, area):
    """
    Elaborate a list of dictionaries {key: value} which can be reused by other objects, referring them by a unique ref_id field

    For the definition of keys:values, two options:
     * If the column has a header, that would be the key
     * If the column does not have a header, both key and value can be specified in the cell, separated by "->" or ":" (which one?)

    :param sh: Input worksheet
    :param area: Tuple (top, bottom, left, right) representing the rectangular area of the input worksheet where the
    command is present
    :return: list of issues (issue_type, message), command label, command content
    """

    some_error = False
    issues = []

    references = []

    # TODO Analyze columns. "ref_id" must exist. Obtain columns where

    # TODO Read each row
    for r in range(area[0]+1, area[1]):
        # Gather row
        reference = {"type": ["undefined", "bibliographic", "geographic"],
                     "key1": "value1"}
        # TODO Depending on the type, check for the presence of certain attributes
        # TODO Depending on the type, validate certain attributes

        references.append(reference)


    content = references

    return issues, None, content


