import openpyxl
import numpy as np

from backend.command_generators.spreadsheet_utils import worksheet_to_numpy_array, obtain_rectangular_submatrices


def parse_pedigree_matrix_command(sh, area):
    """
    A pedigree matrix is formed by several columns, with a header and below a list of codes, in ascending qualitative
    order.

    Codes can be referred later by the order number (first code is 1 and so on). The order of the columns serves also
    to sequence the codes of the template, from left to right.

    Columns can be accompanied by a description column, to the right

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

    columns = []
    column = dict(h=header, d=description, )


    return issues, None, dict()


