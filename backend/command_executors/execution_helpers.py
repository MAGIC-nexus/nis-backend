from backend.command_generators import parser_field_parsers
from backend.common.helper import create_dictionary


def parse_line(item, fields):
    """

    :param item:
    :param fields:
    :return:
    """
    asts = {}
    for f, v in item.items():
        if not f.startswith("_"):
            field = fields[f]
            # Parse (success is guaranteed because of the first pass dedicated to parsing)
            asts[f] = parser_field_parsers.string_to_ast(field.parser, v)
    return asts


def classify_variables(asts, datasets, hierarchies, parameters):
    """

    :param asts:
    :param datasets:
    :param hierarchies:
    :param parameters:
    :return:
    """
    ds = set()
    ds_concepts = set()
    hh = set()
    params = set()
    not_classif = set()
    for ast in asts:
        for var in ast["variables"] if "variables" in ast else []:
            parts = var.split(".")
            first_part = parts[0]
            if first_part in datasets:
                ds.add(datasets[first_part])
                ds_concepts.add(parts[1])  # Complete concept name: dataset"."concept
            elif first_part in hierarchies:
                hh.add(hierarchies[first_part])
            elif first_part in parameters:
                params.add(first_part)
            else:
                not_classif.add(first_part)

    return dict(datasets=ds, ds_concepts=ds_concepts, hierarchies=hh, parameters=params, not_classif=not_classif)


def obtain_dictionary_with_literal_fields(item, asts):
    d = create_dictionary()
    for f in item:
        if not f.startswith("_"):
            ast = asts[f]
            if "complex" not in ast or ("complex" in ast and not ast["complex"]):
                d[f] = item[f]
    return d
