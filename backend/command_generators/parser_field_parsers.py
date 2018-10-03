"""
Parsing of fields of the spreadsheet using PyParsing

Ideas for expression rules copied/adapted from:
https://gist.github.com/cynici/5865326

"""
from functools import partial
from pyparsing import (ParserElement, Regex,
                       oneOf, srange, operatorPrecedence, opAssoc,
                       Forward, Regex, Suppress, Literal, Word,
                       Optional, OneOrMore, ZeroOrMore, Or, alphas, alphanums, White,
                       Combine, Group, delimitedList, nums, quotedString, NotAny,
                       Keyword, removeQuotes, CaselessKeyword, OnlyOnce)

from backend import ureg
from backend.common.helper import create_dictionary, PartialRetrievalDictionary

# Enable memoizing
ParserElement.enablePackrat()

# Number
# Variable(key=expression, key=expression)[key=expression, key=expression]
# Hierarchical variable name
# Arithmetic Expression

# FORWARD DECLARATIONS
arith_boolean_expression = Forward()
expression = Forward()  # Generic expression (boolean, string, numeric)
expression_with_parameters = Forward()  # TODO Parameter value definition. An expression potentially referring to other parameters. Boolean operators. Simulation of IF ELIF ELSE or SWITCH
hierarchy_expression = Forward()
hierarchy_expression_v2 = Forward()
geo_value = Forward()  # TODO Either a Geo or a reference to a Geo
url_parser = Forward()  # TODO
context_query = Forward()  # TODO A way to find a match between pairs of processors. A pair of Processor Selectors
domain_definition = Forward()  # TODO Domain definition. Either a Category Hierarchy name or a numeric interval (with open closed)
parameter_value = Forward()  # TODO Parameter Value. Could be "expression_with_parameters"
indicator_expression = Forward()

# TOKENS

# Separators and operators (arithmetic and boolean)
lparen, rparen, lbracket, rbracket, lcurly, rcurly, dot, equals, hash = map(Literal, "()[]{}.=#")
double_quote = Literal('"')
single_quote = Literal("'")
quote = oneOf('" ''')  # Double quotes, single quotes
signop = oneOf('+ -')
multop = oneOf('* / // %')
plusop = oneOf('+ -')
comparisonop = oneOf("< <= > >= == != <>")
andop = CaselessKeyword("AND")
orop = CaselessKeyword("OR")
notop = CaselessKeyword("NOT")
processor_factor_separator = Literal(":")

# Boolean constants
true = Keyword("True")
false = Keyword("False")
# Simple identifier
simple_ident = Word(alphas, alphanums+"_")  # Start in letter, then "_" + letters + numbers
# Basic data types
positive_int = Word(nums).setParseAction(lambda t: {'type': 'int', 'value': int(t[0])})
positive_float = (Combine(Word(nums) + Optional("." + Word(nums))
                          + oneOf("E e") + Optional(oneOf('+ -')) + Word(nums))
                  | Combine(Word(nums) + "." + Word(nums))
                  ).setParseAction(lambda _s, l, t: {'type': 'float', 'value': float(t[0])}
                                   )
boolean = Or([true, false]).setParseAction(lambda t: {'type': 'boolean', 'value': bool(t[0])})

quoted_string = quotedString(r".*")
unquoted_string = Regex(r".*")  # Anything
alphanums_string = Word(alphanums)
code_string = Word(alphanums+"_")  # For codes in Categories, Code Lists

# RULES - literal ANY string
string = quotedString.setParseAction(
    lambda t: {'type': 'str', 'value': t[0][1:-1]}
)


# RULES - unit name
def parse_action_unit_name(s, l, tt):
    try:
        ureg.parse_expression(s)
        return {"type": "unit_name", "unit": s}
    except:
        raise Exception("Unit name invalid")


unit_name = Regex(r".*").setParseAction(parse_action_unit_name)


# RULES - Simple hierarchical name
#         A literal hierarchical name - A non literal hierarchical name could be "processor_name"
simple_h_name = (Group(simple_ident + ZeroOrMore(dot.suppress() + simple_ident)).setResultsName("ident")
                 ).setParseAction(lambda _s, l, t: {'type': 'h_var',
                                                    'parts': t.ident.asList(),
                                                    }
                                  )

# RULES - simple_hierarchical_name [":" simple_hierarchical_name]
factor_name = (Optional(simple_h_name.setResultsName("processor")) + Optional(Group(processor_factor_separator.suppress() + simple_h_name).setResultsName("factor"))
               ).setParseAction(lambda _s, l, t: {'type': 'pf_name',
                                                                'processor': t.processor if t.processor else None,
                                                                'factor': t.factor[0] if t.factor else None
                                                        }
                                      )

# RULES - processor_name
# "h{c}ola{b}.sdf{c}",
# "{a}.h{c}ola{b}.sdf{c}",
# "{a}b",
# "{a}b{c}",
# "aa",
# "aa{b}aa",
# "{a}",
item = lcurly.suppress() + simple_ident + rcurly.suppress()
processor_name_part_separator = Literal(".")
processor_name_part = Group((item("variable") | Word(alphanums)("literal")) + ZeroOrMore(
    item("variable") | Word(alphanums + "_")("literal")))


def parse_action_processor_name(s, l, tt, node_type="processor_name"):
    """
    Function to elaborate a node for evaluation of processor name (definition of processor name) or
    selection of processors, with variable names and wildcard (..)
    :param s:
    :param l:
    :param tt:
    :return:
    """
    variables = set()
    parts = []
    expandable = False
    for t in tt:
        st = eval(repr(t))  # Inefficient??
        if st == ".":
            parts.append(("separator", st))
            continue
        elif st == "..":
            parts.append(("separator", st))
            expandable = True
            continue
        classif = st[1]
        for tok in st[0]:
            # Find it in literals or in variables
            if "literal" in classif and tok in classif["literal"]:
                parts.append(("literal", tok))
            else:
                parts.append(("variable", tok))
                variables.add(tok)
                expandable = True

    # s2 = ""
    # for t in parts:
    #     if t[0] == "literal":
    #         s2 += t[1]
    #     elif t[0] == "separator":
    #         s2 += "."
    #     else:
    #         s2 += "{" + t[1] + "}"
    # print(s2)

    return dict(type=node_type, parts=parts, variables=variables, input=s, expandable=expandable, complex=expandable)


processor_name = (processor_name_part("part") + ZeroOrMore(processor_name_part_separator("separator") + processor_name_part("part"))).\
    setParseAction(parse_action_processor_name)

# RULES - processor_names (note the plural)
# "..",
# "..Crop",
# "Farm..Crop",
# "Farm..",
# "..Farm..",
# ".Farm",  # Invalid
# "Farm.",  # Invalid
# "h{c}ola{b}.sdf{c}",
# "{a}.h{c}ola{b}.sdf{c}",
# "{a}b",
# "{a}b{c}",
# "aa",
# "aa{b}aa",
# "{a}",
processor_names_wildcard_separator = Literal("..")
processor_names_pre = (Or([processor_name_part, Literal("*")])("part") +
                       ZeroOrMore((processor_name_part_separator("separator")+processor_name_part("part")) |
                                  (processor_names_wildcard_separator("separator")+Optional(processor_name_part("part")))
                                  )
                       )
processor_names = ((processor_names_wildcard_separator + Optional(processor_names_pre)) | processor_names_pre
                   ).setParseAction(partial(parse_action_processor_name, node_type="processor_names"))

# RULES - reference
reference = (lbracket + simple_ident.setResultsName("ident") + rbracket
             ).setParseAction(lambda _s, l, t: {'type': 'reference',
                                                'ref_id': t.ident
                                                }
                              )


# RULES - namespace, parameter list, named parameters list, function call
namespace = simple_ident + Literal("::").suppress()
named_parameter = Group(simple_ident + equals.suppress() + expression).setParseAction(lambda t: {'type': 'named_parameter', 'param': t[0][0], 'value': t[0][1]})
named_parameters_list = delimitedList(named_parameter, ",")
parameters_list = delimitedList(Or([expression, named_parameter]), ",")
func_call = Group(simple_ident + lparen.suppress() + parameters_list + rparen.suppress()
                  ).setParseAction(lambda t: {'type': 'function',
                                              'name': t[0][0],
                                              'params': t[0][1:]
                                              }
                                   )

# RULES - key-value list
# key "=" value. Key is a simple_ident; "Value" can be an expression referring to parameters
key_value = Group(simple_ident + equals.suppress() + arith_boolean_expression).setParseAction(lambda t: {'type': 'key_value', 'key': t[0][0], 'value': t[0][1]})
key_value_list = delimitedList(key_value, ",").setParseAction(
    lambda _s, l, t: {'type': 'key_value_list',
                      'parts': {t2["key"]: t2["value"] for t2 in t}
                      }
)


# RULES - dataset variable, hierarchical var name
dataset = (simple_ident.setResultsName("ident") +
           Optional(lparen.suppress() + parameters_list + rparen.suppress()).setResultsName("func_params") +
           lbracket.suppress() + named_parameters_list.setResultsName("slice_params") + rbracket.suppress()
           ).setParseAction(lambda _s, l, t: {'type': 'dataset',
                                              'name': t.ident,
                                              'func_params': t.func_params if t.func_params else None,
                                              'slice_params': t.slice_params,
                                              }
                            )
# [ns::]dataset"."column_name
dataset_with_column = (Optional(namespace).setResultsName("namespace") +
                       Group(Or([simple_ident]) + dot.suppress() + simple_ident).setResultsName("parts")
                       ).setParseAction(lambda _s, l, t:
                                                         {'type': 'h_var',
                                                          'ns': t.namespace[0] if t.namespace else None,
                                                          'parts': t.parts.asList(),
                                                          }
                                        )
# RULES - hierarchical var name
obj_types = Or([simple_ident, func_call, dataset])
# h_name, the most complex NAME, which can be a hierarchical composition of names, function calls and datasets
h_name = (Optional(namespace).setResultsName("namespace") +
          Group(obj_types + ZeroOrMore(dot.suppress() + obj_types)).setResultsName("parts")
          # + Optional(hash + simple_ident).setResultsName("part")
          ).setParseAction(lambda _s, l, t: {'type': 'h_var',
                                             'ns': t.namespace[0] if t.namespace else None,
                                             'parts': t.parts.asList(),
                                             }
                           )

# RULES - Arithmetic and Boolean expression
arith_boolean_expression << operatorPrecedence(Or([positive_float, positive_int, string, boolean,
                                     Optional(Literal('{')).suppress() + simple_h_name + Optional(Literal('}')).suppress(),
                                     func_call]),  # Operand types (REMOVED h_name: no "namespace" and no "datasets")
                                 [(signop, 1, opAssoc.RIGHT, lambda _s, l, t: {
                                     'type': 'u'+t.asList()[0][0],
                                     'terms': [0, t.asList()[0][1]],
                                     'ops': ['u'+t.asList()[0][0]]
                                  }),
                                  (multop, 2, opAssoc.LEFT, lambda _s, l, t: {
                                      'type': 'multipliers',
                                      'terms': t.asList()[0][0::2],
                                      'ops': t.asList()[0][1::2]
                                  }),
                                  (plusop, 2, opAssoc.LEFT, lambda _s, l, t: {
                                      'type': 'adders',
                                      'terms': t.asList()[0][0::2],
                                      'ops': t.asList()[0][1::2]
                                  }),
                                  (comparisonop, 2, opAssoc.LEFT, lambda _s, l, t: {
                                      'type': 'comparison',
                                      'terms': t.asList()[0][0::2],
                                      'ops': t.asList()[0][1::2]
                                  }),
                                  (notop, 1, opAssoc.RIGHT, lambda _s, l, t: {
                                      'type': 'not',
                                      'terms': [0, t.asList()[0][1]],
                                      'ops': ['u'+t.asList()[0][0]]
                                  }),
                                  (andop, 2, opAssoc.LEFT, lambda _s, l, t: {
                                      'type': 'and',
                                      'terms': t.asList()[0][0::2],
                                      'ops': t.asList()[0][1::2]
                                  }),
                                  (orop, 2, opAssoc.LEFT, lambda _s, l, t: {
                                      'type': 'or',
                                      'terms': t.asList()[0][0::2],
                                      'ops': t.asList()[0][1::2]
                                  }),
                                  ],
                                 lpar=lparen.suppress(),
                                 rpar=rparen.suppress())

# RULES - Expression varying value depending on conditions
condition = Optional(arith_boolean_expression("if") + Literal("->")) + arith_boolean_expression("then")
conditions_list = delimitedList(condition, ",").setParseAction(lambda _s, l, t:
                                                               {
                                                                   'type': 'conditional',
                                                                   'parts': t.asList()[0]
                                                               })

# RULES - Expression type 2
expression << operatorPrecedence(Or([positive_float, positive_int, string, h_name]),  # Operand types
                                 [(signop, 1, opAssoc.RIGHT, lambda _s, l, t: {'type': 'u'+t.asList()[0][0], 'terms': [0, t.asList()[0][1]], 'ops': ['u'+t.asList()[0][0]]}),
                                  (multop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'multipliers', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                  (plusop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'adders', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                  ],
                                 lpar=lparen.suppress(),
                                 rpar=rparen.suppress())

# RULES - Expression type 2. Can mention only parameters and numbers
# (for parameters, namespaces are allowed, and also hierarchical naming)
# TODO Check if it can be evaluated with "ast_evaluator"
expression_with_parameters = arith_boolean_expression
# expression_with_parameters << operatorPrecedence(Or([positive_float, positive_int, simple_h_name]),  # Operand types
#                                                  [(signop, 1, opAssoc.RIGHT, lambda _s, l, t: {'type': 'u'+t.asList()[0][0], 'terms': [0, t.asList()[0][1]], 'ops': ['u'+t.asList()[0][0]]}),
#                                                   (multop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'multipliers', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
#                                                   (plusop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'adders', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
#                                                   ],
#                                                  lpar=lparen.suppress(),
#                                                  rpar=rparen.suppress())

# RULES: Expression type 3. For hierarchies. Can mention only simple identifiers (codes) and numbers
hierarchy_expression << operatorPrecedence(Or([positive_float, positive_int, simple_ident]),  # Operand types
                                           [(signop, 1, opAssoc.RIGHT, lambda _s, l, t: {'type': 'u'+t.asList()[0][0], 'terms': [0, t.asList()[0][1]], 'ops': ['u'+t.asList()[0][0]]}),
                                            (multop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'multipliers', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                            (plusop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'adders', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                            ],
                                           lpar=lparen.suppress(),
                                           rpar=rparen.suppress())

# RULES: Expression type 4. For indicators. Can mention only numbers and core concepts
indicator_expression << operatorPrecedence(Or([positive_float, positive_int, factor_name]),  # Operand types
                                           [(signop, 1, opAssoc.RIGHT, lambda _s, l, t: {'type': 'u'+t.asList()[0][0], 'terms': [0, t.asList()[0][1]], 'ops': ['u'+t.asList()[0][0]]}),
                                            (multop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'multipliers', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                            (plusop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'adders', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                            ],
                                           lpar=lparen.suppress(),
                                           rpar=rparen.suppress())


# [expression2 (previously parsed)] [relation_operator] processor_or_factor_name
relation_operators = Or([Literal('|'),  # Part-of
                         Literal('>'), Literal('<'),  # Directed flow
                         Literal('<>'), Literal('><'),  # Undirected flow
                         Literal('||')  # Upscale (implies part-of also)
                         ]
                        )
relation_expression = (Optional(expression_with_parameters).setResultsName("weight") +
                       Optional(relation_operators).setResultsName("relation_type") +
                       factor_name.setResultsName("destination")
                       ).setParseAction(lambda _s, l, t: {'type': 'relation',
                                                          'name': t.destination,
                                                          'relation_type': t.relation_type,
                                                          'weight': t.weight
                                                          }
                                        )

# RULES: Expression type 5. For hierarchies (version 2). Can mention quoted strings (for codes), parameters and numbers
hierarchy_expression_v2 << operatorPrecedence(Or([positive_float, positive_int, quotedString, named_parameter]),  # Operand types
                                              [(signop, 1, opAssoc.RIGHT, lambda _s, l, t: {'type': 'u'+t.asList()[0][0], 'terms': [0, t.asList()[0][1]], 'ops': ['u'+t.asList()[0][0]]}),
                                            (multop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'multipliers', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                            (plusop, 2, opAssoc.LEFT, lambda _s, l, t: {'type': 'adders', 'terms': t.asList()[0][0::2], 'ops': t.asList()[0][1::2]}),
                                            ],
                                              lpar=lparen.suppress(),
                                              rpar=rparen.suppress())

# RULES: Level name
level_name = (simple_ident + Optional(plusop + positive_int)
              ).setParseAction(lambda t: {'type': 'level',
                                          'domain': t[0],
                                          'level': (t[1][0]+str(t[2]["value"])) if len(t) > 1 and t[1] else None
                                          }
                               )

# RULES: Time expression
# A valid time specification. Possibilities: Year, Month-Year / Year-Month, Time span (two dates)
period_name = Or([Literal("Year"), Literal("Semester"), Literal("Quarter"), Literal("Month")])
four_digits_year = Word(nums, min=4, max=4)
month = Word(nums, min=1, max=2)
year_month_separator = oneOf("- /")
date = Group(Or([four_digits_year.setResultsName("y") +
                 Optional(year_month_separator.suppress()+month.setResultsName("m")),
                 Optional(month.setResultsName("m") + year_month_separator.suppress()) +
                 four_digits_year.setResultsName("y")
                 ]
                )
             )
two_dates_separator = oneOf("- /")
time_expression = Or([(date + Optional(two_dates_separator.suppress()+date)
                       ).setParseAction(
                                       lambda _s, l, t:
                                       {'type': 'time',
                                        'dates': [{k: int(v) for k, v in d.items()} for d in t]
                                        }
                                       ),
                      period_name.setParseAction(lambda _s, l, t:
                                                 {'type': 'time',
                                                  'period': t[0]})
                      ])

# RULES: "Relative to"
factor_unit = (simple_h_name.setResultsName("factor") + Optional(Regex(".*").setResultsName("unparsed_unit"))).setParseAction(lambda _s, l, t:
                                                                            {'type': 'factor_unit',
                                                                             'factor': t.factor,
                                                                             'unparsed_unit': t.unparsed_unit if t.unparsed_unit else ""})

# #### URL Parser ################################################################

url_chars = alphanums + '-_.~%+'
fragment = Combine((Suppress('#') + Word(url_chars)))('fragment')
scheme = oneOf('http https ftp file')('scheme')
host = Combine(delimitedList(Word(url_chars), '.'))('host')
port = Suppress(':') + Word(nums)('port')
user_info = (
Word(url_chars)('username')
  + Suppress(':')
  + Word(url_chars)('password')
  + Suppress('@')
)

query_pair = Group(Word(url_chars) + Suppress('=') + Word(url_chars))
query = Group(Suppress('?') + delimitedList(query_pair, '&'))('query')

path = Combine(
  Suppress('/')
  + OneOrMore(~query + Word(url_chars + '/'))
)('path')

url_parser = (
  scheme.setResultsName("scheme")
  + Suppress('://')
  + Optional(user_info).setResultsName("user_info")
  + host.setResultsName("host")
  + Optional(port).setResultsName("port")
  + Optional(path).setResultsName("path")
  + Optional(query).setResultsName("query")
  + Optional(fragment).setResultsName("fragment")
).setParseAction(lambda _s, l, t: {'type': 'url',
                                   'scheme': t.scheme,
                                   'user_info': t.user_info,
                                   'host': t.host,
                                   'port': t.port,
                                   'path': t.path,
                                   'query': t.query,
                                   'fragment': t.fragment})


# #################################################################################################################### #

def clean_str(us):
    # "En dash"                 character is replaced by minus (-)
    # "Left/Right double quote" character is replaced by double quote (")
    # "Left/Right single quote" character is replaced by single quote (')
    # "€"                       character is replaced by "eur"
    # "$"                       character is replaced by "usd"
    return us.replace(u'\u2013', '-').\
              replace(u'\u201d', '"').replace(u'\u201c', '"').\
              replace(u'\u2018', "'").replace(u'\u2019', "'"). \
              replace('€', 'eur').\
              replace('$', 'usd')


def string_to_ast(rule: ParserElement, input_: str):
    res = rule.parseString(clean_str(input_), parseAll=True)
    res = res.asList()[0]
    while isinstance(res, list):
        res = res[0]
    return res


if __name__ == '__main__':
    from backend.model_services import State
    from dotted.collection import DottedDict

    examples = [
                "'Hola'",
                "'Hola' + 'Adios'",
                "{Param} * 3 >= 0.3",
                "5 * {Param1}",
                "True",
                "(Param * 3 >= 0.3) AND (Param2 * 2 <= 0.345)",
                "cos(Param*3.1415)",
                "'Hola' + Param1"
    ]
    for e in examples:
        try:
            ast = string_to_ast(arith_boolean_expression, e)
            print(ast)
        except:
            print("Incorrect")

    s = "c1 + c30 - c2 - 10"
    res = string_to_ast(hierarchy_expression, s)
    s = "ds.col"
    res = string_to_ast(h_name, s)

    s = State()
    examples = [
        "2017-05 - 2018-01",
        "2017",
        "5-2017",
        "5-2017 - 2018-1",
        "2017-2018"
    ]
    for example in examples:
        print(example)
        res = string_to_ast(time_expression, example)
        print(res)
        print("-------------------")

    s.set("HH", DottedDict({"Power": {"p": 34.5, "Price": 2.3}}))
    s.set("EN", DottedDict({"Power": {"Price": 1.5}}))
    s.set("HH", DottedDict({"Power": 25}), "ns2")
    s.set("param1", 0.93)
    s.set("param2", 0.9)
    s.set("param3", 0.96)
    examples = [
        "EN(p1=1.5, p2=2.3)[d1='C11', d2='C21'].v2",  # Simply sliced Variable Dataset (function call)
        "a_function(p1=2, p2='cc', p3=1.3*param3)",
        "-5+4*2",  # Simple expression #1
        "HH",  # Simple name
        "HH.Power.p",  # Hierarchic name
        "5",  # Integer
        "1.5",  # Float
        "1e-10",  # Float scientific notation
        "(5+4)*2",  # Simple expression #2 (parenthesis)
        "3*2/6",  # Simple expression #3 (consecutive operators of the same kind)
        "'hello'",  # String
        "ns2::HH.Power",  # Hierarchic name from another Namespace
        "HH.Power.Price + EN.Power.Price * param1",
        "EN[d1='C11', d2='C21'].d1",  # Simple Dataset slice
        "b.a_function(p1=2, p2='cc', p3=1.3*param3)",
        "b.EN[d1='C11', d2='C21'].d1",  # Hierachical Dataset slice
        "tns::EN(p1=1.5+param2, p2=2.3 * 0.3)[d1='C11', d2='C21'].v2",  # Simply sliced Variable Dataset (function call)
    ]
    # for example in examples:
    #     print(example)
    #     res = string_to_ast(expression, example)
    #     print(res)
    #     issues = []
    #     value = ast_evaluator(res, s, None, issues)
    #     print(str(type(value))+": "+str(value))

