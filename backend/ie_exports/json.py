"""
Serialize a model once it has been built
Before solving and/or after solving. Clone operations executed. Scales not performed.

It could be something like:

Parameters:
[
 {
 },
],
CodeHierarchies:
[
 {
 },
],
Observers:
[
 {
 },
],
InterfaceTypes:
[
 {
 },
],
InterfaceTypeConverts: (if conversions not done already)
[
 {
 },
],
Processors:
[
 {
  Interfaces:
  [
   {...,
    Observations:
    [
     {
     },
    ]
   }
  ]
 }
],
Relationships:
[
 {
  Origin: {name, uuid}
  Destination: {name, uuid}
  Type
 }
]
"""
import json
import jsonpickle

from backend.common.helper import create_dictionary, CustomEncoder
from backend.model_services import State
from backend.models.musiasem_concepts import Processor, Factor, Parameter, Hierarchy, Taxon, Observer
from backend.solving import BasicQuery


def export_model_to_json(state: State) -> str:
    object_types = [Processor, Parameter, Hierarchy, Observer
     #, Factor, FactorType,
     # FactorQuantitativeObservation,
     # FactorTypesRelationUnidirectionalLinearTransformObservation,
     # ProcessorsRelationPartOfObservation, ProcessorsRelationUpscaleObservation,
     # ProcessorsRelationUndirectedFlowObservation,
     # FactorsRelationDirectedFlowObservation
     ]
    query = BasicQuery(state)
    objects = query.execute(object_types, filt="")

    for obj_type in object_types:
        print(obj_type.__name__)

        selected_objects = objects.get(obj_type)

        if selected_objects:

            if obj_type is Hierarchy:
                # Just get the Hierarchy objects of type Taxon
                selected_objects = [o for o in selected_objects if o.hierarchy_type == Taxon]
                # Sort the list to show the "code lists" first
                selected_objects.sort(key=lambda h: not h.is_code_list)

            for obj in selected_objects:
                print(json.dumps(obj, cls=CustomEncoder, indent=4))

    return "TODO"


def test():
    class Foo:
        def __init__(self):
            self.x = 1
            self.y = 2

        def __getstate__(self):
            return {'a': self.x * 2, 'b': self.y * 2}

    foo = Foo()
    foo_str = jsonpickle.encode(foo, unpicklable=False)
    print(type(foo_str))
    print(foo_str)

    dic = create_dictionary(case_sens=False, data={'A': 2, 'b': 4})
    print(type(dic))
    print(jsonpickle.encode(dic, unpicklable=False))
    print(dic.get_data())
    print(dic.get_original_data())


if __name__ == '__main__':
    test()
