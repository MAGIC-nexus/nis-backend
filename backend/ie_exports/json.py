"""
Serialize a model once it has been built
Before solving and/or after solving. Clone operations executed. Scales not performed.

It could be something like:

Metadata:
{
{,
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
from collections import OrderedDict

import jsonpickle
from typing import Dict, List, Union, Optional, Any

from backend.common.helper import create_dictionary, CustomEncoder, values_of_nested_dictionary
from backend.model_services import State
from backend.models.musiasem_concepts import Processor, Parameter, Hierarchy, Taxon, Observer, FactorType, \
    ProcessorsRelationPartOfObservation, ProcessorsRelationUpscaleObservation, ProcessorsRelationIsAObservation, \
    FactorsRelationDirectedFlowObservation, FactorTypesRelationUnidirectionalLinearTransformObservation
from backend.solving import BasicQuery


JsonStructureType = Dict[str, Optional[Union[type, "JsonStructureType"]]]


def objects_list_to_string(objects_list: List[object], object_type: type) -> str:
    json_string = ""

    if objects_list:
        if object_type is Hierarchy:
            # Just get the Hierarchy objects of type Taxon
            objects_list = [o for o in objects_list if o.hierarchy_type == Taxon]
            # Sort the list to show the "code lists" first
            objects_list.sort(key=lambda h: not h.is_code_list)

        for obj in objects_list:
            json_string += ", " if json_string else ""
            json_string += json.dumps(obj, cls=CustomEncoder)

    return json_string


def create_json_string_from_objects(objects: Dict[type, List[object]], json_structure: JsonStructureType) -> str:

    def json_structure_to_string(sections_and_types: JsonStructureType) -> str:
        json_string = ""
        for section_name, output_type in sections_and_types.items():
            json_string += ", " if json_string else ""
            json_string += f'"{section_name}": '
            if not isinstance(output_type, dict):
                json_string += "[" + objects_list_to_string(objects.get(output_type), output_type) + "]"
            else:
                json_string += "{" + json_structure_to_string(output_type) + "}"

        return json_string

    return "{" + json_structure_to_string(json_structure) + "}"


def export_model_to_json(state: State) -> str:

    json_structure: JsonStructureType = OrderedDict(
        {"Parameters": Parameter,
         "CodeHierarchies": Hierarchy,
         "Observers": Observer,
         "InterfaceTypes": FactorType,
         "InterfaceTypeConverts": FactorTypesRelationUnidirectionalLinearTransformObservation,
         "Processors": Processor,
         "Relationships": OrderedDict(
             {"PartOf": ProcessorsRelationPartOfObservation,
              "Upscale": ProcessorsRelationUpscaleObservation,
              "IsA": ProcessorsRelationIsAObservation,
              "DirectedFlow": FactorsRelationDirectedFlowObservation
              }
         )
         }
    )

    # Get objects from state
    query = BasicQuery(state)
    objects = query.execute(values_of_nested_dictionary(json_structure), filt="")

    json_dict = create_json_string_from_objects(objects, json_structure)

    print(json_dict)

    return json_dict

