from typing import List, Union
from abc import ABCMeta, abstractmethod

from backend.models.musiasem_concepts import Processor, Observer, FactorType, Factor, \
    FactorQuantitativeObservation, FactorTypesRelationUnidirectionalLinearTransformObservation, \
    ProcessorsRelationPartOfObservation, ProcessorsRelationUndirectedFlowObservation, \
    ProcessorsRelationUpscaleObservation, FactorsRelationDirectedFlowObservation
from backend.model_services import get_case_study_registry_objects
from backend.model_services.workspace import State
from backend.common.helper import create_dictionary, PartialRetrievalDictionary


class IQueryObjects(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, object_classes: List, filt: Union[dict, str]) -> str:
        """
        Query state to obtain objects of types enumerated in "object_classes", applying a filter
        In general, interface to pass state, select criteria (which kinds of objects to retrieve) and
        filter criteria (which of these objects to obtain).

        :param object_classes: A list with the names/codes of the types of objects to obtain
        :param filt: A way of expressing a filter of objects of each of the classes to be retrieved
        :return:
        """
        pass


class BasicQuery(IQueryObjects):
    def __init__(self, state: State):
        self._state = state
        self._registry, self._p_sets, self._hierarchies, self._datasets, self._mappings = get_case_study_registry_objects(state)

    def execute(self, object_classes: List, filt: Union[dict, str]) -> str:
        requested = {}
        types = [Observer, Processor, FactorType, Factor,
                 FactorQuantitativeObservation, FactorTypesRelationUnidirectionalLinearTransformObservation,
                 ProcessorsRelationPartOfObservation, ProcessorsRelationUpscaleObservation,
                 ProcessorsRelationUndirectedFlowObservation,
                 FactorsRelationDirectedFlowObservation]
        for o_class in object_classes:
            for t in types:
                if (isinstance(o_class, str) and o_class.lower() == t.__name__.lower()) or \
                        (isinstance(o_class, type) and o_class == t):
                    requested[t] = None

        if Observer in requested:
            # Obtain All Observers
            oers = self._registry.get(Observer.partial_key())
            # Apply filter
            if "observer_name" in filt:
                oers = [o for o in oers if o.name.lower() == filt["observer_name"]]
            # Store result
            requested[Observer] = oers
        if Processor in requested:
            # Obtain All Processors
            procs = set(self._registry.get(Processor.partial_key()))
            # TODO Apply filter
            # Store result
            requested[Processor] = procs
        if FactorType in requested:
            # Obtain FactorTypes
            fts = set(self._registry.get(FactorType.partial_key()))
            # TODO Apply filter
            # Store result
            requested[FactorType] = fts
        if Factor in requested:
            # Obtain Factors
            fs = self._registry.get(Factor.partial_key())
            # TODO Apply filter
            # Store result
            requested[Factor] = fs
        if FactorQuantitativeObservation in requested:
            # Obtain Observations
            qqs = self._registry.get(FactorQuantitativeObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[FactorQuantitativeObservation] = qqs
        if FactorTypesRelationUnidirectionalLinearTransformObservation in requested:
            # Obtain Observations
            ftlts = self._registry.get(FactorTypesRelationUnidirectionalLinearTransformObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[FactorTypesRelationUnidirectionalLinearTransformObservation] = ftlts
        if ProcessorsRelationPartOfObservation in requested:
            # Obtain Observations
            pos = self._registry.get(ProcessorsRelationPartOfObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[ProcessorsRelationPartOfObservation] = pos
        if ProcessorsRelationUndirectedFlowObservation in requested:
            # Obtain Observations
            ufs = self._registry.get(ProcessorsRelationUndirectedFlowObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[ProcessorsRelationUndirectedFlowObservation] = ufs
        if ProcessorsRelationUpscaleObservation in requested:
            # Obtain Observations
            upss = self._registry.get(ProcessorsRelationUpscaleObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[ProcessorsRelationUpscaleObservation] = upss
        if FactorsRelationDirectedFlowObservation in requested:
            # Obtain Observations
            dfs = self._registry.get(FactorsRelationDirectedFlowObservation.partial_key())
            # TODO Apply filter
            # Store result
            requested[FactorsRelationDirectedFlowObservation] = dfs

        return requested


def get_processor_id(p: Processor):
    return p.name.lower()


def get_processor_ident(p: Processor):
    return p.ident


def get_processor_label(p: Processor):
    return p.name.lower()


def get_processor_unique_label(p: Processor, reg: PartialRetrievalDictionary):
    return p.full_hierarchy_names(reg)[0]


def get_factor_id(f_: Union[Factor, Processor], ft: FactorType=None):
    if isinstance(f_, Factor):
        return (f_.processor.name + ":" + f_.taxon.name).lower()
    elif isinstance(f_, Processor) and isinstance(ft, FactorType):
        return (f_.name + ":" + ft.name).lower()


def get_factor_type_id(ft: (FactorType, Factor)):
    if isinstance(ft, FactorType):
        return ":"+ft.name.lower()
    elif isinstance(ft, Factor):
        return ":" + ft.taxon.name.lower()


def processor_to_dict(p: Processor, reg: PartialRetrievalDictionary):
    return dict(name=get_processor_id(p), uname=get_processor_unique_label(p, reg), ident=p.ident)


def factor_to_dict(f_: Factor):
    return dict(name=get_factor_id(f_), rep=str(f_), ident=f_.ident)
