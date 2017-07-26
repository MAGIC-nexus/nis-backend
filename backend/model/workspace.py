# -*- coding: utf-8 -*-
"""
* Registry of objects. Add, remove, search
* Support for high level operations: directly create and/or modify objects, calling the specification API. Create connections
"""
import logging

from typing import *
from backend.helper import create_dictionary

logger = logging.getLogger(__name__)


class Scope:
    """ The scope allows to assign a name to an entity """
    def __init__(self, name):
        self._name = name
        self._registry = create_dictionary()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __getitem__(self, name):
        if name in self._registry:
            return self._registry[name]
        else:
            return None

    def __setitem__(self, name: str, entity):
        if name not in self._registry:
            existing = True
        else:
            existing = False

        self._registry[name] = entity
        return existing

    def __delitem__(self, name):
        del self._registry[name]


class Namespace:
    def __init__(self):
        self.__scope = []  # List of scopes
        self.__current_scope = None
        self.__current_scope_idx = -1
        self.__rev_registry = create_dictionary()  # A registry of entity names to a list of scopes where they are mentioned

    # The registry will have a sequence of scopes (instead of nested scopes)
    # The scope initially will correspond to a transaction
    # When searching for names, the search will go from the most recent scope to the oldest
    def new_scope(self, name=None):
        """ Create a new scope """
        if self.__current_scope:
            raise Exception("Scopes cannot be nested. The current scope "+self.__current_scope.get_name()+" must be closed ('close_scope').")

        self.__current_scope = Scope()
        self.__scope.append(self.__current_scope)
        self.__current_scope_idx = len(self.__scope) - 1
        if not name:
            name = "Scope" + str(self.__current_scope_idx)
        self.__current_scope.set_name(name)

    def close_scope(self):
        self.__current_scope = None

    def set(self, name: str, entity):
        """ Set a named entity in the current scope """
        if self.__current_scope:
            if self.__current_scope.set(name, entity):
                if name in self.__rev_registry:
                    lst = self.__rev_registry[name]
                else:
                    lst = []
                lst.append(self.__current_scope_idx)
            else:
                logger.warning("'"+name+"' overwritten.")

    def get(self, name: str):
        """ Return the entity named "name". Return also the Scope in which it was found most recently """
        if name in self.__rev_registry:
            for scope_idx in reversed(self.__rev_registry[name]):
                if name in self.__scope[scope_idx]:
                    return self.__scope[scope_idx][name], self.__scope[scope_idx]
                else:
                    logger.error("The name '"+name+"' was expected in the scope ")


class WorkSpace:
    """ Keeps a registry of variable names and the objects behind them.
    
        It is basically a list of Namespaces. One is active by default.
        The others have a name. Variables inside these other Namespaces may be accessed using that 
        name then "::", same as C++
    """


if __name__ == '__main__':
    ms = Namespace()
