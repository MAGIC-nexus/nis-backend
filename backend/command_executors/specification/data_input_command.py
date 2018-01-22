import json

from backend.domain import IExecutableCommand, State, get_case_study_registry_objects
from backend.model.memory.musiasem_concepts import FactorTaxon, Observer, FactorInProcessorType, \
    Processor, \
    Factor, FactorQuantitativeObservation, QualifiedQuantityExpression, \
    FlowFundRoegenType, ProcessorsSet, HierarchiesSet, allowed_ff_types


class DataInputCommand(IExecutableCommand):
    """
    Serves to specify quantities (and their qualities) for observables
    If observables (Processor, Factor, FactorInProcessor) do not exist, they are created. This makes this command very powerfu: it may express by itself a MuSIASEM 1.0 structure

    """
    def __init__(self, name: str):
        self._name = name
        self._content = None

    def execute(self, state: "State"):
        """ The execution creates one or more Processors, Factors, FactorInProcessor and Observations
            It also creates "flat" Categories (Code Lists)
            It also Expands referenced Datasets
            Inserting it into "State"
        """
        def process_row(row):
            """
            Process a dictionary representing a row of the data input command. The dictionary can come directly from
            the worksheet or from a dataset.

            Implicitly uses "glb_idx"

            :param row: dictionary
            """
            # From "ff_type" extract: flow/fund, external/internal, incoming/outgoing
            # ecosystem/society?
            ft = row["ff_type"].lower()
            if ft == "int_in_flow":
                roegen_type = FlowFundRoegenType.flow
                internal = True
                incoming = True
            elif ft == "int_in_fund":
                roegen_type = FlowFundRoegenType.fund
                internal = True
                incoming = True
            elif ft == "int_out_flow":
                roegen_type = FlowFundRoegenType.flow
                internal = True
                incoming = False
            elif ft == "ext_in_flow":
                roegen_type = FlowFundRoegenType.flow
                internal = False
                incoming = True
            elif ft == "ext_out_flow":
                roegen_type = FlowFundRoegenType.flow
                internal = False
                incoming = False

            # CREATE FactorTaxon (if it does not exist). A Type of Observable
            ft_key = {"_type": "FactorTaxon", "_name": row["factor"]}
            ft = glb_idx.get(ft_key)
            if not ft:
                ft = FactorTaxon(row["factor"],  #
                                 parent=None, hierarchy=None,
                                 tipe=roegen_type,  #
                                 tags=None,  # No tags
                                 attributes=None,  # No attributes
                                 expression=None  # No expression
                                 )
                glb_idx.put(ft_key, ft)

            # CREATE processor (if it does not exist). An Observable
            p_key = {"_type": "Processor", "_name": row["processor"]}
            p_key.update(row["taxa"])
            # if "_processors_type" in p_key:
            #     del p_key["_processors_type"]
            p = glb_idx.get(p_key)
            if not p:
                p = Processor(row["processor"],
                              external=False,  # TODO Is it an external processor??
                              location=None,
                              tags=None,
                              attributes=row["taxa"]
                              )
                glb_idx.put(p_key, p)
                p_set.append(p)  # Appends the processor to the processor set ONLY if it does not exist
                p_set.append_attributes_codes(row["taxa"])

            # CREATE Observer, the ANALYST, for the ProcessorObservations
            author = state.get("_identity")
            if not author:
                author = "_anonymous"

            oer_key = {"_type": "Observer", "_name": author}
            oer = glb_idx.get(oer_key)
            if not oer:
                oer = Observer(author, "Current user" if author != "_anonymous" else "Default anonymous user")
                glb_idx.put(oer_key, oer)

            # Create ProcessorObservation
            # - For now, structural aspects of the processor
            # - Existence observation is not needed (the mere declaration without relations does not affect calculations)
            # -
            # TODO Use "level", "parent". If they are specified, create a ProcessorObservation and a relation
            # TODO po = ProcessorObservation(p, oer, tags=None, attributes=None)

            # CREATE Factor (if it does not exist). An Observable
            f_key = {"_type": "Factor", "_name": row["processor"] + "|" + row["factor"]}
            f = glb_idx.get(f_key)
            if not f:
                f = Factor(row["factor"],
                           p,
                           in_processor_type=FactorInProcessorType(external="", incoming=""),
                           taxon=ft,
                           location=None,
                           tags=None,
                           attributes=None)
                glb_idx.put(f_key, f)

            # CREATE the Observer of the Quantity
            oer_key = {"_type": "Observer", "_name": row["source"]}
            oer = glb_idx.get(oer_key)
            if not oer:
                oer = Observer(row["source"])
                glb_idx.put(oer_key, oer)

            # >>>>>>>>>>>>>>>>>>> FINALLY, ADD FactorObservation <<<<<<<<<<<<<<<<<<<<<<<<<<<<
            qq = QualifiedQuantityExpression(row["value"]+row["unit"])
            # Creates and Inserts also
            fo = FactorQuantitativeObservation.create_and_append(v=qq,
                                                                 factor=f,
                                                                 observer=oer,
                                                                 tags=None,
                                                                 attributes={"relative_to": row["relative_to"],
                                                                 "time": row["time"],
                                                                 "geolocation": row["geolocation"] if "geolocation" in row else None,
                                                                 "spread": row["uncertainty"] if "uncertainty" in row else None,
                                                                 "assessment": row["assessment"] if "assessment" in row else None,
                                                                 "pedigree": row["pedigree"] if "pedigree" in row else None,
                                                                 "pedigree_template": row["pedigree_template"] if "pedigree_template" in row else None,
                                                                 "comments": row["comments"] if "comments" in row else None
                                                                 }
                                                                 )

        # TODO Check semantic validity, elaborate issues
        issues = []

        glb_idx, p_sets, hh, datasets, mappings = get_case_study_registry_objects(state)

        if self._name not in p_sets:
            p_set = ProcessorsSet(self._name)
            p_sets[self._name] = p_set
        else:
            p_set = p_sets[self._name]

        # Store code lists (flat "hierarchies")
        for h in self._content["code_lists"]:
            # TODO If some hierarchies already exist, check that they grow (if new codes are added)
            hh.append(h, self._content["code_lists"][h])

        # Read each of the rows
        for r in self._content["rows"]:
            # Create processor, hierarchies (taxa) and factors
            # Check if the processor exists. Two ways to characterize a processor: name or taxa
            """
            The processor can have a name and/or a set of qualifications, defining its identity
            The name can be assumed to be one more qualifications concatenated
            Is assigning a name for all processors a difficult task? 
            * In the specification moment it can get in the middle
            * When operating it is not so important
            * If taxa allow obtaining the processor the name is optional
            * The benefit is that it can help reducing hierarchic names
            * It may help in readability of the case study
            * Automatic naming from taxa? From the columns, concatenate
            
            """
            # TODO If a row contains a reference to a dataset, expand it
            def dataset_referenced(r):
                return False

            if dataset_referenced(r):
                # TODO Obtain dataset
                # ds = obtain_dataset(r)
                # # TODO Prepare non changing fields from the row
                # fixed_dict = prepared_constant_fields(r)
                # for r2 in ds:
                #     r3 = r2.copy().update(fixed_dict)
                #     process_row(r3)
                pass
            else:
                process_row(r)

        return issues, None

    def estimate_execution_time(self):
        return 0

    def json_serialize(self):
        # Directly return the content
        return self._content

    def json_deserialize(self, json_input):
        # TODO Check validity
        issues = []
        if isinstance(json_input, dict):
            self._content = json_input
        else:
            self._content = json.loads(json_input)
        return issues
