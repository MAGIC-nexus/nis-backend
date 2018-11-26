# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import collections
import pandasdmx
from magic_box import app, SDMXConcept
from magic_box.source_eurostat_bulk import get_eurostat_filtered_dataset_into_dataframe
from magic_box.file_processing_auxiliary import create_dictionary, cell_content_to_str
from magic_box.source_ssp import get_ssp_datasets, get_ssp_dataset, get_ssp_dimension_names_dataset
from anytree import Node, NodeMixin, RenderTree
from anytree.dotexport import RenderTreeGraph


def create_estat_request():
    # EuroStat datasets
    if 'CACHE_FILE_LOCATION' in app.config:
        cache_name = app.config['CACHE_FILE_LOCATION']
    else:
        cache_name = "/tmp/sdmx_datasets_cache"
    r = pandasdmx.Request("ESTAT", cache={"backend": "sqlite", "include_get_headers": True,
                                          "cache_name": cache_name})
    r.timeout = 180
    return r

estat = create_estat_request()

# ------------------------------------------------------------------------------------------------------------
# FUNCTIONS TO BE MODIFIED WHEN NEW SOURCES ARE NEEDED
# ------------------------------------------------------------------------------------------------------------


def get_supported_sources():
    return ["Eurostat", "SSP"]


def get_codes_all_statistical_datasets(source, sh_out, dataset_manager):
    """
    Obtain a list of datasets available from a source
    If no source is specified, all the sources are queried
    For each dataset, the source, the name, the periods available, an example command and a description are obtained

    :param source:
    :param sh_out: Output worksheet
    :param dataset_manager: It is a DataSourceManager
    :return: A Dataframe with the list of datasets
    """
    if dataset_manager:
        lst = dataset_manager.get_datasets(source)
        if sh_out:
            sh_out.cell(row=0 + 1, column=0 + 1).value = "Dataset ID"
            sh_out.cell(row=0 + 1, column=1 + 1).value = "Description"
            sh_out.cell(row=0 + 1, column=2 + 1).value = "URN"
            sh_out.cell(row=0 + 1, column=3 + 1).value = "Data Source"
        lst2 = []
        for r, k in enumerate(lst):
            if len(k) == 4:
                source = k[3]
            else:
                source = ""
            lst2.append((k[0], k[1], k[2], source))
            if sh_out:
                sh_out.cell(row=r + 1 + 1, column=0 + 1).value = k[0]
                sh_out.cell(row=r + 1 + 1, column=1 + 1).value = k[1]
                sh_out.cell(row=r + 1 + 1, column=2 + 1).value = k[2]
                sh_out.cell(row=r + 1 + 1, column=3 + 1).value = source
        return pd.DataFrame(data=lst2, columns=["Dataset ID", "Description", "URN", "Data Source"])
    else:
        if source.lower() == "eurostat":
            import xmltodict
            # Make a table of datasets, containing three columns: ID, description, URN
            # List of datasets
            xml = requests.get("http://ec.europa.eu/eurostat/SDMX/diss-web/rest/dataflow/ESTAT/all/latest")
            t = xml.content.decode("utf-8")
            j = xmltodict.parse(t)
            sh_out.cell(row=0 + 1, column=0 + 1).value = "Dataset ID"
            sh_out.cell(row=0 + 1, column=1 + 1).value = "Description"
            sh_out.cell(row=0 + 1, column=2 + 1).value = "URN"
            rr = 0
            for r, k in enumerate(j["mes:Structure"]["mes:Structures"]["str:Dataflows"]["str:Dataflow"]):
                for n in k["com:Name"]:
                    if n["@xml:lang"] == "en":
                        desc = n["#text"]
                        break
                if k["@id"][:3] != "DS-":
                    dsd_id = k["str:Structure"]["Ref"]["@id"]
                    sh_out.cell(row=rr + 1 + 1, column=0 + 1).value = k["@id"]
                    sh_out.cell(row=rr + 1 + 1, column=1 + 1).value = desc
                    sh_out.cell(row=rr + 1 + 1, column=2 + 1).value = k["@urn"]
                    rr += 1
                # print(dsd_id + "; " + desc + "; " + k["@id"] + "; " + k["@urn"])
        elif source.lower()=="ssp":
            ssp_ds = get_ssp_datasets()
            sh_out.cell(row=0 + 1, column=0 + 1).value = "Dataset ID"
            sh_out.cell(row=0 + 1, column=1 + 1).value = "Description"
            rr = 0
            for r in ssp_ds:
                sh_out.cell(row=rr + 1 + 1, column=0 + 1).value = r
                sh_out.cell(row=rr + 1 + 1, column=1 + 1).value = ssp_ds[r]
                rr += 1


def get_statistical_dataset_structure(source, dataset, sh_out=None, dataset_manager=None):
    """
    Obtain the DSD containing the dimensions, attributes, measures, code lists

    :param source:
    :param dataset:
    :param sh_out: If passed, the metadata is output into the worksheet "sh_out"
    :param dataset_manager: It is a DataSourceManager
    :return: List of tuples formed by the dimension names (for the header) and its code lists (with description). 
    Also, the dimensions, attributes and measures, in a tuple, ready to be stored in the "metadatasets" registry
    """
    dims = None
    attrs = None
    meas = None
    if dataset_manager:
        # Obtain DATASET: Datasource -> Database -> DATASET -> Dimension(s) -> CodeList (no need for "Concept")
        dset = dataset_manager.get_dataset_structure(source, dataset)
        # TODO Generate "dims", "attrs" and "meas" from "dset"
        dims = create_dictionary()  # Each dimension has a name, a description and a code list
        attrs = create_dictionary()
        meas = create_dictionary()
        for dim in dset.dimensions:
            if dim.is_measure:
                meas[dim.code] = None
            else:
                # Convert the code list to a dictionary
                if dim.code_list:
                    cl = dim.code_list.to_dict()
                else:
                    cl = None
                dims[dim.code] = SDMXConcept("dimension", dim.code, dim.is_time, "", cl)
    else:
        if source.lower() == "eurostat":
            refs = dict(references='all')
            dsd_response = estat.datastructure("DSD_" + dataset, params=refs)
            dsd = dsd_response.datastructure["DSD_" + dataset]
            metadata = dsd_response.write()
            # Dimensions and Attributes
            dims = create_dictionary()  # Each dimension has a name, a description and a code list
            attrs = create_dictionary()
            meas = create_dictionary()
            for d in dsd.dimensions:
                istime = str(dsd.dimensions.get(d)).split("|")[0].strip() == "TimeDimension"
                dims[d] = SDMXConcept("dimension", d, istime, "", None)
            for a in dsd.attributes:
                attrs[a] = SDMXConcept("attribute", a, False, "", None)
            for m in dsd.measures:
                meas[m] = None
            for l in metadata.codelist.index.levels[0]:
                first = True
                # Read code lists
                cl = create_dictionary()
                for m, v in list(zip(metadata.codelist.loc[l].index, metadata.codelist.loc[l]["name"])):
                    if not first:
                        cl[m] = v
                    else:
                        first = False

                if metadata.codelist.loc[l]["dim_or_attr"][0] == "D":
                    istime = str(dsd.dimensions.get(l)).split("|")[0].strip() == "TimeDimension"
                    dims[l] = SDMXConcept("dimension", l, istime, "", cl)
                else:
                    attrs[l] = SDMXConcept("attribute", l, False, "", cl)

            # dict_dim[dims[l].name] = [c for c in dims[l].code_list]

        elif source.lower() == "ssp":
            # Read file
            df = get_ssp_dataset(dataset)
            # Obtain unique values for Dimension columns
            dims = {}
            attrs = []
            meas = []
            # For each dimension, return the list of codes
            # TODO (the descriptions are the codes themselves for now. For regions and countries it can be decoded)
            for c in get_ssp_dimension_names_dataset(dataset):
                if c in df.index.names:
                    lst = sorted(df.index.get_level_values(c).unique())
                    cl = create_dictionary()
                    for k in lst:
                        k2 = cell_content_to_str(k)
                        cl[k2] = k2
                    dims[c] = SDMXConcept("dimension", c, False, "", cl)
                    # lst_dim.append((c, zip(lst, lst)))

    # Make a table of dimensions and code lists, containing three columns: dimension name, code, code_description
    if sh_out:
        sh_out.cell(row=0 + 1, column=0 + 1, value="Dimension name")
        sh_out.cell(row=0 + 1, column=1 + 1, value="Code")
        sh_out.cell(row=0 + 1, column=2 + 1, value="Code description")
    r = 1
    time_dim = False

    lst_dim = []

    for l in dims:
        lst_dim_codes = []
        if dims[l].istime:
            time_dim = True
        else:
            lst_dim.append((dims[l].name, lst_dim_codes))

        if dims[l].code_list:
            for c in dims[l].code_list:
                if sh_out:
                    sh_out.cell(row=r + 1, column=0 + 1, value=l + (" (TimeDimension)" if dims[l].istime else ""))
                    sh_out.cell(row=r + 1, column=1 + 1, value=c)
                    sh_out.cell(row=r + 1, column=2 + 1, value=dims[l].code_list[c])

                lst_dim_codes.append((c, dims[l].code_list[c]))

                r += 1
        else:
            if sh_out:
                sh_out.cell(row=r + 1, column=0 + 1, value=l + (" (TimeDimension)" if dims[l].istime else ""))
            r += 1
    if time_dim:
        lst_dim.append(("StartPeriod", None))
        lst_dim.append(("EndPeriod", None))

    return lst_dim, (dims, attrs, meas)


def get_statistical_dataset(source, dataset, dataset_params, dataset_manager=None):
    """
    Obtain a dataset given some parameters
    :param source: "Eurostat", "FAOSTAT" (TO BE DONE), "SSP" (Climate Change), ...
    :param dataset: name of the dataset to retrieve. To obtain a list, call "obtain_datasets"
    :param dataset_params: list of (key, value) pairs filtering the dataset to be obtained. The possible parameters depend
    :param dataset_manager: It is a DataSourceManager
    :return: pd.Dataframe containing the resulting dataset as a facts table, ready for OLAP analysis (like Pivot Table)
    """
    if source.lower() == "eurostat":
        method = 1
        if method == 1:
            try:
                df1 = get_eurostat_filtered_dataset_into_dataframe(dataset, update=False)
                return filter_dataset_into_dataframe(df1, dataset_params, True) # RETURN
            except:
                pass
        # CURRENTLY DISABLED vvvvv OLD METHOD (SOMETIMES DOES NOT WORK, BULK PREFERRED)
        params = {}
        if "StartPeriod" in dataset_params:
            params["StartPeriod"] = dataset_params["StartPeriod"]
            del dataset_params["StartPeriod"]
        if "EndPeriod" in dataset_params:
            params["EndPeriod"] = dataset_params["EndPeriod"]
            del dataset_params["EndPeriod"]

        # Convert list params into concatenated strings
        ds_params = {}
        for k in dataset_params:
            v = dataset_params[k]
            if isinstance(v, list):
                ds_params[k] = '+'.join(v)
            else:
                ds_params[k] = v
        # Then, GET the dataset (pandaSDMX)
        d = estat.get(resource_type="data", resource_id=dataset, key=ds_params, params=params)
        #if d.data.series and len(d.data.series) > 0:
        try:
            df = d.write(d.msg)
        except:
            df = None
        if isinstance(df, pd.DataFrame):
            # Convert to a table of facts, which could be processed by a PivotTable
            col_names = []
            c = 0
            if isinstance(df.columns, pd.MultiIndex):
                for c, d in enumerate(df.columns.names):
                    col_names.append(d)
                c += 1
            else:
                pass  # What to do in this case?

            if isinstance(df.index, pd.MultiIndex):
                for d in df.index.names:
                    col_names.append(d)
                    c += 1
            else:
                col_names.append(df.index.name)
                c += 1

            col_names.append("VALUE")

            data = np.zeros((df.shape[0]*df.shape[1], len(col_names))).astype(object)
            r = 0
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    # Get values for the columns
                    c = 0
                    if isinstance(df.columns, pd.MultiIndex):
                        for c, l in enumerate(df.columns.values[col]):
                            data[r, c] = str(l)
                        c += 1
                    else:
                        pass  # What to do in this case?

                    if isinstance(df.index, pd.MultiIndex):
                        for l in df.index.values[row]:
                            data[r, c] = str(l)
                            c += 1
                    else:
                        data[r, c] = str(df.index[row])
                        c += 1
                    # Value
                    data[r, c] = df.iloc[row, col]
                    r += 1
            # Create a Dataframe
            df = pd.DataFrame(data, columns=col_names)
            cn = df.columns[-1]
            # Convert "Value" column (the last column) to numeric
            df[cn] = df[cn].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            return df
        else:
            # ERROR: it did not return a Dataset
            return None
    elif source.lower()=="ssp":
        # Read file, Dimension values in LOWER CASE
        df = get_ssp_dataset(dataset, True)
        # Filter the dataset and return it
        return filter_dataset_into_dataframe(df, dataset_params)
    else:
        return None


# ---------------------------------------------------------------------------------------------------------------------

def convert_code_list_to_hierarchy(cl, as_list=False):
    """
    Receives a list of codes. Codes are sorted lexicographically (to include numbers).
    
    Two types of coding schemes are supported by assuming that trailing zeros can be ignored to match parent -> child
    relations. The first is uniformly sized codes (those with trailing zeros). The second is growing length codes.
     
    Those with length less than others but common prefix are parents
                                  
    :param cl: 
    :param as_list: if True, return a flat tree (all nodes are siblings, descending from a single root)
    :return: 
    """

    def can_be_child(parent_candidate, child_candidate):
        # Strip zeros to the right, from parent_candidate, and
        # check if the child starts with the resulting substring
        return child_candidate.startswith(parent_candidate.rstrip("0"))

    root = Node("")
    path = [root]
    code_to_node = create_dictionary()
    for c in sorted(cl):
        if as_list:
            n = Node(c, path[-1])
        else:
            found = False
            while len(path) > 0 and not found:
                if can_be_child(path[-1].name, c):
                    found = True
                else:
                    path.pop()
            if c.rstrip("0") == path[-1].name:
                # Just modify (it may enter here only in the root node)
                path[-1].name = c
                n = path[-1]
            else:
                # Create node and append it to the active path
                n = Node(c, path[-1])
                path.append(n)
        code_to_node[c] = n  # Map the code to the node

    return root, code_to_node


def filter_dataset_into_dataframe(in_df, filter_dict, eurostat_postprocessing=False):
    """
    Function allowing filtering a dataframe passed as input,
    using the information from "filter_dict", containing the dimension names and the list
    of codes that should pass the filter. If several dimensions are specified an AND combination
    is done

    :param in_df: Input dataset, pd.DataFrame
    :param filter_dict: A dictionary with the items to keep, per dimension
    :param eurostat_postprocessing: Eurostat dataframe needs special postprocessing. If True, do it
    :return: Filtered dataframe
    """

    # TODO If a join is requested, do it now. Add a new element to the INDEX
    # TODO The filter params can contain a filter related to the new joins

    start = None
    if "StartPeriod" in filter_dict:
        start = filter_dict["StartPeriod"]
    if "EndPeriod" in filter_dict:
        endd = filter_dict["EndPeriod"]
    else:
        if start:
            endd = start
    if not start:
        columns = in_df.columns  # All columns
    else:
        # Assume year, convert to integer, generate range, then back to string
        start = int(start)
        endd = int(endd)
        columns = [str(a) for a in range(start, endd + 1)]

    # Rows (dimensions)
    cond_acum = None
    for i, k in enumerate(in_df.index.names):
        if k in filter_dict:
            lst = filter_dict[k]
            if not isinstance(lst, list):
                lst = [lst]
            if len(lst) > 0:
                if cond_acum is not None:
                    cond_acum &= in_df.index.isin([l.lower() for l in lst], i)
                else:
                    cond_acum = in_df.index.isin([l.lower() for l in lst], i)
            else:
                if cond_acum is not None:
                    cond_acum &= in_df[in_df.columns[0]] == in_df[in_df.columns[0]]
                else:
                    cond_acum = in_df[in_df.columns[0]] == in_df[in_df.columns[0]]
    if cond_acum is not None:
        tmp = in_df[columns][cond_acum]
    else:
        tmp = in_df[columns]
    # Convert columns to a single column "TIME_PERIOD"
    if eurostat_postprocessing:
        if len(tmp.columns) > 0:
            lst = []
            for i, cn in enumerate(tmp.columns):
                df2 = tmp[[cn]].copy(deep=True)
                df2.columns = ["value"]
                df2["time_period"] = cn
                lst.append(df2)
            in_df = pd.concat(lst)
            in_df.reset_index(inplace=True)
            # Value column should be last column
            lst = [l for l in in_df.columns]
            for i, l in enumerate(lst):
                if l == "value":
                    lst[-1], lst[i] = lst[i], lst[-1]
                    break
            in_df = in_df.reindex_axis(lst, axis=1)
            return in_df
        else:
            return None
    else:
        tmp.reset_index(inplace=True)
        if len(tmp.columns) > 0:
            return tmp
        else:
            return None


def map_codelists(src, dst, corresp, dst_tree=False) -> (list, set):
    """
    Obtain map of two code lists
    If the source is a tree, children of a mapped node are assigned to the same mapped node
    The same source may be mapped more than once, to different nodes
    The codes from the source not mapped, are stored in "unmapped"
     
    :param src: source full code list
    :param dst: destination full code list
    :param corresp: list of tuples with the correspondence
    :param dst_tree: Is the dst code list a tree?
    :return: List of tuples (source code, target code), set of unmapped codes
    """

    def assign(n: str, v: str):
        """
        Assign a destination code name to a source code name
        If the source has children, assign the same destination to children, recursively
        
        :param n: Source code name 
        :param v: Destination code name
        :return: 
        """
        mapped.add(n, v)
        if n in unmapped:
            unmapped.remove(n)
        for c in cn_src[n].children:
            assign(c.name, v)

    unmapped = set(src)
    r_src, cn_src = convert_code_list_to_hierarchy(src, as_list=True)
    if dst_tree:
        r_dst, cn_dst = convert_code_list_to_hierarchy(dst)
    else:
        cn_dst = create_dictionary()
        for i in dst:
            cn_dst[i] = None  # Simply create the entry
    mapped = create_dictionary(multi_dict=True)  # MANY TO MANY
    for t in corresp:
        if t[0] in cn_src and t[1] in cn_dst:
            # Check that t[1] is a leaf node. If not, ERROR
            if isinstance(cn_dst[t[1]], Node) and len(cn_dst[t[1]].children) > 0:
                # TODO ERROR: the target destination code is not a leaf node
                pass
            else:
                # Node and its children (recursively) correspond to t[1]
                assign(t[0], t[1])

    for k in sorted(unmapped):
        print("Unmapped: "+k)
    # for k in sorted(r):
    #     print(k+" -> "+r[k])

    # Convert mapped to a list of tuples
    # Upper case
    mapped_lst = []
    for k in mapped:
        for i in mapped.getall(k):
            mapped_lst.append((k.upper(), i.upper()))

    return mapped_lst, unmapped


def obtain_reverse_codes(mapped, dst):
    """
    Given the list of desired dst codes and an extensive map src -> dst,
    obtain the list of src codes
     
    :param mapped: Correspondence between src codes and dst codes
    :param dst: List of dst codes
    :return: List of src codes
    """
    src = set()
    for k in mapped:
        if k[1] in [d.upper() for d in dst]:
            src.add(k[0])

    return list(src)


# class MNode:
#     def __init__(self, name):
#         self.name = name
#         self.trees = {} # The node can be a member of more than one tree
#
#     def member_of(self, n: NodeMixin):
#         if n:
#             self.trees[n.root] = n
#             print("N: "+str(len(self.trees)))
#
#
# class A(NodeMixin):
#     def __init__(self, name: str, real_node: MNode, parent=None, taxonomic_rank=None):
#         self.parent = parent  # The tree making sentence
#         self.name = name
#         self.payload = real_node
#         if real_node:
#             real_node.member_of(self)
#         self.taxonomic_rank = taxonomic_rank


if __name__ == '__main__':
    # a = MNode("P1")
    # b = MNode("P2")
    # c = MNode("P3")
    # d = MNode("P4")
    # n1 = A("r1", None)
    # n2 = A("r11", a, n1)
    # n3 = A("r12", b, n1)
    # N1 = A("q1", None)
    # N2 = A("q2", b, N1)
    # N3 = A("q3", c, N2)
    # print(RenderTree(n1))
    # print(RenderTree(N1))
    cl = ["0000", "2000", "2112", "2115", "2116", "2117", "2130", "2230", "2310", "3000", "3105", "3106", "3191",
          "3214", "3215", "3220", "3235", "3244", "3250", "3260", "3270A", "3281", "3282", "3286", "3295", "4000",
          "4100", "4210", "4220", "4240", "5100", "5200", "5500", "5510", "5510", "5543", "55431", "55432"]
    cl2 = ["Heat", "Electricity", "Fuel"]
    m = [("2000", "Heat"), ("2116", "Fuel"), ("3000", "Electricity"), ("4000", "Fuel"), ("5100", "Fuel"), ("5543", "Electricity")]
    # r = convert_code_list_to_hierarchy(cl)
    # for pre, _, node in RenderTree(r):
    #     print("%s%s" % (pre, node.name))
    map_codelists(cl, cl2, m)