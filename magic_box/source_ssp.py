import requests
from magic_box import app
import pandas as pd


def test_download_excel_file():
    # Login
    r = requests.post("https://tntcat.iiasa.ac.at/SspDb/dsd?Action=loginform",
                      data={'usr': 'rnebot@itccanarias.org', 'pwd': 'pouteldi'})
    # Obtain the JSESSIONID cookie
    ck = r.request.headers["cookie"].split("=")
    # REPEAT
    # Download file. Dimensions: region, scenario, source, model, variable
    r = requests.get("https://tntcat.iiasa.ac.at/SspDb/dsd?Action=exceltable&regions=R266&scenarios=R911,R961,R941,R971,R1526,History,S351,S352,S353&variable=V1&suffix=xls", cookies={"JSESSIONID": ck[1]})
    with open("/home/rnebot/test.xls", "wb") as f:
        f.write(r.content)


def get_ssp_datasets():
    return {"regions": "SSP by regions (6) and global", "countries": "SSP by country"}


def get_ssp_dimension_names_dataset(dataset_name):
    if dataset_name.lower()=="regions":
        return ["source_id", "model", "scenario", "spatial", "temporal", "variable"]
    elif dataset_name.lower()=="countries":
        return []


def get_ssp_dataset(dataset_name: str, convert_dimensions_to_lower=False):
    """
    Read into a Dataframe the requested SSP dataset
    
    :param dataset_name: either "regions" or "countries"
    :param convert_dimensions_to_lower: True is to convert dimensions to lower case
    :return: A DataFrame with the dataset 
    """
    # Read some configuration knowing where the ssp is stored
    if "SSP_FILES_DIR" in app.config:
        base_path = app.config["SSP_FILES_DIR"]
    else:
        base_path = "/home/rnebot/GoogleDrive/AA_MAGIC/Data/SSP/"
    fname = base_path + "/" + dataset_name.lower() + ".csv"
    # Read the file into a Dataframe
    df = pd.read_csv(fname)
    dims = get_ssp_dimension_names_dataset(dataset_name)
    # Convert to lower case if needed
    if convert_dimensions_to_lower:
        for d in dims:
            df[d] = df[d].astype(str).str.lower()
    # Index the dataframe on the dimensions
    df.set_index(dims, inplace=True)

    # Return the Dataframe
    return df
