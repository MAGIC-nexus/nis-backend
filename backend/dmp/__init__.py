"""
A package in which functionality specific to MAGIC project can be found
The rest of the project should not depend on MAGIC, using pure MuSIASEM
and of course its evolution inside the project (but transcending it)

* Nextcloud.
  - NIS -> Nextcloud. Export a case study to a Nextcloud folder:
    - Link a Nextcloud folder to a NIS case study.
    - Dublin Core. "convert_generator_to_dublin_core"
    - MSM JSON. The NIS format for the representation of sequences of commands. "convert_generator_to_json_generator"
  - Nextcloud -> NIS. ---
* Zenodo.
  - NIS -> Zenodo. Upload the information in Nextcloud case study to Zenodo. Update it if already exists.
  - Zenodo -> NIS. After uploading, automatically modify DOI, create a new version.
* Geoserver.
  - NIS -> Geonetwork. Add case study as dataset in Geonetwork. Only if it is geolocated
"""