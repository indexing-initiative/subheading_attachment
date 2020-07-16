# MEDLINE Database

The code requires MEDLINE data to be stored in a MySQL database (v5.6). The database schema is defined in ```create_database.sql``` and data should be loaded from the MEDLINE 2019 baseline (https://mbr.nlm.nih.gov/Download/Baselines/2019/). 2019 main heading and subheading definition files (desc2019.xml, qual2019.xml) are available at ftp://ftp.nlm.nih.gov/online/mesh/2019/xmlmesh/. The desc2019.xml file includes the allowed subheadings for each main heading.

## Key Points

- The code assumes that all database primary keys start from 1.
- In the schema, main headings are called mesh descriptors, subheadings are called mesh qualifiers, and main heading/subheading pairs (MeSH pairs) are called mesh topics.
- In our database, we chose to include an additional null subheading to record indexing of a main heading without a subheading.
- The mesh topics table includes all allowed MeSH pairs.
- Article indexing is recorded in the citation mesh topics table. In our database, we recorded indexing of all 76 subheadings (not just critical subheadings).

## Recreating the database id files

- Lists of pmids for the train, validation, and test sets are included in ../input_data/pmid_datasets.tar.gz.
- The code references articles by database primary keys and so these lists of pmids need to be converted into lists of database ids. The database ids will be unqiue for your database (see the files in ../input_data/db_id_datasets.tar.gz as an example).