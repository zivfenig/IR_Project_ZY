# Building search engine for Wikipedia
## IR project on Wikipedia by Ziv Fenigstein and Yarden Tzaraf
In our project, we developed a search engine tailored to Wikipedia data. Using the BM25 similarity function, we optimized query processing time by employing multithreading techniques. As part of our preparatory work, we conducted essential computations such as building inverted indexes, calculating page ranks, and more, all of which were stored in a Google Cloud Platform bucket.

The repository comprises four key files:
* create_indices_project_gcp.ipynb - Used for create the Invrted Indexes as well as create useful calculations for the implementation of BM25 and store them in our GCP bucket
* inverted_index_gcp.py - This fle initializes the inverted index and creates necessary objects for reading and writing relevant files to the appropriate path in the Google Cloud Platform (GCP) environment.
* search_forntend.py - The creation of a Flask app receive queries from clients and provide the most relevant answers.
* search_backend.py - The implementation of the search engine using BM25 similarity function.
