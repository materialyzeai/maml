---
layout: default
title: maml.data.md
nav_exclude: true
---

# maml.data package

Get data from various sources.

## *class* maml.data.MaterialsProject(api_key: str | None = None)

Bases: `BaseDataSource`

Query the Materials Project for Data.

### get(criteria: str | dict, properties: list[str])

* **Parameters**
  * **criteria** (*str*\* or \**dict*) – Criteria for query
  * **properties** (*list*) – Properties to be queried.
* **Returns**
  pandas DataFrame

## *class* maml.data.URLSource(fmt: str = ‘csv’, read_kwargs=None)

Bases: `BaseDataSource`

Load raw data from a URL, e.g., figshare.

### get(url: str)

Get url data source.

* **Parameters**
  **url** – URL to obtain raw data from.
* **Returns**
  pd.DataFrame

## maml.data._mp module

Materials Project DataSource.

### *class* maml.data._mp.MaterialsProject(api_key: str | None = None)

Bases: `BaseDataSource`

Query the Materials Project for Data.

#### get(criteria: str | dict, properties: list[str])

* **Parameters**
  * **criteria** (*str*\* or \**dict*) – Criteria for query
  * **properties** (*list*) – Properties to be queried.
* **Returns**
  pandas DataFrame

## maml.data._url module

Get data from url.

### *class* maml.data._url.FigshareSource(fmt: str = ‘csv’, read_kwargs=None)

Bases: `URLSource`

Load data from figshare.

#### get(file_id: str)

Get data from Figshare
:param file_id: file id.

* **Returns**
  data frame

### *class* maml.data._url.URLSource(fmt: str = ‘csv’, read_kwargs=None)

Bases: `BaseDataSource`

Load raw data from a URL, e.g., figshare.

#### get(url: str)

Get url data source.

* **Parameters**
  **url** – URL to obtain raw data from.
* **Returns**
  pd.DataFrame