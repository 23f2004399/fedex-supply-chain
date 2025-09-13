# Notebook Folder Summary
This file contains summary of `copula_alt.ipynb`, `data-cleaning.ipynb`, `distribution-fitting.ipynb` and `notebook4c8e6c820a.ipynb`


## `copula_alt.ipynb`

- This notebook focuses on a specific statistical technique called **copulas**, which are used to model the dependency or relationship between different variables.
- It defines a function to compare two types of copulas: the **Gaussian copula** and the **Student-t copula**.
- It applies this comparison to the `order_profit_per_order` and `shipping_delay_days` variables to understand how financial impact and shipping delays are related.
- It calculates a correlation value for each copula model to quantify the strength of the relationship.
- It also creates a scatter plot to visually show the dependency structure between the two variables.
- It analyze the statistical relationship between key risk variables (profit and delay) and compare different mathematical models (copulas) to see which one best describes that relationship.


## `data-cleaning.ipynb`

- This notebook takes a raw supply chain dataset (`DataCoSupplyChainDataset.csv`) and prepares it for analysis.
- It performs several cleaning steps:
    - Fills in missing data for columns like `Customer Lname` and `Customer Zipcode`.
    - Drops the `Product Description` column because it is completely empty.
    - Converts text-based dates into a proper `datetime` format that can be used for calculations.
- It creates important new columns (a process called feature engineering):
    - `shipping_delay_days`: Calculates how many days late a shipment was.
    - `inter_arrival_time`: Calculates the time between a customer's orders.
- It renames many columns to make them shorter and easier to work with (e.g., `Benefit per order` becomes `benefit_per_order`).
- Finally, it saves the cleaned data into a new file called `cleaned_supply_chain_data.csv` and creates some basic charts to visualize the data.
- It cleans, processes, and enriches the original messy dataset to make it usable and ready for more advanced analysis and modeling.

---

## `distribution-fitting.ipynb`

- This notebook uses the cleaned data (`cleaned_supply_chain_data.csv`) from the previous step.
- Its goal is to find the best mathematical formula (a probability distribution) to describe the patterns in the key risk variables.
- It focuses on two variables: `inter_arrival_time` and `order_profit_per_order`.
- It uses a special library called `Fitter` to automatically test several common distributions (like `weibull_min`, `lognorm`, etc.) to see which one is the best match for the data.
- It generates plots that visually show how well each distribution fits the actual data.
- It creates a comparison table that ranks the distributions based on how well they fit, using a score called "Sum of Squared Errors."
- It identifies the most accurate probability distributions for key variables, which are essential for building a realistic Monte Carlo simulation of supply chain risks.

---

## `notebook4c8e6c820a.ipynb`

- This notebook is an early, exploratory version of the data analysis.
- It loads the raw supply chain dataset and performs initial checks to understand the data, such as looking at the first few rows (`.head()`) and checking for missing values (`.isnull()`).
- It identifies the same data quality issues that are handled in `data-cleaning.ipynb`, like the empty `Product Description` column.
- The notebook contains markdown cells with notes and thoughts about how to handle the data issues.
- It also performs some basic data exploration by creating histograms to see the distribution of different numerical columns and a correlation heatmap to see how variables relate to each other.
- It performs an initial investigation of the raw dataset to understand its structure, find any problems (like missing data), and discover basic patterns before starting a more formal and structured cleaning process.

---
