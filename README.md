# Dataset Review: DataCo Supply Chain Dataset

## Overview

- **Total Records**: 180,519
- **Total Features**: 53
- **File Name**: `DataCoSupplyChainDataset.csv`

## Key Characteristics

- Contains transactional and operational data for a supply chain use case.
- Features cover aspects such as:
  - **Customer Information** (e.g., Name, Email, Segment, Location)
  - **Order Details** (e.g., Order ID, Order Date, Quantity, Price, Profit)
  - **Shipping Information** (e.g., Shipping Mode, Dates, Delivery Risk)
  - **Product Details** (e.g., Name, Category, Description)
  - **Geographical Data** (Latitude, Longitude, State, Country)

## Data Quality Summary

- **Missing Values**: 336,209 missing entries across various columns.
- **Duplicated Rows**: 0 duplicate records.
- **Data Types**:
  - `object (categorical/text)` columns: 24
  - `float64 (decimal/numeric)` columns: 15
  - `int64 (integer/numeric)` columns: 14

## Notes for Missing Values and Outliers

- **Missing Values**: Documented and addressed with imputation rules (e.g., fill with "Unknown", mode, or domain-consistent placeholder). Columns with excessive missingness may be dropped.
- **Outliers**: Flagged variables will be treated using winsorization (1st–99th percentile) or domain-appropriate clipping to avoid distortion in statistical analysis.
- **Invalid Dates**: None detected.
- **Overall Rule**: Cleaning will balance data integrity with completeness, using conservative imputation and robust outlier handling.

## Other Notes

- Several columns appear to have overlapping or redundant information (e.g., multiple customer/order/product IDs).
- Text and date-based fields might need preprocessing for analysis or modeling.
- The dataset is rich and suitable for various supply chain analytics tasks like:
  - Delivery risk analysis
  - Profit optimization
  - Customer segmentation
  - Sales forecasting
