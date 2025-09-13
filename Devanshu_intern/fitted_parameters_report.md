# Output Folder Summary
This file contains summary of the `fitted_parameters.json`, `fitted_parameters_updated.json` and the key differences between both them

---

## `fitted_parameters.json`

- This is a configuration file, written in JSON format, that holds the statistical settings for a simulation.
- It defines the probability distributions for three key variables in a supply chain:
  - `inter_arrival_time` (modeled with a Weibull distribution)
  - `order_profit_per_order` (modeled with a Weibull distribution)
  - `shipping_delay_days` (modeled with an Exponential distribution)
- It also includes a section to model how variables relate to each other using a **gaussian copula**.
- This Gaussian copula specifically describes the relationship between `order_profit_per_order` and `shipping_delay_days`.
- It provide a complete set of initial parameters for the supply chain simulation, defining both how individual events behave and how two of those events are correlated in a relatively simple way.

## `fitted_parameters_updated.json`

- This is an updated JSON configuration file that focuses only on defining the dependency between simulation variables.
- Unlike the first file, it does not contain the individual distributions for each variable.
- It introduces a more advanced dependency model called a `student_t copula`. This type of model is better at capturing situations where extreme events happen together.
- This copula models the relationship between three variables: `order_profit_per_order`, `shipping_delay_days`, and `inter_arrival_time`.
- It includes a special parameter called `degrees_of_freedom`, which is unique to the Student-t copula.
- It upgrade the simulation's dependency model to be more realistic and to include an additional variable (inter_arrival_time) in the correlation structure.


### Key Differences Between the Two Files

| `fitted_parameters.json` | `fitted_parameters_updated.json` |
| --- | --- |
| Contains both individual variable distributions and a copula. | Contains only the copula definition. |
| Uses a simpler **Gaussian** copula. | Uses a more advanced **Student-t** copula. |
| Models the dependency between 2 variables: (**order_profit_per_order**, **shipping_delay_days**). | Models the dependency between **3 variables** (**order_profit_per_order**, **shipping_delay_days**, **inter_arrival_time**). |
| Has a 2x2 correlation matrix. | Has a 3x3 correlation matrix and **degrees_of_freedom**. |
| Represents a baseline or initial model. | Represents a more sophisticated model, better for capturing extreme risk. |

