# Copula Choice Summary
This file contains summary of the `copula_choice.md` file


## `copula_choice.md`

- This document explains the decision-making process for choosing a statistical model, called a **copula**, for a supply chain risk simulation.
- It compares two types of models: the **Gaussian copula** and the **Student-t copula**.
- The final decision was to choose the **Student-t copula** because it fits the real-world data better.
- The most important reason for this choice is that the Student-t model can accurately show how extreme events happen together, like big financial losses and long shipping delays. This is called **tail dependence**.
- The analysis focused on three key variables: `order_profit_per_order` (Severity), `inter_arrival_time` (Frequency), and `shipping_delay_days` (Lead Time).
- The document shows that the Student-t model had better statistical scores (like AIC and BIC) than the Gaussian model.
- A key finding was that the tail dependence predicted by the Student-t model (**0.31**) was very close to the dependence observed in the actual data (**0.28**), proving it was a good choice.
- It formally justify why the Student-t copula is the right choice for the simulation, ensuring that the model is realistic and that the risk calculations are trustworthy.