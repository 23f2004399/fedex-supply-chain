# SRC Folder Summary
This file contains summary of the `simulation_model.py`, `updated_simulation_model.py` and the key differences between both the 

---

## `simulation_model.py`

- This file is a first version of a program to **simulate supply chain problems** using Monte Carlo simulation.  
- It reads probability distributions and copula details from a JSON file.  
- It works with distributions like `expon`, `weibull_min`, `lognorm`, `pareto`, and `norm`.  
- It can make random samples for:
  - `inter_arrival_time` (time between disruptions)  
  - `shipping_delay_days` (delay in days)  
  - `order_profit_per_order` (profit or loss per order)  
- It has a working **Gaussian copula** to connect variables.  
- It has a placeholder for **Student-t copula**, but it is not finished.  
- The simulation gives:
  - How many disruptions happen  
  - The costs of disruptions  
  - The delays caused  
- It also calculates a simple **Supply Chain Risk Index (SCRI)**.  
- At the end, it shows a chart of disruption costs using **Matplotlib and Seaborn**.  
- it study and measure the risks in supply chains, like delays, costs, and losses.  


---

##  `updated_simulation_model.py`

- This file starts with the first simulator but also adds **inventory management features**.  
- It still runs disruption simulations but also adds a real **Student-t copula** (using `rpy2` and R’s `copula` library).  
- A new class is added: **`InventorySimulator`**.  
  - Simulates daily inventory with random demand and delivery times.  
  - Tracks three types of costs:
    - Holding cost (for keeping stock)  
    - Shortage cost (when items run out)  
    - Ordering cost (when placing an order)  
  - Supports two inventory rules:
    - **(s, S) policy** → order new stock when inventory goes below `s`, and restock up to `S`.  
    - **Myopic policy** → a simple rule that orders based on the average demand.  
  - Can run many experiments with different settings and compare results.  
- The example at the bottom shows how to use both simulators together:
  - It takes fitted distributions from the supply chain simulator.  
  - Then uses them in the inventory simulator to test policies.  
- It not only model supply chain problems but also test which inventory policies work better when disruptions happen.  
---

### Key Differences Between the Two Files

| `simulation_model.py` | `updated_simulation_model.py` |
|------------------------|-------------------------------|
| Focuses only on supply chain disruption simulation. | Extends disruptions simulation and adds full inventory simulation. |
| Gaussian copula works, Student-t is only a placeholder. | Gaussian copula + working Student-t copula (via R). |
| Uses `json, numpy, scipy.stats, pandas, matplotlib, seaborn`. | Same imports, plus `sys, os`, and optional `rpy2` for Student-t copula. No plotting libraries. |
| Prints results and shows a distribution plot with Seaborn. | Prints results mainly as Pandas DataFrames (good for experiments, no plots). |
| Inventory simulation not included. | New `InventorySimulator` class with (s, S) and Myopic policies, cost tracking, and parameter experiments. |
| Runs Monte Carlo supply chain simulation, prints averages, and plots disruption costs. | Runs supply chain + inventory experiments, prints DataFrames with policy performance. |
| Evaluate supply chain risk (costs, delays, number of disruptions). | Evaluate both supply chain risks and test **inventory policies** to see which works better under disruptions. |

---