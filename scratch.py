import matplotlib.pyplot as plt
import numpy as np

# Equilibrium details
equilibrium_price = 4
equilibrium_quantity = 60000

# Ceiling details
price_ceiling = 2.50
quantity_supplied_ceiling = 25000
quantity_demanded_ceiling = 62500

# Demand and Supply functions for plotting
def demand(P):
    return 100000 - 10000 * P

def supply(P):
    return 10000 * P

# Prices and quantities for plotting
prices = np.linspace(0, 6, 100)
demand_quantities = demand(prices)
supply_quantities = supply(prices)

# Plotting the market for gasoline
plt.figure(figsize=(10, 6))
plt.plot(demand_quantities, prices, label='Demand', color='blue')
plt.plot(supply_quantities, prices, label='Supply', color='red')

# Plot equilibrium point
plt.scatter(equilibrium_quantity, equilibrium_price, color='black', zorder=5)
plt.text(equilibrium_quantity, equilibrium_price + 0.1, 'Equilibrium (60,000, $4)', horizontalalignment='center')

# Plot price ceiling
plt.axhline(y=price_ceiling, color='green', linestyle='--', label='Price Ceiling ($2.50)')
plt.scatter(quantity_supplied_ceiling, price_ceiling, color='black', zorder=5)
plt.text(quantity_supplied_ceiling, price_ceiling + 0.1, '(25,000, $2.50)', horizontalalignment='center')
plt.scatter(quantity_demanded_ceiling, price_ceiling, color='black', zorder=5)
plt.text(quantity_demanded_ceiling, price_ceiling + 0.1, '(62,500, $2.50)', horizontalalignment='center')

# Annotate shortage
plt.annotate('Shortage (37,500 units)', xy=(45000, price_ceiling), xytext=(50000, 3),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlabel('Quantity of Gasoline')
plt.ylabel('Price per Gallon of Gasoline')
plt.title('Market for Gasoline')
plt.legend()
plt.grid(True)
plt.show()
