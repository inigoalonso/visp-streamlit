import math
import simpy
import simpy.util
import settings as s

# Input variables

# FullHorizontalFOV = 10
# FullVerticalFOV = 4
# mirrorSize = 200
# volume = 4

# Calculate engineering variables

## calculate % FoV

def calculateFoV(FullHorizontalFOV,FullVerticalFOV):
    """calculate a metric for the FOV of the HUD"""
    FoV = (FullHorizontalFOV * FullVerticalFOV * 100) / (67 * 20)
    return FoV

## cost model

def calculateCostVehicle(cost_mirror_unit, mirrorSize, cost_vehicle_without_HUD, assembly_cost):
    """calculate the cost of the vehicle"""
    cost_mirror = cost_mirror_unit * mirrorSize
    cost_vehicle = cost_mirror + cost_vehicle_without_HUD + assembly_cost # euro
    return cost_vehicle

## weight model

def calculateWeightVehicle(volume, weight_vehicle_without_HUD):
    """calculate the weight of the vehicle"""       
    weight_hud = 0.1 * volume  # kg
    weight_vehicle = weight_hud + weight_vehicle_without_HUD # kg
    return weight_vehicle

## fuel consumption model

def calculateFuelConsumption(volume, weight_vehicle_without_HUD):
    """calculate the fuel consumption of the vehicle"""
    fuel_consumpt = calculateWeightVehicle(volume, weight_vehicle_without_HUD) * math.exp(-4.7)  # km / liter
    return fuel_consumpt

def calculateFuelCost(volume, weight_vehicle_without_HUD, kilometers_year, cost_fuel):
    """calculate the fuel cost of the vehicle"""
    fuel_consumpt = calculateFuelConsumption(volume, weight_vehicle_without_HUD)
    fuel_cost_year = (kilometers_year/fuel_consumpt) * cost_fuel # keuro / year
    return fuel_cost_year

## calculate demand, copied from demand.py

def calculateDemand(FullHorizontalFOV, FullVerticalFOV, volume, weight_vehicle_without_HUD, cost_fuel, person_height, price_vehicle, year):
    """calculate the demand of the vehicle"""
    Lin_no__1 = (400.389457364428
                + -0.0496637629531165 * (calculateFoV(FullHorizontalFOV, FullVerticalFOV))
                + 0.0438458326033747  * (calculateFuelConsumption(volume, weight_vehicle_without_HUD))
                + 3.53646955685314 * (cost_fuel * 1000)
                + -0.0958055046356103 * (person_height)
                + 0.0000987106990985412 * (price_vehicle * 1000)
                + -0.193495221339535 * (year)
                )
    Prob_yes_1 = 1 / (1 + math.exp(Lin_no__1)) # demand
    return Prob_yes_1


# create class for Surplus Value Model

class SV:

    # configuration

    cost_mirror_unit = 5 / 1000 # keuro/mm^2
    price_vehicle = 35000 / 1000 # k€uro/vehicle
    assembly_cost = 556.20 / 1000 # k€euro/vehicle
    cost_vehicle_without_HUD = 20000 / 1000  # k€uro /vehicle
    weight_vehicle_without_HUD = 1800 # kg /vehicle
    cost_fuel = 1.9 / 1000 # kEuro / liter
    kilometers_year = 10000 # km/year
    market_window = 10 # years
    sales_distr = [0.2, 0.4, 0.8, 1, 1, 1, 1, 1, 0.8, 0.4, 0.2]
    theoretical_demand = 70000 # vehicles/year
    operation_time = 10 # years
    development_time = 0 # years
    production_time = 0.1 # years
    sales_time = 0.1 # years
    year = 2035 #year
    person_height = 180 # cm

    # configuration of simulation

    discrete_time = 0.01
    discount_rate = 0.08
    simulation_duration = 30 # years
    time_between_counting = 1 # year

    def __init__(self, env, FullHorizontalFOV, FullVerticalFOV, mirrorSize, volume):

        self.env = env

        self.FullHorizontalFOV = FullHorizontalFOV
        self.FullVerticalFOV = FullVerticalFOV
        self.mirrorSize = mirrorSize
        self.volume = volume

        self.i = 0

       # initializing costs and revenues

        self.development_cost = 0
        self.production_cost = 0
        self.integration_cost = 0
        self.operation_cost = 0

        self.total_development_cost = 0
        self.total_production_cost = 0
        self.total_operation_cost = 0
        self.total_revenue = 0
        self.total_cost = 0

        self.cost = 0
        self.revenue = 0
        self.net_revenue = - self.cost
        self.NPV = self.net_revenue / (1 + self.discount_rate) ** env.now
        self.cumulativeNPV = self.NPV

        # initializing data for plot
        self.obs_time = [env.now]
        self.cashflow_level = [self.cumulativeNPV]

        # initializing counters for the different phases

        self.total_demand = 0
        self.in_development = 0
        self.in_production = 0
        self.in_sale = 0
        self.total_produced = 0
        self.in_operation = 0
        self.EoL = 0

        # start the business process every time an option is defined, start the development,
        # if development time != 0, start the market delayed, else start instantaneously

        # self.action = start_delayed(env, self.market(market_window, time_between_counting), self.development_time)

        self.action = env.process(self.lifecycle())

        self.action = env.process(self.observe())

        self.action = env.process(self.calculate())

    def lifecycle(self):

        self.in_development = 1

        yield self.env.timeout(self.development_time)

        self.in_development = 0

        self.action = self.env.process(self.market())

    def market(self):

        while self.env.now < self.market_window:

            self.action = self.env.process(self.business_run())

            yield self.env.timeout(self.time_between_counting)

            self.i += 1

    def business_run(self):

        # set vehicle demand

        self.current = math.ceil(s.theoretical_demand 
                                * calculateDemand(self.FullHorizontalFOV, 
                                                  self.FullVerticalFOV, 
                                                  self.volume, 
                                                  self.weight_vehicle_without_HUD, 
                                                  self.cost_fuel, 
                                                  self.person_height, 
                                                  self.price_vehicle, 
                                                  self.year)
                                * s.sales_distr[self.i])   # calculation of current demand

        # set vehicles in production

        self.in_production += self.current

        yield self.env.timeout(s.production_time)

        # after production_time, vehicles are in sale

        self.in_production -= self.current

        self.in_sale += self.current

        yield self.env.timeout(s.sales_time)

        # after sales time, set vehicles in use

        self.in_sale -= self.current

        self.in_operation += self.current

        yield self.env.timeout(s.operation_time)

        # after operation time, set vehicles out of life

        self.in_operation -= self.current

        self.EoL += self.current

    def observe(self):
        while True:

            self.obs_time.append(self.env.now)
            self.cashflow_level.append(self.cumulativeNPV)

            #print (self.revenue)

            #if self.cumulativeNPV < 0:
                #self.payback_period = env.now

            yield self.env.timeout(s.discrete_time)

    def calculate(self):
        """function for cost, revenue and NPV calculations"""

        while True:
            # all calculations are scaled down by the discrete time

            self.production_cost = (self.in_production
                                   * (calculateCostVehicle(self.cost_mirror_unit, 
                                                           self.mirrorSize, 
                                                           self.cost_vehicle_without_HUD, 
                                                           self.assembly_cost))
                                   * s.discrete_time)  # the total cost is "spread" along the production time

            self.revenue = self.in_sale * (s.price_vehicle) * s.discrete_time

            #self.operation_cost = self.in_operation * ((kilometers_year/fuel_consumpt) * cost_fuel) * discrete_time

            self.total_revenue += self.revenue

            # sum of cost and revenue and caluclation of NPV
            self.cost = self.production_cost #+ self.operation_cost
            self.total_cost -= self.cost
            self.net_revenue = self.revenue - self.cost
            self.NPV = self.net_revenue / (1 + s.discount_rate) ** self.env.now
            self.cumulativeNPV += self.NPV

            yield self.env.timeout(s.discrete_time)


# run Surplus Value of alternative
# launch environment

def SurplusValue(FullHorizontalFOV, FullVerticalFOV, mirrorSize, volume):
    FullHorizontalFOV = FullHorizontalFOV
    FullVerticalFOV = FullVerticalFOV
    mirrorSize = mirrorSize
    volume = volume
    env = simpy.Environment()
    #simpy.util.start_delayed(env, simulation_run(env), delay=1)
    hud = SV(env, FullHorizontalFOV, FullVerticalFOV, mirrorSize, volume)
    #print("Starting simulation.")
    env.run(until=s.simulation_duration)
    #print("End of simulation.")

    surplus_value = hud.cumulativeNPV/1000 # converted in million euro
    lifecycle_costs = -hud.total_cost/1000

    #print(surplus_value)
    return surplus_value

if __name__ == '__main__':
    SurplusValue(10, 4, 200, 4)


# def SurplusValue():
#     return surplus_value