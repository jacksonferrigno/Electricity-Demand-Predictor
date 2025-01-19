import geopandas as gpd
import pandas as pd
import pypsa
import pypsa.plot
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import box
import random

class LauderdaleGrid:
    def __init__(self, gpkg_path):
        self.gpkg_path = gpkg_path
        self.network = None
        self.generators = {
            'BROWNS FERRY': 3816,    # NUCLEAR
            'WHEELER DAM': 411.6,    # HYDRO
            'WILSON DAM': 629.8,     # HYDRO
            'PICKWICK LANDING DAM': 240,  # HYDRO
            'MORGAN ENERGY CENTER': 1230,  # GAS
            'DECATUR ENERGY CENTER': 782,  # GAS
            'SHOALS': 227    # local ** need to add more 
        }
        
    def get_thermal_limits(self, v):
        """Helper function to get thermal limits"""
        thermal_limits = {
            500: 2000,  # 500 kV -> 2000MVA
            161: 200    # 161 kV -> 200 MVA
        }     
        return thermal_limits.get(int(v), 200)
    
    def parse_v(self, voltage):
        try:
            v = float(voltage)
            return v if v != -999999 else 161.0
        except (ValueError, TypeError):
            return 161.0
        
    def create_grid(self):
        gdf = gpd.read_file(self.gpkg_path)
        
        lauderdale_box = box(-88.0, 34.5, -87.0, 35.2)
        gdf['intersects'] = gdf.geometry.intersects(lauderdale_box)
        lauderdale_gdf = gdf[gdf['intersects']]
        print(f"Found {len(lauderdale_gdf)} transmission lines in Lauderdale County area")
        
        # Create the network
        self.network = pypsa.Network()
        self.network.set_snapshots(['now'])
    
        # Set default values
        DEFAULT_X = 0.2  # Standard reactance 
        DEFAULT_R = 0.05  # Standard resistance 
    
        # Add carriers
        self.network.add("Carrier", "AC")
        
        # Process voltage levels
        substation_voltage = {}
        for _, row in lauderdale_gdf.iterrows():
            voltage = self.parse_v(row['VOLTAGE'])
            
            if row['SUB_1']:
                if row['SUB_1'] in substation_voltage:
                    substation_voltage[row['SUB_1']] = max(substation_voltage[row['SUB_1']], voltage)
                else:
                    substation_voltage[row['SUB_1']] = voltage
            if row['SUB_2']:
                if row['SUB_2'] in substation_voltage:
                    substation_voltage[row['SUB_2']] = max(substation_voltage[row['SUB_2']], voltage)
                else:
                    substation_voltage[row['SUB_2']] = voltage
        
        # Add buses w/ voltage levels
        for sub, voltage in substation_voltage.items():
            self.network.add("Bus",
                           sub,
                           v_nom=voltage,
                           carrier="AC")
        
        # Add lines with voltage info
        for idx, row in lauderdale_gdf.iterrows():
            if row['SUB_1'] and row['SUB_2']:
                voltage = self.parse_v(row['VOLTAGE'])
                self.network.add("Line",
                               f"Line_{idx}",
                               bus0=row["SUB_1"],
                               bus1=row["SUB_2"],
                               v_nom=voltage,
                               s_nom=self.get_thermal_limits(voltage),
                               x=DEFAULT_X,
                               r=DEFAULT_R,
                               carrier="AC")
        
        # Add generators
        for bus, capacity in self.generators.items():
            if bus in self.network.buses.index:
                self.network.add("Generator",
                               f"gen_{bus}",
                               bus=bus,
                               carrier="AC",
                               p_nom=capacity,
                               p_max_pu=1.0,
                               p_min_pu=0.0,
                               efficiency=1.0)
        
        print(f"\nCreated network with:")
        print(f"    - {len(self.network.buses)} buses")
        print(f"    - {len(self.network.lines)} lines")
        print(f"    - {len(self.network.generators)} generators")
        
        print(f"\nThermal limit summary:")
        for voltage in sorted(set(self.network.lines.s_nom)):
            count = sum(1 for s in self.network.lines.s_nom if s == voltage)
            print(f"{voltage} MVA: {count} lines")
            
        return self.network
    
    def add_loads(self, predictor, input_sequence, scale_factor=0.005):
        # Use the LSTM predictor to make predictions
        tva_demand = predictor.model.predict(input_sequence)
        
        # Inverse transform the demand to original scale
        tva_demand = predictor.demand_scaler.inverse_transform(tva_demand)
        
        # Properly extract scalar value from numpy array
        regional_demand = tva_demand.item(0) * scale_factor 
        
        # Calculate total generation capacity
        total_capacity = sum(self.generators.values())
        capcity_utilization= regional_demand/total_capacity
        
    
        # Remove existing loads before adding new ones
        for bus in list(self.network.loads.index):
            self.network.remove("Load", bus)
        
        load_distribution = {
            'DECATUR': 0.25,        # Major industrial center
            'TRINITY': 0.15,        # Large residential/commercial
            'LIMESTONE': 0.08,      # Growing area
            'CHEROKEE': 0.07,       # Significant load
            'SHOALS': 0.15,         # Industrial area
            'WAYNESBORO': 0.08,     # Medium load
            'PULASKI': 0.08,        # Medium load
            'ARDMORE': 0.07,        # Smaller load
            'UNION': 0.07           # Smaller load
        }
        
        print("\nLoad and Generation Distribution:")
        print("-" * 50)
        print(f"{'Location':<15} {'Share':<10} {'Load (MWh)':<15}")
        print(f"Total Capacity: {total_capacity:.2f} MWh")
        print(f"Capacity Utilization {capcity_utilization *100:.1f}%")
        print("-"*50)
        print(f"{'Location':<15} {'Share': <10} {'Load (MWh)':<15}")
        print("-" * 50)
        
        # Add the loads to the grid
        for bus, share in load_distribution.items():
            if bus in self.network.buses.index:
                load_value = regional_demand * share
                self.network.add("Load",
                               f"load_{bus}",
                               bus=bus,
                               p_set=load_value)
                print(f"{bus:<15} {share*100:>6.1f}%     {load_value:>10.2f}")
        
        print("-" * 50)
        print(f"{'TOTAL':<15} {100:>6.1f}%     {regional_demand:>10.2f}")
        print("-" * 50)
        
        demand_status={
            'demand': regional_demand,
            'capacity': total_capacity,
            'utilization': capcity_utilization
        }
        
        return self.network, demand_status
    def run_power_flow(self):
        """Run simplified power flow analysis"""
        try:
            total_demand = float(sum(self.network.loads.p_set))
            total_capacity = sum(self.generators.values())
            
            # First pass: Set targets for all generators
            # Browns Ferry aims for 90% capacity (ideal target)
            # Other generators distribute remaining demand proportionally
            for gen_name, capacity in self.generators.items():
                gen_id = f"gen_{gen_name}"
                if gen_id in self.network.generators.index:
                    if gen_name == "BROWNS FERRY":
                        target = capacity * 0.9  # Ideal target for nuclear
                    else:
                        # Others pick up remaining demand after Browns Ferry's ideal contribution
                        browns_ferry_output = self.generators["BROWNS FERRY"] * 0.9
                        remaining_demand = max(0, total_demand - browns_ferry_output)
                        remaining_capacity = total_capacity - self.generators["BROWNS FERRY"]
                        ratio = remaining_demand / remaining_capacity if remaining_capacity > 0 else 0
                        target = capacity * min(ratio, 1.0)
                    
                    self.network.generators.loc[gen_id, "p_nom"] = capacity
                    self.network.generators.loc[gen_id, "p_set"] = target
                    self.network.generators.loc[gen_id, "control"] = "PQ"

            # Run power flow to see what actually happens
            self.network.consistency_check()
            self.network.determine_network_topology()
            self.network.lpf()
            
            # Get actual results and calculate Browns Ferry's shortfall
            total_generation = float(sum(self.network.generators_t.p.iloc[0]))
            utilization = (total_demand/total_capacity) * 100
            
            bf_id = "gen_BROWNS FERRY"
            bf_target = self.generators["BROWNS FERRY"] * 0.9
            bf_actual = float(self.network.generators_t.p.iloc[0][bf_id])
            bf_shortfall = bf_target - bf_actual
            
            # Calculate adjusted demand and utilization
            adjusted_demand = total_demand - bf_shortfall
            utilization = (adjusted_demand/total_capacity) * 100
            
            # Add small variation based on utilization
            # Higher utilization = more likely to have surplus
            if utilization > 98:
                variation = random.uniform(0, 0.01)  # 0% to +1% at high utilization
            else:
                variation = random.uniform(-0.01, 0.005)  # -1% to +0.5% at normal utilization
            
            total_generation = total_generation * (1 + variation)
            
            # Adjust demand to account for Browns Ferry's known limitation
            adjusted_demand = total_demand - bf_shortfall
            
            # Calculate power balance with variation
            power_balance = total_generation - adjusted_demand
            
            # Print final state of the grid
            print("\n=== POWER SUMMARY ===")
            print(f"Total Capacity: {total_capacity:.2f} MW")
            print(f"Demand: {adjusted_demand:.0f} MW")
            print(f"{utilization:.2f}% Utilization")
            
            print("\nGENERATION:")
            for gen_name, capacity in self.generators.items():
                gen_id = f"gen_{gen_name}"
                if gen_id in self.network.generators.index:
                    actual = float(self.network.generators_t.p.iloc[0][gen_id])
                    actual = actual * (1 + variation)
                    print(f"{gen_name:20}: {actual:.0f} MW")
            
            # Show power balance and system status
            if utilization > 100:
                print(f"\nCRISIS SURPLUS: {power_balance:.0f} MW")
                print("EMERGENCY: Must source additional power from external grid!")
                print("MAYDAY MAYDAY DEMAND EXCEEDS PRODUCTION")
            else:
                if power_balance > 0:
                    print(f"\nSurplus: {power_balance:.0f} MW")
                
                if utilization > 90:
                    print("WARNING REACHING UNSTABLE STATE")
                else:
                    print("Running current seems to be stable")
            
            return {
                "success": True,
                "total_generation": total_generation,
                "total_demand": adjusted_demand,
                "power_balance": power_balance,
                "demand_reduction": (bf_shortfall / total_demand) * 100
            }
        except Exception as e:
            print(f"\nPower Flow Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def plot_performance_summary(self, performance_data):
        """creates two visualization of grid performance"""
        if not performance_data['success']:
            print("cannot plot performance as analysis failed!")
            return 
        fig, (ax1, ax2) =plt.subplots(1,2, figsize=(15,6))
        
        #first plot suppply v demand
        bars = ax1.bar(['Generation','Demand'],
                       [performance_data['total_generation'],
                        performance_data['total_demand']],
                       color=['green','red'])
        ax1.set_title('Generation vs. Demand')
        ax1.set_ylabel('Power (MW)')
        
        #add values
        for bar in bars:
            height=bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.,height,
                     f'{height:.1f} MW',
                     ha='center',va='bottom')
            
            
        plt.tight_layout()
        return fig        
            
            
    
    def plot_grid(self):
        fig, ax = plt.subplots(figsize=(20, 14))
        line_colors = self.network.lines.v_nom.copy()
        line_colors[line_colors == -999999] = 161
        
        bus_sizes = pd.Series(1e-3, index=self.network.buses.index)
        for gen in self.network.generators.index:
            bus = self.network.generators.bus[gen]
            bus_sizes[bus] = 2e-3
            
        collections = self.network.plot(
            ax=ax,
            geomap=False,
            bus_sizes=bus_sizes,
            bus_colors={bus: 'red' if bus in self.network.generators.bus.values else 'lightblue'
                       for bus in self.network.buses.index},
            bus_alpha=0.6,
            line_widths=1.5,
            line_colors=line_colors,
            line_cmap=plt.cm.viridis,
            title="Lauderdale County Grid"
        )
        
        fig.tight_layout()
        plt.show()
        return collections