import pandas as pd
import pypsa
import numpy as np
import geopandas as gpd
from shapely import box

class LauderdaleGrid:
    def __init__(self, gpkg_path):
        # Store path to network data file and initialize empty network
        self.gpkg_path = gpkg_path
        self.network = None

        # Define generation resources with operating parameters 
        self.generators = {
            'BROWNS FERRY': {
                'capacity': 3816,
                'type': 'nuclear',
                'marginal_cost': 10, 
                'min_up_time': 24,
                'p_min_pu': 0.2,
                'ramp_limit': 0.1
            },
            'WHEELER DAM': {
                'capacity': 411.6,
                'type': 'hydro',
                'marginal_cost': 5,
                'min_up_time': 1, 
                'p_min_pu': 0.0,
                'ramp_limit': 0.3
            },
            'WILSON DAM': {
                'capacity': 629.8,
                'type': 'hydro',
                'marginal_cost': 5,
                'min_up_time': 1,
                'p_min_pu': 0.0,
                'ramp_limit': 0.3
            },
            'PICKWICK LANDING DAM': {
                'capacity': 240,
                'type': 'hydro', 
                'marginal_cost': 5,
                'min_up_time': 1,
                'p_min_pu': 0.0,
                'ramp_limit': 0.3
            },
            'MORGAN ENERGY CENTER': {
                'capacity': 1230,
                'type': 'ccgt',
                'marginal_cost': 40,
                'min_up_time': 4,
                'p_min_pu': 0.0,
                'ramp_limit': 0.25
            },
            'DECATUR ENERGY CENTER': {
                'capacity': 782,
                'type': 'ccgt',
                'marginal_cost': 40,
                'min_up_time': 4,
                'p_min_pu': 0.0,
                'ramp_limit': 0.25
            }
        }

        # Define thermal limits by voltage level
        self.voltage_thermal_limits = {
            500: 3300,  
            161: 400  
        }

    def parse_voltage(self, voltage):
        """Helper function to handle voltage values."""
        try:
            return float(voltage) if float(voltage) != -999999 else 161.0
        except ValueError:
            return 161.0

    def get_thermal_limit(self, voltage):
        """Get thermal limit based on voltage."""
        v = self.parse_voltage(voltage)
        return self.voltage_thermal_limits[500] if v >= 400 else self.voltage_thermal_limits[161]

    def create_network(self):
        """Create core PyPSA network from source data"""
        self.network = pypsa.Network()
        
        
        # Set up 24 hourly timestamps
        self.network.set_snapshots(range(24))
        
        # Add required carriers first
        self.network.add("Carrier", "AC")
        self.network.add("Carrier", "nuclear")
        self.network.add("Carrier", "hydro")
        self.network.add("Carrier", "ccgt")
        
        # Load and filter network data for region
        gdf = gpd.read_file(self.gpkg_path)
        lauderdale_box = box(-88.0, 34.5, -87.0, 35.2)
        gdf['intersects'] = gdf.geometry.intersects(lauderdale_box)
        lauderdale_gdf = gdf[gdf['intersects']]
        
        # Process voltage levels
        substation_voltages = {}
        for _, row in lauderdale_gdf.iterrows():
            voltage = self.parse_voltage(row['VOLTAGE'])
            for sub in [row['SUB_1'], row['SUB_2']]:
                if sub and sub.strip():
                    substation_voltages[sub] = max(
                        substation_voltages.get(sub, 0),
                        voltage
                    )

        # Add buses
        for sub, voltage in substation_voltages.items():
            self.network.add("Bus",
                        sub,
                        v_nom=voltage,
                        carrier="AC")

        # Add transmission lines with proper impedance values
        for idx, row in lauderdale_gdf.iterrows():
            if row['SUB_1'] and row['SUB_2']:
                voltage = self.parse_voltage(row['VOLTAGE'])
                s_nom = self.get_thermal_limit(voltage)*1.05
                
                #length in kms
                length_km =row["SHAPE__Len"]/1000
                # Calculate actual impedance values
                base_z = (voltage**2) / 100  # Using 100 MVA base
                x = 0.1 * base_z 
                r = 0.01 * base_z
                
                self.network.add("Line",
                            f"Line_{idx}",
                            bus0=row['SUB_1'],
                            bus1=row['SUB_2'], 
                            v_nom=voltage,
                            s_nom=s_nom,
                            length=length_km,
                            x=x,          # Using actual impedance instead of per unit
                            r=r,          # Using actual impedance instead of per unit
                            carrier="AC")

        # Add generators
        for name, params in self.generators.items():
            if name in self.network.buses.index:
                self.network.add("Generator",
                    f"gen_{name}",
                    bus=name,
                    carrier=params['type'],
                    p_nom=params['capacity'],
                    marginal_cost=params['marginal_cost'],
                    p_min_pu=params['p_min_pu'],
                    ramp_limit_up=params['ramp_limit'],
                    ramp_limit_down=params['ramp_limit'],
                    min_up_time=params['min_up_time'])

        
        
        # topology and filter out other sub networks
        self.network.determine_network_topology()
        #sub network we keep
        main_sub ="1"
        to_keep = self.network.buses[self.network.buses["sub_network"] == main_sub].index
        
        #keep what we want
        self.network.buses = self.network.buses.loc[to_keep]
        self.network.lines= self.network.lines[
            self.network.lines["bus0"].isin(to_keep) & self.network.lines["bus1"].isin(to_keep)
        ]
        self.network.generators = self.network.generators[self.network.generators["bus"].isin(to_keep)]
        self.network.loads["bus"].isin(to_keep)
        # add transformers
        for _, row in self.network.lines.iterrows():
            bus0_v= self.network.buses.at[row["bus0"],"v_nom"]
            bus1_v= self.network.buses.at[row["bus1"],"v_nom"]
            if bus0_v != bus1_v:
                self.network.add("Transformer",
                                 f"Transformer_{row.name}",
                                 bus0=row["bus0"],
                                 bus1=row["bus1"],
                                 s_nom=row["s_nom"],
                                 x=0.9,
                                 r=0.05,
                                 v_nom0=bus0_v,
                                 v_nom1=bus1_v,
                                 tap_pos=0,
                                 tap_step_percent=2.5,
                                 tap_min=-10,
                                 tap_max=10)
        # update the stand alone line
        self.network.lines.at["Line_1024", "s_nom"] *= 5.5  # scale the capacity 
        self.network.lines.at["Line_27871", "s_nom"] *=1.25
        self.network.lines.at["Line_35936", "s_nom"] *=1.55
        #solve power flow
        self.network.lpf()
        print(f"\nNetwork Creation Summary:")
        print(f"Buses: {len(self.network.buses)}")
        print(f"Lines: {len(self.network.lines)}")
        print(f"Generators: {len(self.network.generators)}")
        print(f"Transformers: {len(self.network.transformers)}")
        return self.network
    
    def add_loads(self, predictor, input_sequence, scale_factor=0.005):
        """Add loads to the network based on predicted demand."""
        tva_demand = predictor.model.predict(input_sequence)
        tva_demand = predictor.demand_scaler.inverse_transform(tva_demand)
        regional_demand = tva_demand.item(0) * scale_factor

        # Safety limit check
        total_capacity = self.network.generators["p_nom"].sum()
        max_safe_load = total_capacity * 0.8

        # Introduce variability with a random scaling factor
        variation_factor = np.random.uniform(0.9, 1.1)  # 10% random variation
        regional_demand *= variation_factor

        # Apply safety limit check with variability considered
        if regional_demand > max_safe_load:
            scale_factor *= max_safe_load / regional_demand
            regional_demand = max_safe_load
        elif regional_demand < 0.5 * max_safe_load:  # Prevent excessively low demand
            regional_demand = 0.5 * max_safe_load  # Set a minimum threshold for demand
        for gen in self.network.generators.index:
            self.network.generators.at[gen, 'p_max_pu'] = np.random.uniform(0.7, 1.0)  # 70%-100% capacity
            self.network.generators.at[gen, 'p_nom_max'] = (
            self.network.generators.at[gen, 'p_nom'] * self.network.generators.at[gen, 'p_max_pu']
            )
            self.network.generators.at[gen, 'p_nom_min'] = 0  # Minimum dispatch
        # Set up snapshots
        snapshots = pd.date_range("2025-01-01", periods=24, freq="H")
        self.network.set_snapshots(snapshots)

        # Clear existing loads
        self.network.loads = pd.DataFrame()

        # Load distribution
        load_distribution = {
            'DECATUR': {'share': 0.15, 'type': 'industrial', 'peak_hour': 14},
            'TRINITY': {'share': 0.12, 'type': 'mixed', 'peak_hour': 18},
            'LIMESTONE': {'share': 0.12, 'type': 'residential', 'peak_hour': 19},
            'CHEROKEE': {'share': 0.12, 'type': 'mixed', 'peak_hour': 17},
            'SHOALS': {'share': 0.12, 'type': 'industrial', 'peak_hour': 13},
            'WAYNESBORO': {'share': 0.12, 'type': 'residential', 'peak_hour': 20},
            'UNION': {'share': 0.13, 'type': 'mixed', 'peak_hour': 16}
        }

        for bus, params in load_distribution.items():
            if bus in self.network.buses.index:
                base_load = regional_demand * params['share']


                print(f"Adding load for {bus}: Base Load = {base_load:.4f}")

                self.network.add(
                    "Load",
                    f"load_{bus}",
                    bus=bus,
                    p_set=float(base_load)
                )

        # After adding loads, check if they were actually set
        print("\n=== Loads After Addition ===")
        print(self.network.loads)
        print("============================\n")

        return self.network
        #currently unused
        """    def _generate_load_profile(self, base_load, params, snapshots):
                profile = []
                for hour in range(24):
                    if params['type'] == 'industrial':
                        factor = 0.9 + 0.2 * np.sin(np.pi * (hour - 8) / 9) if 8 <= hour <= 17 else 0.6
                    elif params['type'] == 'residential':
                        morning = 0.7 + 0.3 * np.exp(-(hour - 7) ** 2 / 8)
                        evening = 0.8 + 0.2 * np.exp(-(hour - params['peak_hour']) ** 2 / 8)
                        factor = max(morning, evening)
                    else:  # mixed
                        factor = 0.7 + 0.3 * np.exp(-(hour - params['peak_hour']) ** 2 / 16)
                    profile.append(base_load * factor)
                return profile"""
