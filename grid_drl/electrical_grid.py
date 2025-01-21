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
            500: 2000,  # 500 kV -> 2000 MVA (from docs)
            161: 200    # 161 kV -> 200 MVA (from docs) 
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
                s_nom = self.get_thermal_limit(voltage)
                
                # Calculate actual impedance values (not per unit)
                base_z = (voltage**2) / 100  # Using 100 MVA base
                x = 0.2 * base_z
                r = 0.02 * base_z
                
                self.network.add("Line",
                            f"Line_{idx}",
                            bus0=row['SUB_1'],
                            bus1=row['SUB_2'], 
                            v_nom=voltage,
                            s_nom=s_nom,
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

        print(f"\nNetwork Creation Summary:")
        print(f"Buses: {len(self.network.buses)}")
        print(f"Lines: {len(self.network.lines)}")
        print(f"Generators: {len(self.network.generators)}")
        
        return self.network
    def add_loads(self, predictor, input_sequence, scale_factor=0.005):
        """Add loads to the network based on predicted demand."""
        # Predict regional demand using the given predictor model
        tva_demand = predictor.model.predict(input_sequence)
        tva_demand = predictor.demand_scaler.inverse_transform(tva_demand)
        regional_demand = tva_demand.item(0) * scale_factor

        # Calculate total generation capacity and enforce safety limits
        total_capacity = sum(params['capacity'] for params in self.generators.values())
        max_safe_load = total_capacity * 0.8  # 80% of total capacity

        if regional_demand > max_safe_load:
            old_scale = scale_factor
            scale_factor *= max_safe_load / regional_demand
            print(f"\nWARNING: Adjusting scale factor from {old_scale:.6f} to {scale_factor:.6f}")
            print(f"Original demand {regional_demand:.2f} MW exceeded safe limit of {max_safe_load:.2f} MW")
            regional_demand = max_safe_load

        # Ensure the network has snapshots set for the 24-hour profile
        self.network.set_snapshots(pd.date_range("2025-01-01", periods=24, freq="H"))

        # Remove existing loads
        for bus in list(self.network.loads.index):
            try:
                self.network.remove("Load", bus)
            except KeyError:
                pass

        # Define load distribution based on substation roles
        load_distribution = {
            'DECATUR': {
                'share': 0.15,        # Major industrial
                'type': 'industrial',
                'peak_hour': 14
            },
            'TRINITY': {
                'share': 0.12,
                'type': 'mixed',
                'peak_hour': 18
            },
            'LIMESTONE': {
                'share': 0.12,
                'type': 'residential',
                'peak_hour': 19
            },
            'CHEROKEE': {
                'share': 0.12,
                'type': 'mixed',
                'peak_hour': 17
            },
            'SHOALS': {
                'share': 0.12,
                'type': 'industrial',
                'peak_hour': 13
            },
            'WAYNESBORO': {
                'share': 0.12,
                'type': 'residential',
                'peak_hour': 20
            },
            'PULASKI': {
                'share': 0.12,
                'type': 'residential',
                'peak_hour': 19
            },
            'UNION': {
                'share': 0.13,
                'type': 'mixed',
                'peak_hour': 16
            }
        }

        print("\nLoad Distribution Analysis")
        print('=' * 50)
        print(f"{'Location':<15}{'Type':<12} {'Share':<10} {'Base Load (MW)':<15}")
        print('=' * 50)

        for bus, params in load_distribution.items():
            if bus in self.network.buses.index:
                base_load = regional_demand * params['share']

                # Generate 24-hour load profile
                load_profile = []
                for hour in range(24):
                    if params['type'] == 'industrial':
                        # Industrial load peaks during working hours
                        if 8 <= hour <= 17:
                            factor = 0.9 + 0.2 * np.sin(np.pi * (hour - 8) / 9)
                        else:
                            factor = 0.6
                    elif params['type'] == 'residential':
                        # Residential load has a double peak (morning and evening)
                        morning_peak = 0.7 + 0.3 * np.exp(-(hour - 7) ** 2 / 8)
                        evening_peak = 0.8 + 0.2 * np.exp(-(hour - params['peak_hour']) ** 2 / 8)
                        factor = max(morning_peak, evening_peak)
                    else:
                        # Mixed load blends profiles
                        factor = 0.7 + 0.3 * np.exp(-(hour - params['peak_hour']) ** 2 / 16)

                    load_value = base_load * factor
                    load_profile.append(load_value)

                self.network.add(
                    "Load",
                    f"load_{bus}",
                    bus=bus,
                    p_set=pd.Series(load_profile, index=self.network.snapshots),  # Align with snapshots
                    q_set=0.0,
                    type=params['type']
                )

                print(f"{bus:<15} {params['type']:<12} {params['share'] * 100:>6.1f}%  {base_load:>10.2f}")

        return self.network

