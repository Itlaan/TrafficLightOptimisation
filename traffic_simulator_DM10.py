"""
Traffic Light Optimization
Version: _DM10
Focus: Fixes 'teleportation' logic and balances throughput/wait time detection.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum
import random
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f
import warnings
import time

warnings.filterwarnings("ignore")

class LightState(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2

@dataclass
class Vehicle:
    id: int
    position: Tuple[float, float]
    destination: Tuple[float, float]
    speed: float
    wait_time: float = 0.0

@dataclass
class Intersection:
    id: int
    position: Tuple[float, float]
    light_states: Dict[str, LightState]
    timer: float = 0.0

class TrafficSimulator:
    def __init__(self, num_intersections: int = 4, traffic_pattern: np.ndarray = None):
        self.intersections = self._create_intersection_grid(num_intersections)
        self.traffic_pattern = traffic_pattern
        self.vehicles = []
        self.time = 21600.0  # Start at 6am
        self.light_cycle_limit = 30.0 
        self.metrics = {'total_wait_time': 0.0, 'avg_wait_time': 0.0, 'throughput': 0}
        
    def _create_intersection_grid(self, n: int) -> List[Intersection]:
        intersections = []
        for i in range(n):
            intersections.append(Intersection(
                id=i,
                position=(i % 2, i // 2),
                light_states={'north': LightState.GREEN, 'south': LightState.RED, 
                              'east': LightState.RED, 'west': LightState.RED}
            ))
        return intersections
    
    def spawn_vehicle(self, spawn_rate: float = 0.1):
        if random.random() < spawn_rate:
            start_node = random.choice(self.intersections)
            possible_destinations = [i for i in self.intersections if i.id != start_node.id]
            if not possible_destinations: return
            end_node = random.choice(possible_destinations)
            
            start_pos = np.array(start_node.position)
            end_pos = np.array(end_node.position)
            direction = end_pos - start_pos
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                spawn_point = start_pos - (direction / dist) * 0.5
                new_vehicle = Vehicle(
                    id=len(self.vehicles) + 1,
                    position=(spawn_point[0], spawn_point[1]),
                    destination=end_node.position,
                    speed=5.0 
                )
                self.vehicles.append(new_vehicle) 
    
    def update_lights(self, dt: float):
        for intersection in self.intersections:
            intersection.timer += dt
            if intersection.timer > self.light_cycle_limit: 
                if intersection.light_states['north'] == LightState.GREEN:
                    intersection.light_states['north'] = LightState.RED
                    intersection.light_states['east'] = LightState.GREEN
                else:
                    intersection.light_states['north'] = LightState.GREEN
                    intersection.light_states['east'] = LightState.RED
                intersection.timer = 0

    def step(self, dt: float = 0.1):
        current_hour_index = int(self.time // 3600) % len(self.traffic_pattern)
        rate = self.traffic_pattern[current_hour_index] / 20.0 
        
        self.spawn_vehicle(spawn_rate=rate)
        self.update_lights(dt)
        
        for vehicle in self.vehicles[:]:
            pos = np.array(vehicle.position)
            dest = np.array(vehicle.destination)
            direction = dest - pos
            dist_to_dest = np.abs(direction[0]) + np.abs(direction[1]) # Manhattan Distance
            
            # Catchment radius must be slightly larger than the step size (speed * dt)
            if dist_to_dest < 0.6:
                self.vehicles.remove(vehicle)
                self.metrics['throughput'] += 1
                continue
            
            vehicle.speed = 5.0
            unit_vector = direction / (np.linalg.norm(direction) + 1e-6)
            new_pos = pos + (unit_vector * vehicle.speed * dt)

            # Path-based intersection check to prevent 'teleporting' past red lights
            for intersection in self.intersections:
                int_pos = np.array(intersection.position)
                
                # Check if the intersection coordinate lies between current and next position
                crossed_x = (pos[0] <= int_pos[0] <= new_pos[0]) or (new_pos[0] <= int_pos[0] <= pos[0])
                crossed_y = (pos[1] <= int_pos[1] <= new_pos[1]) or (new_pos[1] <= int_pos[1] <= pos[1])
                
                if (crossed_x or crossed_y) and intersection.light_states['north'] == LightState.RED:
                    # Check distance to ensure we only stop when close to the intersection
                    if np.linalg.norm(pos - int_pos) < 1.0:
                        vehicle.speed = 0.0
                        vehicle.wait_time += dt
                        self.metrics['total_wait_time'] += dt
                        break
            
            if vehicle.speed > 0:
                vehicle.position = (new_pos[0], new_pos[1])
        
        self.time += dt
            
    def get_metrics(self) -> Dict[str, float]:
        total_vehicles = self.metrics['throughput'] + len(self.vehicles)
        self.metrics['avg_wait_time'] = self.metrics['total_wait_time'] / total_vehicles if total_vehicles > 0 else 0.0
        return self.metrics

class TrafficDataGenerator:
    @staticmethod
    def generate_daily_patterns(days: int = 7) -> np.ndarray:
        hours = np.arange(24 * days)
        traffic_data = np.zeros(24 * days)
        for h in hours:
            hour_of_day = h % 24
            day_of_week = (h // 24) % 7
            base = 10.0
            if day_of_week < 5:
                m_rush = 80 * np.exp(-((hour_of_day - 8)**2) / (2 * 1.5**2))
                e_rush = 90 * np.exp(-((hour_of_day - 17)**2) / (2 * 1.5**2))
                volume = base + m_rush + e_rush
            else:
                volume = base + 50 * np.exp(-((hour_of_day - 13)**2) / (2 * 3.0**2))
            traffic_data[h] = max(0, volume + np.random.normal(0, 3))
        return traffic_data

def run_comparison(mode="static", gam_model=None, test_pattern=None, mean_vol=33.5):
    np.random.seed(42) 
    sim = TrafficSimulator(num_intersections=4, traffic_pattern=test_pattern)
    n_hrs = 14 
    sim.time = 6 * 3600 
    
    wait_times, throughput_data, time_axis = [], [], []
    last_total_throughput = 0 
    dt = 0.1

    for step in range(int(n_hrs * 3600 / dt)):
        current_hour = (sim.time / 3600) % 24
        
        if mode == "gam":
            pred_vol = gam_model.predict(np.array([[current_hour, 0]]))[0]
            raw_limit = (max(0, pred_vol) / max(0.1, mean_vol)) * 30
            sim.light_cycle_limit = np.clip(raw_limit, 5.0, 30.0)
        else:
            sim.light_cycle_limit = 30.0 
            
        sim.step(dt=dt)
        
        if int(sim.time) % 300 == 0:
            time_axis.append(current_hour)
            metrics = sim.get_metrics()
            wait_times.append(metrics['avg_wait_time'])
            
            current_total = metrics['throughput']
            interval_val = current_total - last_total_throughput
            throughput_data.append(interval_val)
            last_total_throughput = current_total 
                
    return time_axis, wait_times, throughput_data

def calculate_performance_gains(w_static, w_gam, th_static, th_gam):
    total_wait_static = np.sum(w_static)
    total_wait_gam = np.sum(w_gam)
    wait_reduction = ((total_wait_static - total_wait_gam) / total_wait_static) * 100 if total_wait_static > 0 else 0
    
    final_th_static = np.sum(th_static)
    final_th_gam = np.sum(th_gam)
    th_change = ((final_th_gam - final_th_static) / final_th_static) * 100 if final_th_static > 0 else 0
    
    print("-" * 30)
    print("SIMULATION PERFORMANCE REPORT")
    print("-" * 30)
    print(f"Total Cumulative Wait (Static): {total_wait_static:.2f}")
    print(f"Total Cumulative Wait (GAM):    {total_wait_gam:.2f}")
    print(f"Delay Reduction:                {wait_reduction:.2f}%")
    print("-" * 15)
    print(f"Total Throughput (Static):      {final_th_static} cars")
    print(f"Total Throughput (GAM):         {final_th_gam} cars")
    print(f"Throughput Improvement:         {th_change:.2f}%")
    print("-" * 30)
    
    return wait_reduction, th_change

def main():
    start_time = time.time()
    data_gen = TrafficDataGenerator()

    ##### Visualise volume and predicted volume

    # # Generate 2 weeks of data to train a GAM model on
    # raw_data = data_gen.generate_daily_patterns(days=14)
    
    # # Create a DataFrame. 
    # #Volume of vehicles
    # #hour of day
    # #day_type flags 1 if that hour belongs to weekend. 0 otherwise
    # df = pd.DataFrame({
    #     'volume': raw_data,
    #     'hour': [i % 24 for i in range(len(raw_data))],
    #     'day_type': [(1 if (i // 24) % 7 >= 5 else 0) for i in range(len(raw_data))]
    # })
    
    # X_train = df[['hour', 'day_type']].values
    # y_train = df['volume'].values  

    # # s(0) -> Smooth spline for the first column (hour)
    # # f(1) -> Factor for the second column (day_type)
    # gam = LinearGAM(s(0, n_splines=12) + f(1)).fit(X_train, y_train)
    
    # gam.summary()   
    
    # # Generate 4 weeks of data to check model fit
    # test_data = data_gen.generate_daily_patterns(days=28)
    
    # # Create a DataFrame. 
    # #Volume of vehicles
    # #hour of day
    # #day_type flags 1 if that hour belongs to weekend. 0 otherwise
    # df = pd.DataFrame({
    #     'volume': test_data,
    #     'hour': [i % 24 for i in range(len(test_data))],
    #     'day_type': [(1 if (i // 24) % 7 >= 5 else 0) for i in range(len(test_data))]
    # })
    
    # X = df[['hour', 'day_type']].values
    
    # plt.figure(figsize=(10, 5))
    # x_ax_values = np.arange(0,len(test_data))
    # plt.plot(x_ax_values, test_data, label='Actual traffic volume', color='blue')
    # plt.plot(x_ax_values, gam.predict(X), label='Predicted Weekend', color='red', linestyle='--')
    # plt.title('GAM Traffic Volume Prediction')
    # plt.xlabel('Hour of Day')
    # plt.ylabel('Expected Volume')
    # plt.legend()
    # plt.show()    

    ##### --------------------------------------
    
    raw_data = data_gen.generate_daily_patterns(days=14)
    mean_vol = np.mean(raw_data)
    df = pd.DataFrame({
        'volume': raw_data,
        'hour': [i % 24 for i in range(len(raw_data))],
        'day_type': [(1 if (i // 24) % 7 >= 5 else 0) for i in range(len(raw_data))]
    })
    gam = LinearGAM(s(0, n_splines=12) + f(1)).fit(df[['hour', 'day_type']].values, df['volume'].values)
    
    test_pattern = data_gen.generate_daily_patterns(days=1)
    
    t_static, w_static, th_static = run_comparison(mode="static", gam_model=gam, test_pattern=test_pattern, mean_vol=mean_vol)
    t_gam, w_gam, th_gam = run_comparison(mode="gam", gam_model=gam, test_pattern=test_pattern, mean_vol=mean_vol)

    wait_gain, throughput_gain = calculate_performance_gains(w_static, w_gam, th_static, th_gam)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Time of Day (Hours)')
    ax1.set_ylabel('Avg Wait Time (Seconds)', color='tab:red')
    ax1.plot(t_static, w_static, 'tab:red', linestyle='--', label='Static Wait')
    ax1.plot(t_gam, w_gam, 'darkred', linewidth=2, label='GAM Wait')
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Total Throughput (Vehicles)', color='tab:blue')
    ax2.plot(t_static, th_static, color='tab:blue', alpha=0.3, label='Static Throughput')
    ax2.plot(t_gam, th_gam, color='tab:cyan', alpha=0.5, label='GAM Throughput')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True)
    
    plt.title(f'Efficiency Analysis: {wait_gain:.2f}% Delay Reduction / {throughput_gain:.2f}% Vol Gain')
    fig.tight_layout()
    plt.show()    
    
    print(f"Exec Time: {((time.time() - start_time)/60.) :.4f}min")

if __name__ == "__main__":
    main()