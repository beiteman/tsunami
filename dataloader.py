from pathlib import Path
from analysis import plot_distributions, print_distribution, plot_data_grid
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata
from settings import *
from tqdm import tqdm

import os
import io
import zipfile
import numpy as np
import sys

class ComcotDataset:
    def __init__(self, zip_path, scenario):
        """
        zip_path: path to the .zip file
        scenario: the directory name inside the zip (e.g., 'scenario1/')
        """
        self.zip_path = zip_path
        self.scenario = scenario.strip('/') + '/'
        self.xx = {}
        self.yy = {}
        self.timesteps = {}
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            namelist = zf.namelist()
            folder_files = [f[len(self.scenario):] for f in namelist if f.startswith(self.scenario)]
            
            self.layers = sorted(list(set(
                int(f[5:7]) for f in folder_files 
                if f.startswith('layer') and f.endswith('_x.dat')
            )))
            
            for layer in self.layers:
                with zf.open(f"{self.scenario}layer{layer:02d}_x.dat") as f:
                    self.xx[layer] = np.loadtxt(f)
                with zf.open(f"{self.scenario}layer{layer:02d}_y.dat") as f:
                    self.yy[layer] = np.loadtxt(f)
                
                self.timesteps[layer] = sorted([
                    int(f.replace(self.scenario, "")[5:11]) for f in namelist 
                    if f.startswith(f'{self.scenario}z_{layer:02d}_') and f.endswith('.dat')
                ])

    def _loadtxt(self, filename):
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(f"{self.scenario}{filename}") as f:
                return np.loadtxt(f, dtype=np.float64)

    def _fromstring(self, filename):
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(f"{self.scenario}{filename}") as f:
                content = f.read().decode('utf-8')
                return np.fromstring(content, dtype=np.float64, sep=' ')
    
    def timestep(self, layer: int):
        return self.timesteps[layer]
    
    def x(self, layer: int): # shape:[X]
        return self.xx[layer]
    
    def y(self, layer: int): # shape:[Y]
        return self.yy[layer]

    def e(self, layer: int): # shape:[Y,X]
        ny, nx = len(self.yy[layer]), len(self.xx[layer])
        return self._loadtxt(f"layer{layer:02d}.dat").reshape((ny, nx))

    def z(self, layer: int): # shape:[T,Y,X]
        records = []
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            ny, nx = len(self.yy[layer]), len(self.xx[layer])
            for t in self.timesteps[layer]:
                records.append(self._fromstring(f"z_{layer:02d}_{t:06d}.dat").reshape((ny, nx)))
        return np.stack(records, axis=0) if records else None

    def zmax(self, layer: int): # shape:[Y,X]
        ny, nx = len(self.yy[layer]), len(self.xx[layer])
        return self._fromstring(f"zmax_layer{layer:02d}.dat").reshape((ny, nx))

    def zmin(self, layer: int): # shape:[Y,X]
        ny, nx = len(self.yy[layer]), len(self.xx[layer])
        return self._fromstring(f"zmin_layer{layer:02d}.dat").reshape((ny, nx))

    def hmax(self, layer: int): # shape:[Y,X]
        ny, nx = len(self.yy[layer]), len(self.xx[layer])
        return self._fromstring(f"hmax_layer{layer:02d}.dat").reshape((ny, nx))

# timesteps to make the model time-invariant
def get_dz(dataset: ComcotDataset, layer: int, threshold = 1e-09, timesteps=30): # shape:[T,Y,X]
    z = dataset.z(layer=layer)
    dz = np.diff(z, axis=0, prepend=z[0:1])
    dz[0] = 0
    dz = np.where(np.abs(dz) < threshold, 0, dz/timesteps)
    return dz

# normalize data with shape [Y,X] that has values [0,inf]
# alpha > 1 increases std; alpha < 1 decreases std
# CNN will favor data with higher STD over the lower 
# instance-independent normalization
def norm(data, alpha=1.0):
    return np.sign(data) * np.log1p(np.abs(data) * alpha)

def denorm(norm_data, alpha=1.0):
    return np.sign(norm_data) * (np.expm1(np.abs(norm_data)) / alpha)

# instance-dependent normalization
def norm_std(data):
    log_data = norm(data, alpha=1.0)
    return (log_data - np.mean(log_data)) / (np.std(log_data) + 1e-8)

# instance-dependent normalization
def norm_scale(data):
    log_data = norm(data, alpha=1.0)
    d_min = np.min(log_data)
    d_max = np.max(log_data)
    return (log_data - d_min) / (d_max - d_min + 1e-8)

# distance with mask=0
def distance_pixel(mask):
    dist = distance_transform_edt(mask == 0)
    return dist / dist.max()

def get_dxdy(X, Y):
    lon, lat = X, Y
    R = 6371000.0  # meters
    dlat = np.deg2rad(np.mean(np.diff(lat)))
    dy = R * dlat
    mean_lat = np.deg2rad(lat.mean())
    dlon = np.deg2rad(np.mean(np.diff(lon)))
    dx = R * np.cos(mean_lat) * dlon
    return dx, dy

# X: lon, Y: lat
def distance_geo(mask, dx, dy):
    return distance_transform_edt(mask == 0, sampling=(dy, dx))

# returns shape:[Y,X] (1/0)
def sensor_placement_area(e, dx, dy):
    dist = distance_geo(e < 0, dx, dy).astype(int)
    area = ((dist > SAMPLE_MIN_OFFSHORE_DIST) & (dist < SAMPLE_MAX_OFFSHORE_DIST)).astype(int)
    return area

# data/output shape:[Y,X]
# 'linear': Best for general gradients (e.g., elevation or distance).
# 'nearest': Best if you want to preserve sharp boundaries and avoid creating "new" intermediate values.
# 'cubic': Smoothest result, but can create "overshoot" artifacts (values slightly higher/lower than your original range).
def interpolate(data, method='cubic'):
    h, w = data.shape
    y, x = np.mgrid[0:h, 0:w]
    
    known_mask = data != 0
    points = np.column_stack((y[known_mask], x[known_mask]))
    values = data[known_mask]
    
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    interpolated_data = griddata(points, values, (grid_y, grid_x), method=method)
    if np.isnan(interpolated_data).any():
        nan_mask = np.isnan(interpolated_data)
        interpolated_data[nan_mask] = griddata(points, values, (grid_y[nan_mask], grid_x[nan_mask]), method='nearest')
    return interpolated_data

# randomly put {num_sensors} sensors in designated area
# sensor_area shape:[Y,X]
# mask shape:[Y,X] or None
# returns shape:[Y,X] (1/0)
def place_sensor(sensor_area, num_sensors, mask=None):
    valid_map = sensor_area.astype(bool)
    if mask is not None:
        valid_map = valid_map & mask.astype(bool)
    candidate_indices = np.argwhere(valid_map)
    num_available = len(candidate_indices)
    if num_available < num_sensors:
        num_sensors = num_available
    selected_indices = np.random.choice(num_available, num_sensors, replace=False)
    final_coords = candidate_indices[selected_indices]
    sensor_grid = np.zeros(sensor_area.shape, dtype=float)
    sensor_grid[final_coords[:, 0], final_coords[:, 1]] = 1.0
    return sensor_grid

def process(src_datapath, scenarios, layers, output_filepath):
    inputs_samples, target_samples, loss_multiplier_samples = [], [], []
    for scenario, layer in tqdm(zip(scenarios, layers), total=len(scenarios)):
        dataset = ComcotDataset(src_datapath, scenario)
        
        e = dataset.e(layer=layer) # [Y,X]
        z = dataset.z(layer=layer) # [B,Y,X]
        dz = get_dz(dataset=dataset, layer=layer, timesteps=30) # [B,Y,X]
        
        batch_size = z.shape[0]
        z_norm = norm(z, alpha=NORM_ALPHA_Z) # shape:[B,Y,X]
        dz_norm = norm(dz, alpha=NORM_ALPHA_DZ) # shape:[B,Y,X]
        e_norm = norm(e, alpha=NORM_ALPHA_E) # shape:[Y,X]
        X = np.stack((z_norm, dz_norm), axis=1) # shape:[B,2,Y,X]
        
        zmax = dataset.zmax(layer=layer)
        hmax = dataset.hmax(layer=layer)
        
        # as targets
        zmax_norm = norm(zmax, alpha=NORM_APLHA_ZMAX) # shape:[Y,X]
        
        #
        dx, dy = get_dxdy(dataset.x(layer=layer), dataset.y(layer=layer))
        
        # distance to coastline
        dist_land = distance_geo(mask=(e < 0), dx=dx, dy=dy)
        dist_ocean = distance_geo(mask=(e > 0), dx=dx, dy=dy)
        sdc = dist_ocean - dist_land
        sdc_norm = norm(data=sdc, alpha=NORM_ALPHA_SDC)
        
        # specific area to capture sample data
        sensor_area = sensor_placement_area(e, dx, dy)
        sensor_area_pixels = np.sum(sensor_area)
        num_sensors = int((SAMPLE_POINT_PERCENT * sensor_area_pixels)/100)
        
        # sampling
        for _ in tqdm(range(SAMPLE_PER_SCENARIO), leave=False):
            
            # --------------------------------------------------------
            # select a timestep randomly
            random_idx = np.random.randint(1, batch_size) # ommit the first data (T=0), since dZ=0 
            sample_data = X[random_idx] # shape:[4,Y,X]
            z_sample = sample_data[0, :, :] # shape:[Y,X]
            dz_sample = sample_data[1, :, :] # shape:[Y,X]
            
            # --------------------------------------------------------
            # select sampling data
            sensors = place_sensor(sensor_area=sensor_area, num_sensors=num_sensors, mask=None)
            # select only sensors that shows significant wave
            sensors = np.where(np.abs(zmax) < 1e-1, 0, sensors)
            # skip if not enough sensors seen
            num_sensor_available = len(np.argwhere(sensors))
            if num_sensor_available < SAMPLE_POINT_MIN: continue
            z_sensor = interpolate(z_sample * sensors)
            dz_sensor = interpolate(dz_sample * sensors)
            loss_multiplier = distance_pixel(mask=sensors) # shape:[Y,X]
            loss_multiplier = 1 / (loss_multiplier + 0.0000001)
            loss_multiplier = norm_scale(loss_multiplier)
            
            inputs = np.stack((z_sensor,dz_sensor,e_norm,sdc_norm), axis=0) # shape:[4,Y,X]
            targets = np.stack((zmax_norm,), axis=0) # shape:[1,Y,X]
            loss_multiplier = np.stack((loss_multiplier,), axis=0) # shape:[1,Y,X]
            
            inputs_samples.append(inputs)
            target_samples.append(targets)
            loss_multiplier_samples.append(loss_multiplier)
            
    inputs = np.stack(inputs_samples, axis=0) # shape:[B,C=4,Y,X] C:[z_sensor,dz_sensor,e_norm,sdc_norm]
    targets = np.stack(target_samples, axis=0) # shape:[B,C=1,Y,X] C:[zmax_norm]
    loss_multiplier = np.stack(loss_multiplier_samples, axis=0) # shape:[B,C=1,Y,X]
    np.savez_compressed(output_filepath, inputs=inputs, targets=targets, loss_multiplier=loss_multiplier)
    

def analysis(dataset_path, plot_samples=5):
    dataset = np.load(dataset_path)
    inputs = dataset["inputs"]
    targets = dataset["targets"]
    loss_multiplier = dataset["loss_multiplier"]
    
    # check stats
    print_distribution(inputs, channel_names=["Z (sample)", "dZ (sample)", "Topology", "SDC"])
    print_distribution(targets, channel_names=["Z_max"])
    print_distribution(loss_multiplier, channel_names=["loss_multiplier"])
    
    # plots
    datasets, titles = [], []
    selected_index = np.random.randint(0, inputs.shape[0], plot_samples)
    for plot_index in selected_index:
        z = denorm(inputs[plot_index][0], alpha=NORM_ALPHA_Z)
        dz = denorm(inputs[plot_index][1], alpha=NORM_ALPHA_DZ)
        e = denorm(inputs[plot_index][2], alpha=NORM_ALPHA_E)
        sdc = denorm(inputs[plot_index][3], alpha=NORM_ALPHA_SDC)
        zmax = denorm(targets[plot_index][0], alpha=NORM_APLHA_ZMAX)
        datasets += [z, dz, e, sdc, zmax, loss_multiplier[plot_index][0]]
        titles += ["Input:z", "Input:dz", "Input:e", "Input:SDC", "Target:zmax", "loss_mult"]
    plot_data_grid(datasets=datasets, titles=titles, cols=6)
    

# CHECKLIST:
# - time invariance: embedded in dz information (timestep)
# - topology invariance: embedded in e
# - dimension size invariance: TODO
# - pixel size invariance: by point sampling
if __name__ == "__main__":
    src_datapath = "data/pelabuhanratu.zip"
    
    # ------------------------------------
    # TEST DATASET
    # ------------------------------------
    scenarios = ["scenario1", "scenario2", "scenario4"]
    layers = [1, 1, 1]
    process(src_datapath=src_datapath, scenarios=scenarios, layers=layers, output_filepath="data/train.npz")

    # ------------------------------------
    # TRAIN DATASET
    # ------------------------------------
    scenarios = ["scenario3"]
    layers = [1]
    process(src_datapath=src_datapath, scenarios=scenarios, layers=layers, output_filepath="data/test.npz")
    
    # ------------------------------------
    # ANALYSIS
    # ------------------------------------
    analysis(dataset_path="data/train.npz", plot_samples=3)