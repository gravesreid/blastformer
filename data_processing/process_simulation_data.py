import os
import json

simulation_data_path = "/home/reid/projects/blast_waves/dataset_parallel_large"

output_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_probe_locations(filepath):
    # make list to store probe coordinates
    probe_locations = []
    with open(filepath, 'r') as f:
        for line in f:
            # probe coordinates lines start with "# Probe"
            if line.startswith("# Probe"):
                parts = line.split()
                # probe_id is the 3rd element in the line
                probe_id = int(parts[2])
                # coordinates are the 4th, 5th, and 6th elements in the line
                coordinate_string =  parts[3] + " " + parts[4] + " " + parts[5]
                # store the coordinates as a tuple of floats
                coordinates = tuple([float(x) for x in coordinate_string.strip("()").split()])
                # store the probe id and coordinates in a dictionary
                probe_locations.append({"id" : probe_id, "coordinates" : coordinates})
    return probe_locations

def parse_probe_data(filepath):
    # make dictionary to store probe data
    probe_data = {"probe_number" : [], "time" : {}}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#         Time"):
                probe_line = line.split()
                probe_entry = tuple([int(x) for x in probe_line[2:]])
                probe_data["probe_number"].append(probe_entry)
            elif line.startswith("# Probe"):
                pass
            else:
                parts = line.split()
                time = float(parts[0])
                pressure_data = tuple([float(x) for x in parts[1:]])
                probe_data["time"].update({time : pressure_data})
    return probe_data

def parse_wall_locations(filepath):
    wall_locations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].strip().startswith("xMin"):
                x_min = float(lines[i].split()[1].strip(";"))
                y_min = float(lines[i+1].split()[1].strip(";"))
                z_min = float(lines[i+2].split()[1].strip(";"))
                x_max = float(lines[i+3].split()[1].strip(";"))
                y_max = float(lines[i+4].split()[1].strip(";"))
                z_max = float(lines[i+5].split()[1].strip(";"))
                wall_locations.append({
                    "x_min" : x_min, "y_min" : y_min, "z_min" : z_min, 
                    "x_max" : x_max, "y_max" : y_max, "z_max" : z_max
                                       })
    return wall_locations

def parse_charge_data(filepath):
    charge_data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("mass"):
                charge_data["mass"] = float(line.split()[1].strip(";"))
            elif line.startswith("cent0"):
                cent0 = line.split()[1].strip("(") + " " + line.split()[2] + " " + line.split()[3].strip(");")
                charge_data["cent0"] = tuple([float(x) for x in cent0.split()]) 
            elif line.startswith("p10"):
                p10 = line.split()[1].strip("(") + " " + line.split()[2] + " " + line.split()[3].strip(");")
                charge_data["p10"] = tuple([float(x) for x in p10.split()])
            elif line.startswith("radius0"):
                charge_data["radius0"] = float(line.split()[1].strip(";"))
            elif line.startswith("cent1"):
                cent1 = line.split()[1].strip("(") + " " + line.split()[2] + " " + line.split()[3].strip(");")
                charge_data["cent1"] = tuple([float(x) for x in cent1.split()])
            elif line.startswith("p11"):
                p11 = line.split()[1].strip("(") + " " + line.split()[2] + " " + line.split()[3].strip(");")
                charge_data["p11"] = tuple([float(x) for x in p11.split()])
            elif line.startswith("radius1"):
                charge_data["radius1"] = float(line.split()[1].strip(";")) 
    return charge_data

def make_samples(probe_data, probe_locations, wall_locations, charge_data):
    for time, pressure_data in probe_data['time'].items():
        yield {
            "time" : time,
            #"probe_locations" : probe_locations,
            "pressure" : pressure_data,
            "wall_locations" : wall_locations,
            "charge_data" : charge_data,
        }





def main():
    simulation_runs = os.listdir(simulation_data_path)
    for run in simulation_runs:
        print(f"Processing {run}")
        pressure_file = f"{run}/postProcessing/pressureProbes/0/p"
        wall_file = f"{run}/variables/wall_loc"
        charge_file = f"{run}/variables/charge_loc"
        pressure_file_path = os.path.join(simulation_data_path, pressure_file)
        probe_locations = parse_probe_locations(pressure_file_path)
        probe_data = parse_probe_data(pressure_file_path)
        wall_locs = parse_wall_locations(os.path.join(simulation_data_path, wall_file))
        charge_locs = parse_charge_data(os.path.join(simulation_data_path, charge_file))
        run_output_dir = os.path.join(output_dir, run)
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)
        for i, sample in enumerate(make_samples(probe_data, probe_locations, wall_locs, charge_locs)):
            output_path = os.path.join(run_output_dir, f"sample_{i}.json")
            with open(output_path, 'w') as f:
                json.dump(sample, f, indent=4)
    


if __name__ == "__main__":
    main()