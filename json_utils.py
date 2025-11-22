import json


def restart_json(filename):
    with open(filename, "w") as file:
        json.dump([], file)


def read_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


def write_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def add_overlap_object(data, filename):
    initial_data = read_json(filename)
    initial_data.append(data)
    write_json(initial_data, filename)
