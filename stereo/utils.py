'''
classes used by multiple modules
'''

import argparse
import yaml


class InputParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_arguments(self, name, default, definition):
        self.parser.add_argument(name, default=default, help=definition)

    def get_arguments(self):
        return self.parser.parse_args()


class YamlReader:
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        config = None
        with open(self.filename, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(exception)
        return config