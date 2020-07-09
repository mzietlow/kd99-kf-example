import argparse
import pickle
from argparse import Namespace

from tensorflow.python import write_file
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.utils.dsl_utils import csv_input

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Example Generator using TFX')
    parser.add_argument('--input-csv', type=str, help='File containing the training samples')
    parser.add_argument('--output-generator-path', type=str, help='Location for the pickled Example Generator')

    return parser.parse_args()


def build_example_gen(input_csv: str) -> CsvExampleGen:
    examples = csv_input(input_csv)
    return CsvExampleGen(examples)


def write_example_gen(example_gen: CsvExampleGen, output_generator_path: str) -> None:
    write_file()
    with open(output_generator_path, 'wb') as output:
        pickle.dump(example_gen, output)


if __name__ == '__main__':
    args = parse_args()

    example_gen = build_example_gen(args.input_csv)
    write_example_gen(example_gen, args.output_generator_path)
