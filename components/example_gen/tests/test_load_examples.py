import pickle
import sys

from tfx.components import CsvExampleGen
from tfx.utils.dsl_utils import csv_input

from custom_components.example_gen.src.load_examples import build_example_gen, write_example_gen


def compare_uri(ex0: CsvExampleGen, ex1: CsvExampleGen):
    return ex0.inputs.__getattribute__('_data')['input'].__getattribute__('_artifacts')[0].uri == \
           ex1.inputs.__getattribute__('_data')['input'].__getattribute__('_artifacts')[0].uri


def test_reads_csv():
    # given
    input_csv = "./kddcup.example"
    # when
    example_gen = build_example_gen(input_csv)
    # then
    assert compare_uri(example_gen, CsvExampleGen(csv_input(input_csv)))

def test_write_example_gen():
    # given
    input_csv = "./kddcup.example"
    example_gen = build_example_gen(input_csv)
    # when
    write_example_gen(example_gen, "./example_gen")
    # then
    with open("./example_gen", 'rb') as file:
        assert compare_uri(example_gen, pickle.load(file))
