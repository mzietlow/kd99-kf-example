import argparse
import gzip
import os
import urllib.request
from shutil import copyfile

from tfx.utils.dsl_utils import external_input


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='File loader for pipeline')
    parser.add_argument('--split', type=str, help='Either train, test or validate.')
    parser.add_argument('--output-path', type=str, help='Location for the respective csv')

    return parser.parse_args()


def download_kdd99(destination: str):
    url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    compressed_filename = "./kddcup.gz"
    uncompressed_filename = os.path.join(destination, 'kddcup.csv')
    urllib.request.urlretrieve(url, compressed_filename)
    with open(uncompressed_filename, "wb") as uncompressed_file:
        with gzip.open(compressed_filename) as compressed_file:
            uncompressed_data = compressed_file.read()
        uncompressed_file.write(uncompressed_data)


if __name__ == '__main__':
    args = parse_arguments()
    print(f"Using split: {args.split}")
    print(f"Writing to {args.output_path}")
    if args.split == "train":
        copyfile("./kddcup.train", args.output_path)
    elif args.split == "test":
        copyfile("./kddcup.test", args.output_path)
    elif args.split == "validate":
        copyfile("./kddcup.validate", args.output_path)
    out_file_path = os.path.join(os.getcwd(), "out")
    with open(out_file_path, "w+") as out_file:
        print(f"Writing external_input to {out_file_path}")
        out_file.write(external_input(args.output_path))
