#!/usr/bin/python

import argparse
import logging
import csv
from pathlib import Path

import libkinetics


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='increase output verbosity')
    parser.add_argument('-wh',
                        '--with_hill',
                        action='store_true',
                        help='compute additional kinetics using Hill equation')
    parser.add_argument('input',
                        type=str,
                        help='directory containing input files in csv format')
    parser.add_argument('output',
                        type=str,
                        help='results will be written to this directory')

    args = parser.parse_args()

    return args


def main():
    # parse command line arguments
    args = parse_arguments()
    if args.with_hill:
        do_hill = args.with_hill
    else:
        do_hill = False
    try:
        input_path = Path(args.input).resolve()
    except FileNotFoundError:
        print('Path containing input data not found: {}'.format(args.input))
        raise
    try:
        output_path = Path(args.output).resolve()
    except FileNotFoundError:
        print('Path for writing results not found: {}'.format(args.output))
        raise

    if output_path.is_dir():
        if input_path.is_dir():
            data_files = sorted(input_path.glob('**/*.csv'))
            exp = libkinetics.Experiment(data_files, (10, 25), do_hill)
            exp.plot_data(str(output_path))
            exp.plot_kinetics(str(output_path))
            exp.write_data(str(output_path))
        else:
            raise ValueError('{} is not a directory!'.format(input_path))
    else:
        raise ValueError('{} is not a directory!'.format(output_path))

if __name__ == "__main__":
    main()
