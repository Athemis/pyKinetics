#!/usr/bin/python
# -*- coding: utf-8 -*-

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


def initialize_logger():
    """
    Initialization of logging subsystem. Two logging handlers are brought up:
    'fh' which logs to a log file and 'ch' which logs to standard output.
    :return logger: returns a logger instance
    """
    logger = logging.getLogger('pyKinetics-cli')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    try:
        log_filename = 'pyKinetics-cli.log'
        fh = logging.FileHandler(log_filename, 'w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    except IOError as error:
        logger.warning('WARNING: Cannot create log file! Run pyKinetics-cli'
                       'from a directory to which you have write access.')
        logger.warning(error.msg)
        pass

    return logger


def main():
    # parse command line arguments
    args = parse_arguments()
    # initialize logger
    logger = initialize_logger()

    if args.with_hill:
        do_hill = args.with_hill
    else:
        do_hill = False
    try:
        input_path = Path(args.input).resolve()
    except FileNotFoundError:
        logger.critical('CRITICAL: Path containing input data '
                        'not found: {}'.format(args.input))
        raise
    try:
        output_path = Path(args.output).resolve()
    except FileNotFoundError:
        logger.critical('CRITICAL: Path for writing results '
                        'not found: {}'.format(args.output))
        raise

    if output_path.is_dir():
        if input_path.is_dir():
            logger.info('INFO: Collecting data files')
            data_files = sorted(input_path.glob('**/*.csv'))
            msg = 'Calculating kinetics'
            if do_hill:
                msg = '{} including Hill kinetics'.format(msg)
            logger.info('INFO: {}'.format(msg))
            exp = libkinetics.Experiment(data_files, (10, 25), do_hill)
            logger.info('INFO: Plotting linear fits to data and kinetics')
            exp.plot_data(str(output_path))
            exp.plot_kinetics(str(output_path))
            logger.info('INFO: Writing results to results.csv')
            exp.write_data(str(output_path))
        else:
            msg = '{} is not a directory!'.format(input_path)
            logger.critical('CRITICAL: '.format(msg))
            raise ValueError(msg)
    else:
        msg = '{} is not a directory!'.format(output_path)
        logger.critical('CRITICAL: '.format(msg))
        raise ValueError(msg)

if __name__ == "__main__":
    main()
