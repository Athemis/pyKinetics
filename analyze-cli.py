#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from colorlog import ColoredFormatter

import libkinetics


class ExperimentHelper():

    def __init__(self, experiment, logger):
        self.exp = experiment
        self.logger = logger

    def linear_regression_function(self, slope, x, intercept):
        y = slope * x + intercept
        return y

    def plot_data(self, exp, outpath):
        # iterate over all measurements
        for m in exp.measurements:
            # plot each measurement
            fig, ax = plt.subplots()
            ax.set_xlabel('Time')
            ax.set_ylabel('Raw Signal')
            ax_title = 'Linear regression {} {}'.format(m.concentration,
                                                        m.concentration_unit)
            ax.set_title(ax_title)

            for r in m.replicates:
                ax.plot(r.x, r.y, linestyle='None',
                        marker='o', ms=3, fillstyle='none',
                        label='replicate #{}'.format(r.num))
                y = self.linear_regression_function(r.fitresult['slope'],
                                                    r.x,
                                                    r.fitresult['intercept'])
                ax.plot(r.x, y, 'k-')
                ax.axvspan(m.xlim[0], m.xlim[1], facecolor='0.8', alpha=0.5)

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig('{}/fit_{}_{}.png'.format(outpath,
                                                  m.concentration,
                                                  m.concentration_unit),
                        bbox_inches='tight')
            plt.close(fig)

    def plot_kinetics(self, exp, outpath):
        fig, ax = plt.subplots()
        ax.set_xlabel('c [mM]')
        ax.set_ylabel('dA/dt [Au/s]')
        ax.set_title('Kinetics')

        ax.errorbar(exp.raw_kinetic_data['x'],
                    exp.raw_kinetic_data['y'],
                    yerr=exp.raw_kinetic_data['yerr'],
                    fmt='ok', ms=3, fillstyle='none', label="Data with error")

        if exp.mm:
            y = exp.mm_kinetics_function(exp.mm['x'],
                                         exp.mm['vmax'],
                                         exp.mm['Km'])
            ax.plot(exp.mm['x'], y, 'b-', label="Michaelis-Menten")
        if exp.hill:
            y = exp.hill_kinetics_function(exp.hill['x'],
                                           exp.hill['vmax'],
                                           exp.hill['Kprime'],
                                           exp.hill['h'])
            ax.plot(exp.hill['x'], y, 'g-', label="Hill")

        ax.legend(loc='best', fancybox=True)
        plt.savefig('{}/kinetics.png'.format(outpath), bbox_inches='tight')
        plt.close(fig)

    def write_data(self, exp, outpath):

        with open('{}/results.csv'.format(outpath),
                  'w',
                  newline='\n') as csvfile:

            writer = csv.writer(csvfile, dialect='excel-tab')
            writer.writerow(['# LINEAR FITS'])
            writer.writerow([])
            writer.writerow(['# concentration',
                             'avg. slope',
                             'slope std_err',
                             'replicates (slope, intercept and r value)'])
            for m in exp.measurements:
                row = [m.concentration, m.avg_slope, m.avg_slope_err]
                for r in m.replicates:
                    row.append(r.fitresult['slope'])
                    row.append(r.fitresult['intercept'])
                    row.append(r.fitresult['r_value'])
                writer.writerow(row)

            writer.writerow([])
            if exp.mm:
                writer.writerow(['# MICHAELIS-MENTEN KINETICS'])
                writer.writerow(['# vmax', 'Km'])
                writer.writerow([exp.mm['vmax'], exp.mm['Km']])
            if exp.hill:
                writer.writerow(['# HILL KINETICS'])
                writer.writerow(['# vmax', 'Kprime', 'h'])
                writer.writerow([exp.hill['vmax'], exp.hill['Kprime'],
                                 exp.hill['h']])


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='increase output verbosity')
    parser.add_argument('-r',
                        '--replicates',
                        action='store_true',
                        help='fit kinetics to individual replicates')
    parser.add_argument('-wh',
                        '--hill',
                        action='store_true',
                        help='compute additional kinetics using Hill equation')
    parser.add_argument('start',
                        type=np.float64,
                        help='start of fitting window')
    parser.add_argument('end',
                        type=np.float64,
                        help='end of fitting window')
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
    fmt = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
    formatter = ColoredFormatter(fmt,
                                 datefmt=None,
                                 reset=True,
                                 log_colors={'DEBUG': 'cyan',
                                             'INFO': 'green',
                                             'WARNING': 'yellow',
                                             'ERROR': 'red',
                                             'CRITICAL': 'red,bg_white'},
                                 secondary_log_colors={},
                                 style='%')

    logging.captureWarnings(True)
    logger = logging.getLogger('pyKinetics-cli')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    try:
        log_filename = 'pyKinetics-cli.log'
        fh = logging.FileHandler(log_filename, 'w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    except IOError as error:
        logger.warning('Cannot create log file! Run pyKinetics-cli'
                       'from a directory to which you have write access.')
        logger.warning(error.msg)
        pass

    return logger


def main():
    # parse command line arguments
    args = parse_arguments()
    # initialize logger
    logger = initialize_logger()

    xlim = (args.start, args.end)

    if args.hill:
        do_hill = args.hill
    else:
        do_hill = False

    if args.replicates:
        fit_to_replicates = args.replicates
    else:
        fit_to_replicates = False

    try:
        input_path = Path(args.input).resolve()
    except FileNotFoundError:
        logger.critical('Path containing input data '
                        'not found: {}'.format(args.input))
        raise
    try:
        output_path = Path(args.output).resolve()
    except FileNotFoundError:
        logger.critical('Path for writing results '
                        'not found: {}'.format(args.output))
        raise

    if output_path.is_dir():
        if input_path.is_dir():
            logger.info('Collecting data files')
            data_files = sorted(input_path.glob('**/*.csv'))
            msg = 'Calculating kinetics'
            if do_hill:
                msg = '{} including Hill kinetics'.format(msg)
            logger.info('{}'.format(msg))
            exp = libkinetics.Experiment(data_files,
                                         xlim,
                                         do_hill=do_hill,
                                         logger=logger,
                                         fit_to_replicates=fit_to_replicates)
            ehlp = ExperimentHelper(exp, logger)
            logger.info('Plotting linear fits to data')
            ehlp.plot_data(exp, str(output_path))
            logger.info('Plotting kinetics fit(s)')
            ehlp.plot_kinetics(exp, str(output_path))
            logger.info('Writing results to results.csv')
            ehlp.write_data(exp, str(output_path))
            logger.info('Finished!')
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
