#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import logging
import csv
import sys
from pathlib import Path

# check for third party modules. Matplotlib and NumPy are essential
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('----- Matplotlib must be installed! -----')

try:
    import numpy as np
except ImportError:
    print('----- NumPy must be installed! -----')

# colorlog is optional; inform user of non-colored output if not found
try:
    from colorlog import ColoredFormatter
    COLORED_LOG = True
except ImportError:
    COLORED_LOG = False
    print('----- Module colorlog not found. Console output will not be '
          'colored! -----')
    sys.exc_clear()

# check for libkinetics; if not found, there is something wrong with the
# installation if pyKinetics
try:
    import libkinetics
except ImportError:
    print('---- Cannot find libkinetics! Check your installation! -----')


class ExperimentHelper():
    """
    Helper class for dealing with results of libkinetics.

    Provides plotting and data output functionality for libkinetics.Experiment.
    """

    def __init__(self, experiment, logger, unit, logx=False, logy=False):
        self.exp = experiment
        self.logger = logger
        self.logx = logx
        self.logy = logy
        self.unit = unit

    def linear_regression_function(self, slope, x, intercept):
        y = slope * x + intercept
        return y

    def plot_data(self, outpath):
        # iterate over all measurements
        for m in self.exp.measurements:
            # plot each measurement
            fig, ax = plt.subplots()
            ax.set_xlabel('Time')
            ax.set_ylabel('Raw Signal')
            ax_title = 'Linear regression {} {}'.format(m.concentration,
                                                        m.concentration_unit)
            ax.set_title(ax_title)

            for r in m.replicates:
                ax.plot(r.x,
                        r.y,
                        linestyle='None',
                        marker='o',
                        ms=3,
                        fillstyle='none',
                        label='replicate #{}'.format(r.num))
                y = self.linear_regression_function(r.fitresult['slope'], r.x,
                                                    r.fitresult['intercept'])
                ax.plot(r.x, y, 'k-')
                ax.axvspan(m.xlim[0], m.xlim[1], facecolor='0.8', alpha=0.5)

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig('{}/fit_{}_{}.png'.format(outpath, m.concentration,
                                                  m.concentration_unit),
                        bbox_inches='tight')
            plt.close(fig)

    def plot_kinetics(self, outpath):
        exp = self.exp
        fig, ax = plt.subplots()
        ax.set_xlabel('c [{}]'.format(self.unit))
        ax.set_ylabel('dA/dt [Au/s]')
        ax.set_title('Kinetics')
        
        if self.logx:
            ax.set_xscale('log')
        if self.logy:
            ax.set_yscale('log')

        ax.errorbar(exp.raw_kinetic_data['x'],
                    exp.raw_kinetic_data['y'],
                    yerr=exp.raw_kinetic_data['yerr'],
                    fmt='ok',
                    ms=3,
                    fillstyle='none',
                    label="Data with error")

        if exp.mm:
            y = exp.mm_kinetics_function(exp.mm['x'], exp.mm['vmax'],
                                         exp.mm['Km'])
            ax.plot(exp.mm['x'], y, 'b-', label="Michaelis-Menten")
        if exp.hill:
            y = exp.hill_kinetics_function(exp.hill['x'], exp.hill['vmax'],
                                           exp.hill['Kprime'], exp.hill['h'])
            ax.plot(exp.hill['x'], y, 'g-', label="Hill")

        ax.legend(loc='best', fancybox=True)
        plt.savefig('{}/kinetics.png'.format(outpath), bbox_inches='tight')
        plt.close(fig)

    def write_data(self, outpath):

        exp = self.exp

        with open('{}/results.csv'.format(outpath),
                  'w',
                  newline='\n') as csvfile:

            writer = csv.writer(csvfile, dialect='excel-tab')
            writer.writerow(['LINEAR FITS'])
            writer.writerow([])
            writer.writerow(['concentration', 'avg. slope', 'slope std_err',
                             'replicates (slope, intercept and r_squared)'])
            for m in exp.measurements:
                row = [m.concentration, m.avg_slope, m.avg_slope_err]
                for r in m.replicates:
                    row.append(r.fitresult['slope'])
                    row.append(r.fitresult['intercept'])
                    row.append(r.fitresult['r_squared'])
                writer.writerow(row)

            writer.writerow([])
            if exp.mm:
                self.logger.debug('    Writing Michaelis-Menten results.')
                writer.writerow(['MICHAELIS-MENTEN KINETICS'])
                writer.writerow(['', 'value', 'std'])
                writer.writerow(['vmax', exp.mm['vmax'], exp.mm['vmax_err']])
                writer.writerow(['Kprime', exp.mm['Km'], exp.mm['Km_err']])
            if exp.hill:
                self.logger.debug('    Writing Hill results.')
                writer.writerow([])
                writer.writerow(['HILL KINETICS'])
                writer.writerow(['', 'value', 'std'])
                writer.writerow(['vmax', exp.hill['vmax'], exp.hill['vmax_err']
                                 ])
                writer.writerow(['Kprime', exp.hill['Kprime'],
                                 exp.hill['Kprime_err']])
                writer.writerow(['h', exp.hill['h'], exp.hill['h_err']])


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
    parser.add_argument('-nm',
                        '--no-michaelis',
                        action='store_true',
                        help='do not compute michaelis-menten kinetics')
    parser.add_argument('-wh',
                        '--hill',
                        action='store_true',
                        help='compute additional kinetics using Hill equation')
    parser.add_argument('-lx',
                        '--log-x',
                        action='store_true',
                        help='x axis in kinetics is log-scaled')
    parser.add_argument('-ly',
                        '--log-y',
                        action='store_true',
                        help='y axis in kinetics is log-scaled')
    parser.add_argument('start',
                        type=np.float64,
                        help='start of fitting window')
    parser.add_argument('end', type=np.float64, help='end of fitting window')
    parser.add_argument('input',
                        type=str,
                        help='directory containing input files in csv format')
    parser.add_argument('output',
                        type=str,
                        help='results will be written to this directory')
    parser.add_argument('unit',
                        type=str,
                        help='unit of concentrations')

    args = parser.parse_args()

    return args


def initialize_logger():
    """
    Initialization of logging subsystem.

    Two logging handlers are brought up: 'fh' which logs to a log file and
    'ch' which logs to standard output. If colorlog is installed, logging
    to console will be colored.

    Returns:
        logger: logging.Logger instance
    """
    if COLORED_LOG:
        fmt = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
        formatter = ColoredFormatter(fmt,
                                     datefmt=None,
                                     reset=True,
                                     log_colors={
                                         'DEBUG': 'cyan',
                                         'INFO': 'green',
                                         'WARNING': 'yellow',
                                         'ERROR': 'red',
                                         'CRITICAL': 'red,bg_white'
                                     },
                                     secondary_log_colors={},
                                     style='%')
    else:
        fmt = '%(levelname)-8s%(message)s'
        formatter = logging.Formatter(fmt, datefmt=None, style='%')

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
    """
    Main method of pyKinetics.

    Will be executed when analyze-cli.py is started from the command line.
    """
    # parse command line arguments
    args = parse_arguments()
    # initialize logger
    logger = initialize_logger()
    # grab fitting window from provided arguments
    fitting_window = (args.start, args.end)
    
    # scaling of axes in kinetics plot
    if args.log_x:
        logx = args.log_x
    else:
        logx = False
        
    if args.log_y:
        logy = args.log_y
    else:
        logy = False
        
    # suppress calculation of Michaelis-Menten kinetics
    if args.no_michaelis:
        no_mm = args.no_michaelis
    else:
        no_mm = False

    # do Hill kinetics
    if args.hill:
        do_hill = args.hill
    else:
        do_hill = False

    # perform global fit of kinetic function(s) to all replicates
    # instead of their means
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
                                         fitting_window,
                                         do_hill=do_hill,
                                         no_mm=no_mm,
                                         logger=logger,
                                         fit_to_replicates=fit_to_replicates)
            ehlp = ExperimentHelper(exp, logger, args.unit, logx=logx, logy=logy)
            logger.info('Plotting linear fits')
            ehlp.plot_data(str(output_path))
            logger.info('Plotting kinetic fit(s)')
            ehlp.plot_kinetics(str(output_path))
            logger.info('Writing results to results.csv')
            ehlp.write_data(str(output_path))
            logger.info('Finished!')
        else:
            msg = '{} is not a directory!'.format(input_path)
            logger.critical('CRITICAL: '.format(msg))
            raise ValueError(msg)
    else:
        msg = '{} is not a directory!'.format(output_path)
        logger.critical('CRITICAL: {}'.format(msg))
        raise ValueError(msg)


if __name__ == "__main__":
    main()
