#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy import stats, optimize
import numpy as np
import logging
import warnings


class Replicate():
    """

    """

    def __init__(self, num, xy, owner):
        self.logger = owner.logger
        self.num = num + 1
        self.x, self.y = xy
        self.owner = owner
        self.xlim = owner.xlim
        self.fitresult = self.fit()

    def fit(self):
        ind_min_max = np.where((self.x >= self.xlim[0]) &
                               (self.x <= self.xlim[1]))
        x_for_fit = np.take(self.x, ind_min_max)
        y_for_fit = np.take(self.y, ind_min_max)

        # ignore warnings about invalid values in sqrt during linear fitting
        # they occur frequently and will just clutter the cli
        warnings.filterwarnings('ignore',
                                category=RuntimeWarning,
                                message='invalid value encountered in sqrt')

        (slope, intercept,
            r_value,
            p_value,
            std_err) = stats.linregress(x_for_fit, y_for_fit)

        r_squared = r_value**2
        conc = '{} {}'.format(self.owner.concentration,
                              self.owner.concentration_unit)

        self.logger.info('Linear fit for {} #{}:'.format(conc, self.num))
        if r_squared < 0.9 and r_squared > 0.7:
            msg = '    r-squared: {} < 0.9; Check fit manually!'
            self.logger.warning(msg.format(round(r_squared, 4)))
        elif r_squared < 0.7:
            msg = '    r-squared: {} < 0.7; Linear fit probably failed!'
            self.logger.warning(msg.format(round(r_squared, 4)))
        else:
            msg = '    r-squared: {}'
            self.logger.info(msg.format(round(r_squared, 4)))
        self.logger.info('    slope: {}'.format(slope))
        if slope < 0:
            self.logger.info('    Slope is negative. Will use absolute '
                             'value for further calculations!')
        self.logger.info('    intercept: {}'.format(slope))

        return {'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_err': std_err}


class Measurement():

    def __init__(self, xy, conc, conc_unit, owner):
        self.logger = owner.logger
        self.concentration = float(conc)
        self.concentration_unit = conc_unit
        self.x, self.y = xy
        self.replicates = []
        self.owner = owner
        self.xlim = owner.xlim
        self.slopes = []
        self.avg_slope = None
        self.avg_slope_err = None

        # extract number of replicates (num_replicates)
        length_x, num_replicates = self.y.shape

        for n in range(num_replicates):
            self.replicates.append(Replicate(n,
                                             (self.x, self.y[:, n:n+1]),
                                             self))

        for r in self.replicates:
            self.slopes.append(r.fitresult['slope'])

        self.avg_slope = np.average(self.slopes)
        self.avg_slope_err = np.std(self.slopes)

        self.logger.info('Average slope: {} ± {}'.format(self.avg_slope,
                                                         self.avg_slope_err))
        if self.avg_slope < 0:
            self.logger.info('Avererage slope is negative. Will use '
                             'absolute value for further calculations!')
        self.logger.info('-----')

    def get_results(self):
        results = []
        for r in self.replicates:
            results.append(r.fitresult)

        return results


class Experiment():

    def __init__(self, data_files, xlim, do_hill=False,
                 fit_to_replicates=False, logger=None):

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # collction of indepentend measurements
        self.measurements = []

        self.fit_to_replicates = fit_to_replicates
        # dictionary to store data for the kinetics calculation
        self.raw_kinetic_data = {'x': [],
                                 'y': [],
                                 'yerr': []}
        self.xlim = xlim

        # parse data files and generate measurements
        for csvfile in data_files:
            tmp = np.genfromtxt(str(csvfile), comments='#')
            with open(str(csvfile)) as datafile:
                head = [next(datafile) for x in range(2)]
            # extract concentration and unit from header
            # TODO: move unit to parameter
            # TODO: More error-proof header detection
            conc = head[0].strip('#').strip()
            unit = head[1].strip('#').strip()
            # split x and y data apart
            # first column is x data (time); following columns contain
            # replicates of y data (absorption, etc.)
            x = tmp[:, 0]
            y = tmp[:, 1:]
            # create new measurement and append to list
            measurement = Measurement((x, y), conc, unit, self)
            self.measurements.append(measurement)

        # iterate over all measurements
        for m in self.measurements:
            if self.fit_to_replicates:
                for r in m.replicates:
                    self.raw_kinetic_data['x'].append(m.concentration)
                    self.raw_kinetic_data['y'].append(
                        np.absolute(r.fitresult['slope']))
                    self.raw_kinetic_data['yerr'].append(
                        r.fitresult['std_err'])

            else:
                # extract relevant data for kinetics calculation
                # (concentration, average slope and error)
                self.raw_kinetic_data['x'].append(m.concentration)
                self.raw_kinetic_data['y'].append(np.absolute(m.avg_slope))
                self.raw_kinetic_data['yerr'].append(m.avg_slope_err)

        # calculate kinetics
        self.mm = self.do_mm_kinetics()
        if do_hill:
            self.hill = self.do_hill_kinetics()
        else:
            self.hill = None

    def plot_data(self, outpath):
        # iterate over all measurements
        for m in self.measurements:
            # plot each measurement
            m.plot(outpath)

    def mm_kinetics_function(self, x, vmax, Km):
        v = (vmax*x)/(Km+x)
        return v

    def hill_kinetics_function(self, x, vmax, Kprime, h):
        v = (vmax*(x**h))/(Kprime+(x**h))
        return v

    def do_mm_kinetics(self):
        try:
            popt, pconv = optimize.curve_fit(self.mm_kinetics_function,
                                             self.raw_kinetic_data['x'],
                                             self.raw_kinetic_data['y'])

            perr = np.sqrt(np.diag(pconv))
            vmax = popt[0]
            Km = popt[1]
            x = np.arange(0, max(self.raw_kinetic_data['x']), 0.0001)

            self.logger.info('Michaelis-Menten Kinetics:')
            self.logger.info('    v_max: {} ± {}'.format(vmax, perr[0]))
            self.logger.info('    Km: {} ± {}'.format(Km, perr[1]))

            return {'vmax': float(vmax),
                    'Km': float(Km),
                    'perr': perr,
                    'x': x}
        except:
            msg = 'Calculation of Michaelis-Menten kinetics failed!'
            if self.logger:
                self.logger.error('{}'.format(msg))
            else:
                print(msg)
            return None

    def do_hill_kinetics(self):
        try:
            popt, pconv = optimize.curve_fit(self.hill_kinetics_function,
                                             self.raw_kinetic_data['x'],
                                             self.raw_kinetic_data['y'])

            perr = np.sqrt(np.diag(pconv))
            vmax = popt[0]
            Kprime = popt[1]
            h = popt[2]

            x = np.arange(0, max(self.raw_kinetic_data['x']), 0.0001)

            self.logger.info('Hill Kinetics:')
            self.logger.info('    v_max: {} ± {}'.format(vmax, perr[0]))
            self.logger.info('    K_prime: {} ± {}'.format(Kprime, perr[1]))
            self.logger.info('    h: {} ± {}'.format(h, perr[2]))

            return {'vmax': float(vmax),
                    'Kprime': float(Kprime),
                    'perr': perr,
                    'h': h,
                    'x': x}
        except:
            msg = 'Calculation of Hill kinetics failed!'
            if self.logger:
                self.logger.error('{}'.format(msg))
            else:
                print(msg)
            return None
