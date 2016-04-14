#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import warnings
import operator

try:
    import numpy as np
except ImportError:
    print('----- NumPy must be installed! -----')

try:
    from scipy import stats, optimize
except ImportError:
    print('----- SciPy must be installed! -----')


class Error(Exception):
    """
        Main Error class derived from Exception.
    """
    pass


class FitError(Error):
    """
        Exception raised if fitting fails.
    """
    pass


class Replicate():
    """
    Represents a single replicate within a measurement.

    Linear regression for v0 is executed at this level.

    Attributes:
        logger: logging.Logger instance inherited by the owning Measurement.
        num: float
            identification number of replicate within the experiment
        x, y: float
            time and signal to be analysed.
        owner: object
            measurement this measurement belongs to
        xlim:
        fitresult:
    """

    def __init__(self, num, xy, owner):
        """
        Inits Replicate class with provided data.

        Args:
            logger: logging.Logger instance inherited by the owning Experiment.
        xy: tuple of floats
            time and signal to be analysed.
        owner: object
            measurement this measurement belongs to
        """
        self.logger = owner.logger
        self.num = num + 1
        self.x, self.y = xy
        self.owner = owner
        self.xlim = owner.xlim
        self.fitresult = self.fit()

    def fit(self):
        ind_min_max = np.where((self.x >= self.xlim[0]) & (self.x <=
                                                           self.xlim[1]))
        x_for_fit = np.take(self.x, ind_min_max)
        y_for_fit = np.take(self.y, ind_min_max)

        # ignore warnings about invalid values in sqrt during linear fitting
        # they occur frequently and will just clutter the cli
        warnings.filterwarnings('ignore',
                                category=RuntimeWarning,
                                message='invalid value encountered in sqrt')

        (slope, intercept, r_value, p_value, std_err
         ) = stats.linregress(x_for_fit, y_for_fit)

        r_squared = r_value ** 2

        n = len(x_for_fit[0]) #  number of observations
        k = 2  # independent variables: slope and intercept
        
        # calculcate adjusted R²
        adj_r_squared = r_squared - (1 - r_squared) * k/(n - k - 1)
        
        conc = '{} {}'.format(self.owner.concentration,
                              self.owner.concentration_unit)

        self.logger.info('Linear fit for {} #{}:'.format(conc, self.num))
        if adj_r_squared < 0.9 and adj_r_squared > 0.7:
            msg = '    adjusted R² < 0.9; Check fit manually!'
            self.logger.warning(msg.format(round(adj_r_squared, 4)))
        elif adj_r_squared < 0.7:
            msg = '    adjusted R² < 0.7; Linear fit probably failed!'
            self.logger.warning(msg.format(round(adj_r_squared, 4)))
            
        self.logger.info('    R²/adjusted R²: {}/{}'.format(round(r_squared, 4), round(adj_r_squared, 4)))
        self.logger.info('    slope: {}'.format(slope))
        if slope < 0:
            self.logger.info('    → Slope is negative. Will use absolute '
                             'value for further calculations!')
        self.logger.info('    intercept: {}'.format(slope))

        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err
        }


class Measurement():
    """
    Represents a single measurement within an experiment.

    Each measurement consists of one ore more replicates and is associated
    with a specific substrate concentration.

    Attributes:
        logger: logging.Logger instance inherited by the owning Experiment.
        concentration: float
            substrate concentration associated with the measurement.
        concentration_unit: string
            unit of the concentration; only used for labeling plots/tables.
        x, y: float
            time and signal to be analysed.
        replicates: array_like
            list of individual replicates of the measurement.
        owner: object
            experiment this measurement belongs to
        xlim: array_like
            lower and upper bounds for calculating the v0 linear fit.
        avg_slope, avg_slope_err: float
            average slope (v0) of the measurement and its standard deviation
    """

    def __init__(self, xy, conc, conc_unit, owner):
        """
        Inits Measurement class with provided data.

        This is usually called from within an Experiment object.

        Args:
            logger: logging.Logger instance inherited by the owning Experiment.
        concentration: float
            substrate concentration associated with the measurement.
        concentration_unit: string
            unit of the concentration; only used for labeling plots/tables.
        xy: tuple of floats
            time and signal to be analysed.
        owner: object
            experiment this measurement belongs to
        """
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
            self.replicates.append(Replicate(n, (self.x, self.y[:, n:n + 1]),
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
        """
        Collect results of the individual replicates.

        Collects results of each replicate and append to results list.

        Returns:
            results: array_like
                list of results from each replicate of the measurement.
        """
        results = []
        for r in self.replicates:
            results.append(r.fitresult)

        return results


class Experiment():
    """
    Represents the actual experiment.

    Representation of a kinetics experiment. It consists of multiple
    objects of type Measurement.

    Attributes:
        logger: logging.Logger instance that is used for logging to console
            and log file.
        measurements: array_like
            list of individual measurements of the experiment.
            Individual measurements are sorted by their substrate
            concentration.
        fit_to_replicates: boolean
            whether to fit to individual replicates instead to the average of
            each measurement.
        raw_kinetic_data: dictionary
            storing x, y and std_err of each measurement for fitting kinetic
            curves.
        xlim: array_like
            lower and upper bounds for calculating the v0 linear fit.
    """

    def __init__(self, data_files, xlim, do_hill=False, no_mm=False,
                 fit_to_replicates=False, logger=None):
        """
        Inits Experiment class with experimental parameters

        This is the only class you should have to use directly in your program.
        Instances of Measurement and Replicate objects are created
        automatically using the provided data files.

        Args:
            data_files: list containing csv-formatted data files
            xlim: tuple of float values defining the lower and upper bound for
                linear fitting of v0
            do_hill:  boolean to define whether to fit Hill-type kinetics in
                addition to Michaelis-Menten kinetics. Defaults to False
            no_mm: boolean to define whether the calculation of Michaelis-Menten
                kinetics is supressed. Defaults to False
            fit_to_replicates: boolean to define wheter to fit to individual
                replicates instead of the avarage slope. Defaults to False
            logger: logging.Logger instance. If not given, a new logger is
                created
        """

        # check if a logger was handed over; if not, create a new instance
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # collction of indepentend measurements
        self.measurements = []

        self.fit_to_replicates = fit_to_replicates
        # dictionary to store data for the kinetics calculation
        self.raw_kinetic_data = {'x': [], 'y': [], 'yerr': []}
        self.xlim = xlim

        # parse data files and generate measurements
        for csvfile in data_files:
            try:
                tmp = np.genfromtxt(str(csvfile), comments='#')
                with open(str(csvfile)) as datafile:
                    head = [next(datafile) for x in range(2)]
            except OSError:
                msg = "Failed reading file {}".format(str(csvfile))
                self.logger.error(msg)
                raise

            # extract concentration and unit from header
            # TODO: move unit to parameter
            # check header for correct number of items
            if len(head) < 2 or len(head) > 2:
                msg = 'Parsing header of data files failed! Wrong format?'
                self.logger.error(msg)
                raise Error(msg)
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

        # sort measurements by concentration
        self.measurements = sorted(self.measurements,
                                   key=operator.attrgetter('concentration'))

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
        if not no_mm:
            self.mm = self.do_mm_kinetics()
        else:
            self.mm = None
        if do_hill:
            self.hill = self.do_hill_kinetics()
        else:
            self.hill = None

    @staticmethod
    def mm_kinetics_function(x, vmax, Km):
        """
        Michaelis-Menten function.

        Classical Michaelis-Menten enzyme kinetics function.

        Args:
            x: float
                concentration at velocity v
            vmax: float
                maximum velocity
            Km: float
                Michaelis constant

        Returns:
            v: float
                velocity at given concentration x
        """
        v = (vmax * x) / (Km + x)
        return v

    @staticmethod
    def hill_kinetics_function(x, vmax, Kprime, h):
        """
        Hill function.

        Hill function for enzyme kinetics with cooperativity.

        Args:
            x: float
                concentration at velocity v.
            vmax: float
                maximum velocity.
            Kprime: float
                kinetics constant related to Michaelis constant.
            h: float
                hill slope; if h=1, Hill function is identical to
                Michaelis-Menten function.

        Returns:
            v: float
                velocity at given concentration x
        """
        v = (vmax * (x ** h)) / (Kprime + (x ** h))
        return v

    def do_mm_kinetics(self):
        """
        Calculates Michaelis-Menten kinetics.

        Returns:
            On success, returns a dictionary containing the kinetic parameters
            and their errors:

            {'vmax': float,
             'Km': float,
             'vmax_err': float,
             'Km_err': float,
             'x': array_like}

        Raises:
            FitError if fitting fails.
        """
        try:
            popt, pconv = optimize.curve_fit(self.mm_kinetics_function,
                                             self.raw_kinetic_data['x'],
                                             self.raw_kinetic_data['y'])
        except ValueError:
            msg = ('Calculation of Michaelis-Menten kinetics failed! Your '
                   'input data (either x or y) contain empty (NaN) values!')
            if self.logger:
                self.logger.error('{}'.format(msg))
            raise FitError(msg)

        perr = np.sqrt(np.diag(pconv))
        vmax = popt[0]
        Km = popt[1]
        x = np.arange(0, max(self.raw_kinetic_data['x']), 0.0001)

        self.logger.info('Michaelis-Menten Kinetics:')
        self.logger.info('    v_max: {} ± {}'.format(vmax, perr[0]))
        self.logger.info('    Km: {} ± {}'.format(Km, perr[1]))

        return {
            'vmax': np.float(vmax),
            'Km': np.float(Km),
            'vmax_err': np.float(perr[0]),
            'Km_err': np.float(perr[1]),
            'x': x
        }

    def do_hill_kinetics(self):
        """
        Calculates Hill kinetics.

        Returns:
            On success, returns a dictionary containing the kinetic parameters
            their errors:

            {'vmax': float,
             'Kprime': float,
             'vmax_err': float,
             'Km_prime': float,
             'h_err': float,
             'h': float,
             'x': array_like}

        Raises:
            FitError if fitting fails.
        """
        try:
            popt, pconv = optimize.curve_fit(self.hill_kinetics_function,
                                             self.raw_kinetic_data['x'],
                                             self.raw_kinetic_data['y'])
        except ValueError:
            msg = ('Calculation of Hill kinetics failed! Your input data '
                   '(either x or y) contains empty (NaN) values!')
            if self.logger:
                self.logger.error('{}'.format(msg))
            raise FitError(msg)

        perr = np.sqrt(np.diag(pconv))
        vmax = popt[0]
        Kprime = popt[1]
        h = popt[2]

        x = np.arange(0, max(self.raw_kinetic_data['x']), 0.0001)

        self.logger.info('Hill Kinetics:')
        self.logger.info('    v_max: {} ± {}'.format(vmax, perr[0]))
        self.logger.info('    K_prime: {} ± {}'.format(Kprime, perr[1]))
        self.logger.info('    h: {} ± {}'.format(h, perr[2]))

        return {
            'vmax': np.float(vmax),
            'Kprime': np.float(Kprime),
            'vmax_err': np.float(perr[0]),
            'Kprime_err': np.float(perr[1]),
            'h_err': np.float(perr[2]),
            'h': np.float(h),
            'x': x
        }
