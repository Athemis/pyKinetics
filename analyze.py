#!/usr/bin/python

from scipy import stats, optimize
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv


class Replicate():
    def __init__(self, x, y, owner):
        self.x = x
        self.y = y
        self.owner = owner
        self.xlim = owner.xlim
        self.fitresult = self.fit()

    def fit(self):
        ind_min_max = np.where((self.x >= self.xlim[0]) &
                               (self.x <= self.xlim[1]))
        x_for_fit = np.take(self.x, ind_min_max)
        y_for_fit = np.take(self.y, ind_min_max)

        (slope, intercept,
            r_value,
            p_value,
            std_err) = stats.linregress(x_for_fit, y_for_fit)

        return {'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err}


class Measurement():

    def __init__(self, x, y, conc, conc_unit, owner):
        self.concentration = float(conc)
        self.concentration_unit = conc_unit
        self.x = x
        self.y = y
        self.replicates = []
        self.owner = owner
        self.xlim = owner.xlim
        self.slopes = []
        self.avg_slope = None
        self.avg_slope_err = None

        # extract number of replicates (num_replicates)
        length_x, num_replicates = self.y.shape

        for n in range(num_replicates):
            self.replicates.append(Replicate(self.x, self.y[:, n:n+1], self))

        for r in self.replicates:
            self.slopes.append(r.fitresult['slope'])

        self.avg_slope = np.average(self.slopes)
        self.avg_slope_err = np.std(self.slopes)

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Absorption (340 nm) [Au]')
        ax.set_title('Linear regression {} {}'.format(self.concentration,
                                                      self.concentration_unit))

        for r in self.replicates:
            ax.plot(r.x, r.y, linestyle='None',
                    marker='o', ms=3, fillstyle='none')
            ax.plot(r.x, r.fitresult['slope']*r.x+r.fitresult['intercept'],
                    'k-')
            ax.axvspan(self.xlim[0], self.xlim[1], facecolor='0.8', alpha=0.5)

        plt.savefig('out/fit_{}_{}.png'.format(self.concentration,
                                               self.concentration_unit),
                    bbox_inches='tight')
        plt.close(fig)

    def get_results(self):
        results = []
        for r in self.replicates:
            results.append(r.fitresult)

        return results


class Experiment():

    def __init__(self, data_files, xlim):

        # collction of indepentend measurements
        self.measurements = []
        # dictionary to store data for the kinetics calculation
        self.raw_kinetic_data = {'x': [],
                                 'y': [],
                                 'yerr': []}
        self.xlim = xlim
        
        # variables to store calculated Michaelis-Menten and Hill kinetics
        self.mm = None
        self.hill = None

        # parse data files and generate measurements
        for csvfile in data_files:
            tmp = np.genfromtxt(csvfile, comments='#')
            with open(csvfile) as datafile:
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
            measurement = Measurement(x, y, conc, unit, self)
            self.measurements.append(measurement)
        
        # iterate over all measurements
        for m in self.measurements:
            # extract relevant data for kinetics calculation (concentration, 
            # average slope and error)
            self.raw_kinetic_data['x'].append(m.concentration)
            self.raw_kinetic_data['y'].append(np.absolute(m.avg_slope))
            self.raw_kinetic_data['yerr'].append(m.avg_slope_err)

    def plot_data(self):
        for m in self.measurements:
            m.plot()

    def mm_kinetics_function(self, x, vmax, Km):
        v = (vmax*x)/(Km+x)
        return v

    def hill_kinetics_function(self, x, vmax, Kprime, h):
        v = (vmax*(x**h))/(Kprime+(x**h))
        return v

    def do_mm_kinetics(self):
        popt, pconv = optimize.curve_fit(self.mm_kinetics_function,
                                         self.raw_kinetic_data['x'],
                                         self.raw_kinetic_data['y'])

        perr = np.sqrt(np.diag(pconv))
        vmax = popt[0]
        Km = popt[1]
        x = np.arange(0, max(self.raw_kinetic_data['x']), 0.0001)

        return {'vmax': float(vmax),
                'Km': float(Km),
                'perr': perr,
                'x': x}

    def do_hill_kinetics(self):
        popt, pconv = optimize.curve_fit(self.hill_kinetics_function,
                                         self.raw_kinetic_data['x'],
                                         self.raw_kinetic_data['y'])

        perr = np.sqrt(np.diag(pconv))
        vmax = popt[0]
        Kprime = popt[1]
        h = popt[2]

        x = np.arange(0, max(self.raw_kinetic_data['x']), 0.0001)

        return {'vmax': float(vmax),
                'Kprime': float(Kprime),
                'perr': perr,
                'h': h,
                'x': x}

    def plot_kinetics(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('c [mM]')
        ax.set_ylabel('dA/dt [Au/s]')
        ax.set_title('Kinetics')

        ax.errorbar(self.raw_kinetic_data['x'],
                    self.raw_kinetic_data['y'],
                    yerr=self.raw_kinetic_data['yerr'],
                    fmt='ok', ms=3, fillstyle='none', label="Data with error")
        # linestyle='None', marker='o', ms=3, fillstyle='none'
        self.mm = self.do_mm_kinetics()
        self.hill = self.do_hill_kinetics()

        print(self.mm)
        print(self.hill)

        ax.plot(self.mm['x'],
                (self.mm['vmax']*self.mm['x'])/(self.mm['Km']+self.mm['x']),
                'b-', label="Michaelis-Menten")
        ax.plot(self.hill['x'],
                ((self.hill['vmax']*(self.hill['x']**self.hill['h'])) /
                    (self.hill['Kprime'] + (self.hill['x']**self.hill['h']))),
                'g-', label="Hill")

        ax.legend(loc='best', fancybox=True)
        plt.savefig('out/kinetics.png', bbox_inches='tight')
        plt.close(fig)

    def write_data(self):

        with open('out/results.csv', 'w', newline='\n') as csvfile:

            writer = csv.writer(csvfile, dialect='excel-tab')
            writer.writerow(['# LINEAR FITS'])
            writer.writerow([])
            writer.writerow(['# concentration',
                             'avg. slope',
                             'slope std_err',
                             'replicates (slope, intercept and r value)'])
            for m in self.measurements:
                row = [m.concentration, m.avg_slope, m.avg_slope_err]
                for r in m.replicates:
                    row.append(r.fitresult['slope'])
                    row.append(r.fitresult['intercept'])
                    row.append(r.fitresult['r_value'])
                writer.writerow(row)

            writer.writerow([])
            writer.writerow(['# MICHAELIS-MENTEN KINETICS'])
            writer.writerow(['# vmax', 'Km'])
            writer.writerow([self.mm['vmax'], self.mm['Km']])

            writer.writerow(['# HILL KINETICS'])
            writer.writerow(['# vmax', 'Kprime', 'h'])
            writer.writerow([self.hill['vmax'], self.hill['Kprime'],
                             self.hill['h']])


def main():
    data_files = glob.glob('csv/*.csv')
    exp = Experiment(data_files, (10, 25))
    exp.plot_data()
    exp.plot_kinetics()
    exp.write_data()

if __name__ == "__main__":
    main()
