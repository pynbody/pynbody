"""

ionfrac
=======

calculates ionization fractions - NEEDS DOCUMENTATION

"""

import abc
import logging
import os

import numpy as np

logger = logging.getLogger('pynbody.analysis.ionfrac')

from .interpolate import interpolate3d


def _run_cloudy(redshift, log_temp, log_den, cloudy_path= '/Users/app/Science/cloudy/source/cloudy.exe'):
    """
    Run cloudy and return the output ionisation fractions
    """
    template = """title pynbody_grid_run
    cmb z={redshift}
    table hm05 z = {redshift}
    hden {log_hden}
    constant temperature {temperature}
    stop zone 1
    """

    input = template.format(redshift=redshift, log_hden=log_den, temperature=10**log_temp)

    # remove any indentation from the input and replace newlines with '\n'
    input = '\n'.join([x.strip() for x in input.split('\n')])
    print(input)

    import subprocess

    # Start the subprocess
    process = subprocess.Popen(
        [cloudy_path],  # Replace with your command and arguments
        stdin=subprocess.PIPE,  # Allows writing to stdin
        stdout=subprocess.PIPE,  # Allows reading from stdout
        stderr=subprocess.PIPE,  # Capture stderr (optional)
        text=True  # Ensures that communication is in string format (instead of bytes)
    )

    process.stdin.write(input)
    process.stdin.flush()  # Flush the input to ensure it is sent

    stdout, stderr = process.communicate()  # Waits for the process to complete and fetches output

    # search for "Log10 Mean Ionisation" in the output
    table_start_line_number = None
    out_lines = stdout.split('\n')
    for i, line in enumerate(out_lines):
        if "Log10 Mean Ionisation (over radius)" in line:
            table_start_line_number = i
            break

    if table_start_line_number is None:
        raise ValueError("Could not find ionisation table in cloudy output")

    element_symbols = {
        "Hydrogen": "H",
        "Helium": "He",
        "Lithium": "Li",
        "Beryllium": "Be",
        "Boron": "B",
        "Carbon": "C",
        "Nitrogen": "N",
        "Oxygen": "O",
        "Fluorine": "F",
        "Neon": "Ne",
        "Sodium": "Na",
        "Magnesium": "Mg",
        "Aluminium": "Al",
        "Silicon": "Si",
        "Phosphorus": "P",
        "Sulphur": "S",
        "Chlorine": "Cl",
        "Argon": "Ar",
        "Potassium": "K",
        "Calcium": "Ca",
        "Scandium": "Sc",
        "Titanium": "Ti",
        "Vanadium": "V",
        "Chromium": "Cr",
        "Manganese": "Mn",
        "Iron": "Fe",
        "Cobalt": "Co",
        "Nickel": "Ni",
        "Copper": "Cu",
        "Zinc": "Zn"
    }


    def _process_line(line):
        element_name = line[1:11].strip()
        ion_stages = []
        for i in range(17):
            this_ion = line[11+i*7:18+i*7].strip()
            try:
                ion_stages.append(float(this_ion))
            except ValueError:
                break
        print(element_symbols[element_name], ion_stages)

    _process_line(out_lines[table_start_line_number])
    _process_line(out_lines[table_start_line_number+1])
    _process_line(out_lines[table_start_line_number + 2])
    _process_line(out_lines[table_start_line_number + 3])





def generate_cloudy_ionfrac_table(cloudy_path = '/Users/app/Science/cloudy/source/cloudy.exe',
                                  redshift_range = (0, 15),
                                  log_temp_range = (2.0, 8.0),
                                  log_den_range = (-8.0, 2.0)):
    """
    Generate a table of ion fractions using cloudy
    """

    template = """title pynbody_grid_run
cmb z={redshift}
table hm05 z = {redshift}
hden {log_hden}
constant temperature {temperature}
stop zone 1
"""

    for redshift in np.arange(redshift_range[0], redshift_range[1], 0.1):
        for log_temp in np.linspace(log_temp_range[0], log_temp_range[1], 10):
            for log_den in np.linspace(log_den_range[0], log_den_range[1], 10):
                with open('cloudy.in', 'w') as f:
                    f.write(template.format(redshift=redshift))
                os.system(cloudy_path + ' < cloudy.in > cloudy.out')
                # parse cloudy.out and store results in a table
                # (or just store cloudy.out and parse it later)


class IonFractionTable(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def calculate(self, simulation, ion='ovi'):
        pass



class ArchivedIonFractionTable(IonFractionTable):
    """Calculates ion fractions from an archived pynbody v1 table"""
    def __init__(self, filename=None):
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "ionfracs.npz")
        if os.path.exists(filename):
            # import data
            logger.info("Loading %s" % filename)
            self._table = np.load(filename)
        else:
            raise OSError("ionfracs.npz (Ion Fraction table) not found")

    def calculate(self, simulation, ion='ovi'):
        x_vals = self._table['redshiftvals'].view(np.ndarray)
        y_vals = self._table['tempvals'].view(np.ndarray)
        z_vals = self._table['denvals'].view(np.ndarray)
        vals = self._table[ion + 'if'].view(np.ndarray)
        x = np.zeros(len(simulation.gas))
        x[:] = simulation.properties['z']
        y = np.log10(simulation.gas['temp']).view(np.ndarray)
        z = np.log10(simulation.gas['rho'].in_units('m_p cm^-3')).view(np.ndarray)
        n = len(simulation.gas)
        n_x_vals = len(x_vals)
        n_y_vals = len(y_vals)
        n_z_vals = len(z_vals)

        # get values off grid to minmax
        x[np.where(x < np.min(x_vals))] = np.min(x_vals)
        x[np.where(x > np.max(x_vals))] = np.max(x_vals)
        y[np.where(y < np.min(y_vals))] = np.min(y_vals)
        y[np.where(y > np.max(y_vals))] = np.max(y_vals)
        z[np.where(z < np.min(z_vals))] = np.min(z_vals)
        z[np.where(z > np.max(z_vals))] = np.max(z_vals)

        # interpolate
        logger.info("Interpolation %s values" % ion)
        result_array = interpolate3d(x, y, z, x_vals, y_vals, z_vals, vals)

        return 10 ** result_array



def calculate(sim, ion='ovi'):
    """

    calculate -- documentation placeholder

    """

    table = ArchivedIonFractionTable()
    return table.calculate(sim, ion)
