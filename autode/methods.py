from autode.config import Config
from autode.exceptions import MethodUnavailable
from autode.log import logger
from abc import ABC
from abc import abstractmethod
from shutil import which
from autode.utils import requires_output
from copy import deepcopy
import os
import numpy as np
from autode.utils import run_external
from autode.config  import OptKeywords, GradientKeywords
from autode.atoms import Atom
from autode.units import Constants
from autode.exceptions import AtomsNotFound
from autode.utils import work_in_tmp_dir
from autode.calculation import CalculationOutput
from autode.calculation import Constraints


class ElectronicStructureMethod(ABC):

    def set_availability(self):
        logger.info(f'Setting the availability of {self.__name__}')

        if self.path is not None:
            if os.path.exists(self.path):
                self.available = True
                logger.info(f'{self.__name__} is available')

        if not self.available:
            logger.info(f'{self.__name__} is not available')
            self.available = False

    @abstractmethod
    def generate_input(self, calculation, molecule):
        """
        Function implemented in individual child classes

        Arguments:
            calculation (autode.calculation.Calculation):
            molecule (any):
        """
        pass

    # def generate_explicitly_solvated_input(self, calculation_input):
    #     """
    #     Function implemented in individual child classes

    #     Arguments:
    #         calculation_input (autode.calculation.CalculationInput):
    #     """
    #     raise NotImplementedError

    def clean_up(self, calc):
        """
        Remove any input files

        Arguments:
            calc (autode.calculation.Calculation):
        """
        for filename in calc.input.get_input_filenames():
            if os.path.exists(filename):
                os.remove(filename)

        return None

    @abstractmethod
    def get_output_filename(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_input_filename(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def execute(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def calculation_terminated_normally(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_energy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_free_energy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_enthalpy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def optimisation_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def optimisation_nearly_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_imaginary_freqs(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_normal_mode_displacements(self, calc, mode_number):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
            mode_number (int): Number of the normal mode to get the
            displacements along 6 == first imaginary mode
        """
        pass

    @abstractmethod
    @requires_output()
    def get_final_atoms(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_atomic_charges(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_gradients(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    def __init__(self, name, path, keywords_set, implicit_solvation_type):
        """
        Arguments:
            name (str): wrapper name. ALSO the name of the executable
            path (str): absolute path to the executable
            keywords_set (autode.wrappers.keywords.KeywordsSet):
            implicit_solvation_type (str):

        """
        self.name = name
        self.__name__ = self.__class__.__name__

        # If the path is not set in config.py or input script search in $PATH
        self.path = path if path is not None else which(name)

        # Availability is set when hlevel and llevel methods are set
        self.available = False

        self.keywords = deepcopy(keywords_set)

        assert type(implicit_solvation_type) is str
        self.implicit_solvation_type = implicit_solvation_type




def print_distance_constraints(inp_file, molecule, force_constant=20):
    """Add distance constraints to the input file"""

    if molecule.constraints.distance is None:
        return

    for (i, j), dist in molecule.constraints.distance.items():
        # XTB counts from 1 so increment atom ids by 1
        print(f'$constrain\n'
              f'force constant={force_constant}\n'
              f'distance:{i+1}, {j+1}, {dist:.4f}\n$',
              file=inp_file)
    return


def print_cartesian_constraints(inp_file, molecule, force_constant=20):
    """Add cartesian constraints to an xtb input file"""

    if molecule.constraints.cartesian is None:
        return

    constrained_atom_idxs = [i + 1 for i in molecule.constraints.cartesian]
    list_of_ranges, used_atoms = [], []

    for i in constrained_atom_idxs:
        atom_range = []
        if i not in used_atoms:
            while i in constrained_atom_idxs:
                used_atoms.append(i)
                atom_range.append(i)
                i += 1
            if len(atom_range) in (1, 2):
                list_of_ranges += str(atom_range)
            else:
                list_of_ranges.append(f'{atom_range[0]}-{atom_range[-1]}')

    print(f'$constrain\n'
          f'force constant={force_constant}\n'
          f'atoms: {",".join(list_of_ranges)}\n'
          f'$', file=inp_file)
    return


def print_point_charge_file(calc):
    """Generate a point charge file"""

    if calc.input.point_charges is None:
        return

    with open(f'{calc.name}_xtb.pc', 'w') as pc_file:
        print(len(calc.input.point_charges), file=pc_file)

        for point_charge in calc.input.point_charges:
            x, y, z = point_charge.coord
            charge = point_charge.charge
            print(f'{charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f}', file=pc_file)

    calc.input.additional_filenames.append(f'{calc.name}_xtb.pc')
    return


def print_xcontrol_file(calc, molecule):
    """Print an XTB input file with constraints and point charges"""

    xcontrol_filename = f'xcontrol_{calc.name}'
    with open(xcontrol_filename, 'w') as xcontrol_file:

        print_distance_constraints(xcontrol_file, molecule)
        print_cartesian_constraints(xcontrol_file, molecule)

        if calc.input.point_charges is not None:
            print_point_charge_file(calc)
            print(f'$embedding\n'
                  f'input={calc.name}_xtb.pc\n'
                  f'input=orca\n'
                  f'$end', file=xcontrol_file)

    calc.input.additional_filenames.append(xcontrol_filename)
    return


class XTB(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):

        calc.molecule.print_xyz_file(filename=calc.input.filename)

        if molecule.constraints.any() or calc.input.point_charges:
            print_xcontrol_file(calc, molecule)

        return None

    def get_input_filename(self, calc):
        return f'{calc.name}.xyz'

    def get_output_filename(self, calc):
        return f'{calc.name}.out'

    def execute(self, calc):
        """Execute an XTB calculation using the runtime flags"""
        # XTB calculation keywords must be a class

        flags = ['--chrg', str(calc.molecule.charge)]

        if isinstance(calc.input.keywords, OptKeywords):
            flags.append('--opt')

        if isinstance(calc.input.keywords, GradientKeywords):
            flags.append('--grad')

        if calc.input.solvent is not None:
            flags += ['--gbsa', calc.input.solvent]

        if len(calc.input.additional_filenames) > 0:
            # XTB allows for an additional xcontrol file, which should be the
            # last file in the list
            flags += ['--input', calc.input.additional_filenames[-1]]

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.xyz', '.out', '.pc', '.grad', 'gradient'))
        def execute_xtb():
            logger.info(f'Setting the number of OMP threads to {calc.n_cores}')
            os.environ['OMP_NUM_THREADS'] = str(calc.n_cores)

            run_external(params=[calc.method.path, calc.input.filename]+flags,
                         output_filename=calc.output.filename)

        execute_xtb()
        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if 'ERROR' in line:
                return False
            if n_line > 20:
                # With xtb we will search for there being no '#ERROR!' in the
                # last few lines
                return True

        return False

    def get_energy(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'total E' in line:
                return float(line.split()[-1])
            if 'TOTAL ENERGY' in line:
                return float(line.split()[-3])

    def get_enthalpy(self, calc):
        raise NotImplementedError

    def get_free_energy(self, calc):
        raise NotImplementedError

    def optimisation_converged(self, calc):

        for line in reversed(calc.output.file_lines):
            if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        raise NotImplementedError

    def get_imaginary_freqs(self, calc):
        raise NotImplementedError

    def get_normal_mode_displacements(self, calc, mode_number):
        raise NotImplementedError

    def _get_final_atoms_6_2_above(self, calc):
        """
        e.g.

        ================
         final structure:
        ================
        5
         xtb: 6.2.3 (830e466)
        Cl        1.62694523673790    0.09780349799138   -0.02455489507427
        C        -0.15839164427314   -0.00942638308615    0.00237760557913
        H        -0.46867957388620   -0.59222865914178   -0.85786049981721
        H        -0.44751262498645   -0.49575975568264    0.92748366742968
        H        -0.55236139359212    0.99971129991918   -0.04744587811734
        """
        atoms = []

        for i, line in enumerate(calc.output.file_lines):
            if 'final structure' in line:
                n_atoms = int(calc.output.file_lines[i+2].split()[0])

                for xyz_line in calc.output.file_lines[i+4:i+4+n_atoms]:
                    atom_label, x, y, z = xyz_line.split()
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))

                break

        return atoms

    def _get_final_atoms_old(self, calc):
        """
        e.g.

        ================
         final structure:
        ================
        $coord
            2.52072290250473   -0.04782551206377   -0.50388676977877      C
                    .                 .                    .              .
        """
        atoms = []
        geom_section = False

        for line in calc.output.file_lines:

            if '$coord' in line:
                geom_section = True

            if '$end' in line and geom_section:
                geom_section = False

            if len(line.split()) == 4 and geom_section:
                x, y, z, atom_label = line.split()

                atom = Atom(atom_label,
                            x=float(x) * Constants.a02ang,
                            y=float(y) * Constants.a02ang,
                            z=float(z) * Constants.a02ang)

                atoms.append(atom)

        return atoms

    def get_final_atoms(self, calc):
        atoms = []

        for i, line in enumerate(calc.output.file_lines):

            # XTB 6.2.x have a slightly different way of printing the atoms
            if 'xtb version' in line and len(line.split()) >= 4:
                if line.split()[3] == '6.2.3' or '6.3' in line.split()[3]:
                    atoms = self._get_final_atoms_6_2_above(calc)
                    break

                elif line.split()[3] == '6.2.2' or '6.1' in line.split()[3]:
                    atoms = self._get_final_atoms_old(calc)
                    break

            # Version is not recognised if we're 50 lines into the output file
            # - try and use the old version
            if i > 50:
                atoms = self._get_final_atoms_old(calc)
                break

        if len(atoms) == 0:
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self, calc):
        charges_sect = False
        charges = []
        for line in calc.output.file_lines:
            if 'Mol.' in line:
                charges_sect = False
            if charges_sect and len(line.split()) == 7:
                charges.append(float(line.split()[4]))
            if 'covCN' in line:
                charges_sect = True
        return charges

    def get_gradients(self, calc):
        gradients = []

        if os.path.exists(f'{calc.name}_xtb.grad'):
            grad_file_name = f'{calc.name}_xtb.grad'
            with open(grad_file_name, 'r') as grad_file:
                for line in grad_file:
                    x, y, z = line.split()
                    gradients.append(np.array([float(x), float(y), float(z)]))

        elif os.path.exists('gradient'):
            with open('gradient', 'r') as grad_file:
                for i, line in enumerate(grad_file):
                    if i > 1 and len(line.split()) == 3:
                        x, y, z = line.split()
                        vec = [float(x.replace('D', 'E')),
                               float(y.replace('D', 'E')),
                               float(z.replace('D', 'E'))]

                        gradients.append(np.array(vec))

            with open(f'{calc.name}_xtb.grad', 'w') as new_grad_file:
                [print('{:^12.8f} {:^12.8f} {:^12.8f}'.format(*line),
                       file=new_grad_file) for line in gradients]
            os.remove('gradient')

        # Convert from Ha a0^-1 to Ha A-1
        gradients = [grad / Constants.a02ang for grad in gradients]
        return np.array(gradients)

    def __init__(self):
        super().__init__(name='xtb', path=Config.XTB.path,
                         keywords_set=Config.XTB.keywords,
                         implicit_solvation_type=Config.XTB.implicit_solvation_type)

xtb = XTB()



def modify_keywords_for_point_charges(keywords):
    """For a list of Gaussian keywords modify to include z-matrix if not
    already included. Required if point charges are included in the calc"""
    logger.warning('Modifying keywords as point charges are present')

    keywords.append('Charge')

    for keyword in keywords:
        if 'opt' not in keyword.lower():
            continue

        opt_options = []
        if '=(' in keyword:
            # get the individual options
            unformated_options = keyword[5:-1].split(',')
            opt_options = [option.lower().strip() for option in unformated_options]

        elif '=' in keyword:
            opt_options = [keyword[4:]]

        if not any(option.lower() == 'z-matrix' for option in opt_options):
            opt_options.append('Z-Matrix')

        new_keyword = f'Opt=({", ".join(opt_options)})'
        keywords.remove(keyword)
        keywords.append(new_keyword)

    return None


def get_keywords(calc_input, molecule):
    """Modify the input keywords to try and fix some Gaussian's quirks"""

    keywords = calc_input.keywords.copy()

    # Mod redundant keywords is required if there are any constraints or
    # modified internal coordinates
    if molecule.constraints.any():
        keywords.append('Geom=ModRedun')

    if calc_input.added_internals is not None:
        keywords.append('Geom=ModRedun')

    # Remove the optimisation keyword if there is only a single atom
    opt = False
    for keyword in keywords:

        if 'opt' not in keyword.lower():
            opt = True
            continue

        if molecule.n_atoms == 1:
            logger.warning('Cannot do an optimisation for a single atom')
            keywords.remove(keyword)

    # Further modification is required if there are surrounding point charges
    if calc_input.point_charges is not None:
        modify_keywords_for_point_charges(keywords)

    # By default perform all optimisations without symmetry
    if opt and not any(kw.lower() == 'nosymm' for kw in keywords):
        keywords.append('NoSymm')

    return keywords


def print_point_charges(inp_file, calc_input):
    """Add point charges to the input file"""

    if calc_input.point_charges is None:
        return

    print('', file=inp_file)
    for point_charge in calc_input.point_charges:
        x, y, z = point_charge.coord
        print(f'{x:^12.8f} {y:^12.8f} {z:^12.8f} {point_charge.charge:^12.8f}',
              file=inp_file)
    return


def print_added_internals(inp_file, calc_input):
    """Add any internal coordinates to the input file"""

    if calc_input.added_internals is None:
        return

    for (i, j) in calc_input.added_internals:
        # Gaussian indexes atoms from 1
        print('B', i + 1, j + 1, file=inp_file)

    return


def print_constraints(inp_file, molecule):
    """Add any distance or cartesian constraints to the input file"""

    if molecule.constraints.distance is not None:

        for (i, j), dist in molecule.constraints.distance.items():
            # Gaussian indexes atoms from 1
            print('B', i + 1, j + 1, dist, 'B', file=inp_file)
            print('B', i + 1, j + 1, 'F', file=inp_file)

    if molecule.constraints.cartesian is not None:

        for i in molecule.constraints.cartesian:
            # Gaussian indexes atoms from 1
            print('X', i+1, 'F', file=inp_file)
    return


def rerun_angle_failure(calc):
    """
    Gaussian will sometimes encounter a 180 degree angle and crash. This
    function performs a few geometry optimisation cycles in cartesian
    coordinates then switches back to internals

    Arguments:
        calc (autode.calculation.Calculation):

    Returns:
        (autode.calculation.Calculation):
    """
    cart_calc = deepcopy(calc)

    # Iterate through a copied set of keywords
    for keyword in cart_calc.input.keywords.copy():
        if keyword.lower().startswith('geom'):
            cart_calc.input.keywords.remove(keyword)

        elif keyword.lower().startswith('opt'):
            options = []
            if '=(' in keyword:
                # get the individual options
                options = [option.lower().strip()
                           for option in keyword[5:-1].split(',')]

                for option in options:
                    if (option.startswith('maxcycles')
                            or option.startswith('maxstep')):
                        options.remove(option)

            elif '=' in keyword:
                options = [keyword[4:]]
            options += ['maxcycles=3', 'maxstep=1', 'cartesian']

            new_keyword = f'Opt=({", ".join(options)})'
            cart_calc.input.keywords.remove(keyword)
            cart_calc.input.keywords.append(new_keyword)

    # Generate the new calculation and run
    cart_calc.name += '_cartesian'
    cart_calc.molecule.atoms = calc.get_final_atoms()
    cart_calc.molecule.constraints = Constraints(distance=None, cartesian=None)
    cart_calc.input.added_internals = None
    cart_calc.output = CalculationOutput()
    cart_calc.run()

    if not cart_calc.terminated_normally():
        logger.warning('Cartesian calculation did not converge')
        return None

    logger.info('Returning to internal coordinates')

    # Reset the required parameters for the new calculation
    fixed_calc = deepcopy(calc)
    fixed_calc.name += '_internal'
    fixed_calc.molecule.atoms = cart_calc.get_final_atoms()
    fixed_calc.output = CalculationOutput()
    fixed_calc.run()

    return fixed_calc


class G09(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):
        """Print a Gaussian input file"""

        with open(calc.input.filename, 'w') as inp_file:
            print(f'%mem={Config.max_core}MB', file=inp_file)
            if calc.n_cores > 1:
                print(f'%nprocshared={calc.n_cores}', file=inp_file)

            keywords = get_keywords(calc.input, molecule)
            print('#', *keywords, file=inp_file, end=' ')

            if calc.input.solvent is not None:
                print(f'scrf=(smd,solvent={calc.input.solvent})', file=inp_file)
            else:
                print('', file=inp_file)

            print(f'\n {calc.name}\n', file=inp_file)

            print(molecule.charge, molecule.mult, file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
                      file=inp_file)

            print_point_charges(inp_file, calc.input)
            print('', file=inp_file)
            print_added_internals(inp_file, calc.input)
            print_constraints(inp_file, molecule)

        return None

    def get_input_filename(self, calc):
        return f'{calc.name}.com'

    def get_output_filename(self, calc):
        return f'{calc.name}.log'

    def execute(self, calc):

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.log', '.com'))
        def execute_g09():
            run_external(params=[calc.method.path, calc.input.filename],
                         output_filename=calc.output.filename)

        execute_g09()
        return None

    def calculation_terminated_normally(self, calc, rerun_if_failed=True):

        termination_strings = ['Normal termination of Gaussian',
                               'Number of steps exceeded']

        for line in reversed(calc.output.file_lines):

            if any(substring in line for substring in termination_strings):
                logger.info('Gaussian09 terminated normally')
                return True

            if 'Bend failed for angle' in line:
                logger.warning('Gaussian encountered a 180° angle and crashed')
                break

        if not rerun_if_failed:
            return False

        # Set a limit on the amount of times we do this
        if calc.name.endswith('internal_internal_internal_internal'):
            return False

        try:
            # To fix the calculation requires the atoms to be in the output
            fixed_calc = rerun_angle_failure(calc)

        except AtomsNotFound:
            return False

        if fixed_calc.terminated_normally():
            logger.info('The 180° angle issue has been fixed')
            calc.output = fixed_calc.output
            calc.name = fixed_calc.name
            calc.output.set_lines()
            return True

        return False

    def get_enthalpy(self, calc):
        """Get the enthalpy (H) from an g09 calculation output"""

        for line in reversed(calc.output.file_lines):
            if 'Sum of electronic and thermal Enthalpies' in line:
                return float(line.split()[-1])

        logger.error('Could not get the enthalpy from the calculation. '
                     'A frequency must be requested')
        return None

    def get_free_energy(self, calc):
        """Get the Gibbs free energy (G) from an g09 calculation output"""

        for line in reversed(calc.output.file_lines):
            if 'Sum of electronic and thermal Free Energies' in line:
                return float(line.split()[-1])

        logger.error('Could not get the enthalpy from the calculation. '
                     'A frequency must be requested')
        return None

    def get_energy(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'SCF Done' in line:
                return float(line.split()[4])
            if 'E(CORR)' in line:
                return float(line.split()[3])
            if 'E(CI)' in line:
                return float(line.split()[3])
            if 'E(CIS)' in line:
                return float(line.split()[4])
            if 'E(CIS(D))' in line:
                return float(line.split()[5])

        return None

    def optimisation_converged(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'Optimization completed' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        geom_conv_block = False

        for line in reversed(calc.output.file_lines):
            if geom_conv_block and 'Item' in line:
                geom_conv_block = False
            if 'Predicted change in Energy' in line:
                geom_conv_block = True
            if geom_conv_block and len(line.split()) == 4:
                if line.split()[-1] == 'YES':
                    return True
        return False

    def get_imaginary_freqs(self, calc):
        imag_freqs = []
        normal_mode_section = False

        for line in calc.output.file_lines:
            if 'normal coordinates' in line:
                normal_mode_section = True
                imag_freqs = []

            if 'Thermochemistry' in line:
                normal_mode_section = False

            if normal_mode_section and 'Frequencies' in line:
                freqs = [float(line.split()[i])
                         for i in range(2, len(line.split()))]
                for freq in freqs:
                    if freq < 0:
                        imag_freqs.append(freq)

        logger.info(f'Found imaginary freqs {imag_freqs}')
        return imag_freqs

    def get_normal_mode_displacements(self, calc, mode_number):
        # mode numbers start at 1, not 6
        mode_number -= 5
        start_col = 0
        normal_mode_section, displacements = False, []
        correct_mode_section = False

        for j, line in enumerate(calc.output.file_lines):
            if 'normal coordinates' in line:
                normal_mode_section = True
                displacements = []

            if 'Thermochemistry' in line:
                normal_mode_section = False

            if correct_mode_section and len(line.split()) > 3 and line.split()[0].isdigit():
                displacements.append([float(line.split()[k]) for k in range(start_col, start_col + 3)])

            if normal_mode_section and len(line.split()) == 3 and line.split()[0].isdigit():
                mode_numbers = [int(n) for n in line.split()]
                if mode_number in mode_numbers:
                    correct_mode_section = True
                    start_col = 3 * [i for i in range(len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 2
                else:
                    correct_mode_section = False

        return np.array(displacements)

    def get_final_atoms(self, calc):

        atoms = None

        for i, line in enumerate(calc.output.file_lines):

            if 'Standard orientation' in line or 'Input orientation' in line:

                atoms = []
                xyz_lines = calc.output.file_lines[i+5:i+5+calc.molecule.n_atoms]

                for xyz_line in xyz_lines:
                    atom_index, _, _, x, y, z = xyz_line.split()
                    atom_index = int(atom_index) - 1
                    atoms.append(Atom(calc.molecule.atoms[atom_index].label, x=x, y=y, z=z))

                if len(atoms) != calc.molecule.n_atoms:
                    raise AtomsNotFound

        if atoms is None:
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self, calc):

        charges_section = False
        charges = []
        for line in reversed(calc.output.file_lines):
            if 'sum of mulliken charges' in line.lower():
                charges_section = True

            if len(charges) == calc.molecule.n_atoms:
                return list(reversed(charges))

            if charges_section and len(line.split()) == 3:
                charges.append(float(line.split()[2]))

        logger.error('Something went wrong finding the atomic charges')
        return None

    def get_gradients(self, calc):
        gradients_section = False
        gradients = []
        dashed_line = 0

        for line in calc.output.file_lines:

            if 'Axes restored to original set' in line:
                gradients_section = True
                gradients = []
                dashed_line = 0

            if gradients_section and '--------' in line:
                dashed_line += 1
                if dashed_line == 3:
                    gradients_section = False

            if gradients_section and len(line.split()) == 5:
                _, _, fx, fy, fz = line.split()
                try:
                    # Ha / a0
                    force = np.array([float(fx), float(fy), float(fz)])

                    grad = -force / Constants.a02ang
                    gradients.append(grad)
                except ValueError:
                    pass
        for line in gradients:
            for i in range(3):
                line[i] *= -1

        return np.array(gradients)

    def __init__(self, name='g09', path=None, keywords_set=None,
                 implicit_solvation_type=None):
        """Gaussian 09"""

        if keywords_set is None:
            keywords_set = Config.G09.keywords

        if implicit_solvation_type is None:
            implicit_solvation_type = Config.G09.implicit_solvation_type

        super().__init__(name=name,
                         path=Config.G09.path if path is None else path,
                         keywords_set=keywords_set,
                         implicit_solvation_type=implicit_solvation_type)


g09 = G09()







"""
Functions to get the high and low level electronic structure methods to use for example high-level methods would be
orca and Gaussian09 which can perform DFT/WF theory calculations, low level methods are for example xtb and mopac which
are non ab-initio methods and are therefore considerably faster
"""

high_level_method_names = ['g09']
low_level_method_names = ['xtb']


def get_hmethod():
    """Get the high-level electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method
    """
    # orca = ORCA()
    # g09 = G09()
    # nwchem = NWChem()
    # g16 = G16()
    all_methods = [XTB(), G09()]
    
    return get_defined_method(name=Config.hcode.lower(),
                                  possibilities=all_methods)


def get_lmethod():
    """Get the low-level electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod):
    """
    all_methods = [XTB(), G09()]

    return get_defined_method(name=Config.lcode.lower(),
                                  possibilities=all_methods)



def get_defined_method(name, possibilities):
    """
    Get an electronic structure method defined by it's name

    Arguments:
        name (str):
        possibilities (list(autode.wrappers.base.ElectronicStructureMethod)):

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method
    """

    for method in possibilities:
        if method.name == name:

            method.set_availability()
            # if method.available:
            if True:
                return method

            else:
                logger.critical('Electronic structure method is not available')
                raise MethodUnavailable

    logger.critical('Requested electronic structure code doesn\'t exist')
    raise MethodUnavailable
