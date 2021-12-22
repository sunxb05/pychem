from abc import ABC
from abc import abstractmethod
from copy import deepcopy
import itertools
import numpy as np
from autode.bond_lengths import get_avg_bond_length
from autode.calculation import Calculation
from autode.exceptions import AtomsNotFound, NoClosestSpecies, FitFailed
from autode.log import logger
from autode.ts_guess import get_ts_guess
from autode.config import Config
from autode.mol_graphs import is_isomorphic, make_graph
from numpy.polynomial import polynomial
from autode.utils import work_in, NoDaemonPool
from autode.units import KcalMol
from autode.calculation import Calculation
from autode.methods import high_level_method_names
import networkx as nx
from scipy.optimize import minimize, Bounds


def get_closest_species(point, pes):
    """
    Given a point on an n-dimensional potential energy surface defined by
    indices where the length is the dimension of the surface

    Arguments:
        pes (autode.pes.PES): Potential energy surface
        point (tuple): Index of the current point

    Returns:
        (autode.complex.ReactantComplex):
    """

    if all(index == 0 for index in point):
        logger.info('PES is at the first point')
        return deepcopy(pes.species[point])

    # The indcies of the nearest and second nearest points to e.g. n,m in a 2
    # dimensional PES
    neareast_neighbours = [-1, 0, 1]
    next_nearest_neighbours = [-2, -1, 0, 1, 2]

    # First attempt to find a species that has been calculated in the nearest
    # neighbours
    for index_array in [neareast_neighbours, next_nearest_neighbours]:

        # Each index array has elements from the most negative to most
        # positive. e.g. (-1, -1), (-1, 0) ... (1, 1)
        for d_indexes in itertools.product(index_array, repeat=len(point)):

            # For e.g. a 2D PES the new index is (n+i, m+j) where
            # i, j = d_indexes
            new_point = tuple(np.array(point) + np.array(d_indexes))

            try:
                if pes.species[new_point] is not None:
                    logger.info(f'Closest point in the PES has indices '
                                f'{new_point}')
                    return deepcopy(pes.species[new_point])

            except IndexError:
                logger.warning('Closest point on the PES was outside the PES')

    logger.error(f'Could not get a close point to {point}')
    raise NoClosestSpecies


def get_point_species(point, species, distance_constraints, name, method,
                      keywords, n_cores, energy_threshold=1):
    """
    On a 2d PES calculate the energy and the structure using a constrained
    optimisation

    Arguments:
        point (tuple(int)): Index of this point e.g. (0, 0) for the first point
                            on a 2D surface

        species (autode.species.Species):

        distance_constraints (dict): Keyed with atom indexes and the constraint
                             value as the value

        name (str):

        method (autode.wrappers.base.ElectronicStructureMethod):

        keywords (autode.wrappers.keywords.Keywords):

        n_cores (int): Number of cores to used for this calculation

    Keyword Arguments:
        energy_threshold (float): Above this energy (Hartrees) the calculation
                                  will be disregarded
    """
    logger.info(f'Calculating point {point} on PES surface')

    species.name = f'{name}_scan_{"-".join([str(p) for p in point])}'
    original_species = deepcopy(species)

    # Set up and run the calculation
    const_opt = Calculation(name=species.name, molecule=species, method=method,
                            n_cores=n_cores,
                            keywords=keywords,
                            distance_constraints=distance_constraints)
    try:
        species.optimise(method=method, calc=const_opt)

    except AtomsNotFound:
        logger.error(f'Optimisation failed for {point}')
        return original_species

    # If the energy difference is > 1 Hartree then likely something has gone
    # wrong with the EST method we need to be not on the first point to compute
    # an energy difference..
    if not all(p == 0 for p in point):
        if species.energy is None or np.abs(original_species.energy - species.energy) > energy_threshold:
            logger.error(f'PES point had a relative energy '
                         f'> {energy_threshold} Ha. Using the closest')
            return original_species

    return species


class PES(ABC):

    @abstractmethod
    def get_species_saddle_point(self, *args):
        """Return the autode.species.Species at the saddle point"""
        pass

    @abstractmethod
    def products_made(self):
        """Have the products been made somewhere on the surface?"""
        pass

    @abstractmethod
    def calculate(self, name, method, keywords):
        """Calculate all energies and optimised geometries on the surface"""
        pass

    species = None
    rs = None
    rs_idxs = None


class ScannedBond:

    def __str__(self):
        i, j = self.atom_indexes
        return f'{i}-{j}'

    def __getitem__(self, item):
        return self.atom_indexes[item]

    def __init__(self, atom_indexes):
        """
        Bond with a current and final distance which will be scanned over

        Arguments:
            atom_indexes (tuple(int)): Atom indexes that make this
            'bond' e.g. (0, 1)
        """
        assert len(atom_indexes) == 2

        self.atom_indexes = atom_indexes

        self.curr_dist = None
        self.final_dist = None


class FormingBond(ScannedBond):

    def __init__(self, atom_indexes, species):
        """"
        Forming bond with current and final distances

        Arguments:
            atom_indexes (tuple(int)):
            species (autode.species.Species):
        """
        super().__init__(atom_indexes)

        i, j = self.atom_indexes
        self.curr_dist = species.get_distance(atom_i=i, atom_j=j)
        self.final_dist = get_avg_bond_length(species.atoms[i].label,
                                              species.atoms[j].label)


class BreakingBond(ScannedBond):

    def __init__(self, atom_indexes, species, reaction=None):
        """
        Form a breaking bond with current and final distances

        Arguments:
            atom_indexes (tuple(int)):
            species (autode.species.Species):
            reaction (autode.reaction.Reaction):
        """
        super().__init__(atom_indexes)

        self.curr_dist = species.get_distance(*self.atom_indexes)

        # Length a breaking bond should increase by (Å)
        bbond_add_dist = 1.5

        # If a reaction is specified and any component is charged then use a
        # larger ∆r shift as the interaction range is likely further
        if reaction is not None:
            if (any(mol.charge != 0 for mol in reaction.prods)
                    or any(mol.charge != 0 for mol in reaction.reacs)):

                bbond_add_dist = 2.5

        self.final_dist = self.curr_dist + bbond_add_dist



class PES1d(PES):

    def get_species_saddle_point(self):
        """Get the possible first order saddle points, which are just the
        peaks in the PES"""
        energies = [self.species[i].energy for i in range(self.n_points)]

        if any(energy is None for energy in energies):
            raise FitFailed

        # Peaks have lower energies both sides of them
        peaks = [i for i in range(1, self.n_points - 1) if energies[i-1] < energies[i] and energies[i+1] < energies[i]]

        # Yield the peak with the highest energy first
        for peak in sorted(peaks, key=lambda p: -self.species[p].energy):
            yield self.species[peak]

        return None

    def products_made(self):
        logger.info('Checking that somewhere on the surface product(s) are made')

        for i in range(self.n_points):
            make_graph(self.species[i])

            if is_isomorphic(graph1=self.species[i].graph, graph2=self.product_graph):
                logger.info(f'Products made at point {i} in the 1D surface')
                return True

        return False

    @work_in('pes1d')
    def calculate(self, name, method, keywords):
        """Calculate all the points on the surface in serial using the maximum
         number of cores available"""

        for i in range(self.n_points):
            closest_species = get_closest_species((i,), self)

            # Set up the dictionary of distance constraints keyed with bond
            # indexes and values the current r1, r2.. value
            distance_constraints = {self.rs_idxs[0]: self.rs[i][0]}

            self.species[i] = get_point_species((i,), closest_species,
                                                distance_constraints,
                                                name,
                                                method,
                                                keywords,
                                                Config.n_cores)
        return None

    def __init__(self, reactant, product, rs, r_idxs):
        """
        A one dimensional potential energy surface

        Arguments:
            reactant (autode.complex.ReactantComplex): Species at rs[0]
            product (autode.complex.ProductComplex):
            rs (np.ndarray): Bond length array
            r_idxs (tuple): Atom indexes that the PES will be calculated over
        """
        self.n_points = len(rs)
        self.rs = np.array([(r, ) for r in rs])

        # Vector to store the species
        self.species = np.empty(shape=(self.n_points,), dtype=object)
        self.species[0] = deepcopy(reactant)

        # Tuple of the atom indices scanned in coordinate r
        self.rs_idxs = [r_idxs]

        # Molecular graph of the product. Used to check that the products have
        # been made & find the MEP
        self.product_graph = product.graph


def get_ts_guess_1d(reactant, product, bond, name, method, keywords, dr=0.1):
    """Scan the distance between two atoms and return a guess for the TS

    Arguments:
        reactant (autode.complex.ReactantComplex):
        product (autode.complex.ProductComplex):
        bond (autode.pes.ScannedBond):
        name (str): name of reaction
        method (autode.): electronic structure wrapper to use for the calcs
        keywords (autode.keywords.Keywords): keywords to use in the calcs

    Keyword Arguments:
        dr (float): Δr on the surface *absolute value* in angstroms

    Returns:
        (autode.transition_states.ts_guess.TSguess)
    """
    logger.info(f'Getting TS guess from 1D relaxed potential energy scan using '
                f'{bond.atom_indexes} as the active bond')

    # Create a potential energy surface in the active bonds and calculate
    pes = PES1d(reactant=reactant, product=product,
                rs=np.arange(bond.curr_dist, bond.final_dist, step=dr if bond.final_dist > bond.curr_dist else -dr),
                r_idxs=bond.atom_indexes)

    pes.calculate(name=name, method=method, keywords=keywords)

    if not pes.products_made():
        logger.error('Products were not made on the whole PES')
        return None


    try:
        # May want to iterate through all saddle points not just the highest(?)
        for species in pes.get_species_saddle_point():
            return get_ts_guess(species=species, reactant=reactant, product=product, name=name)

    except FitFailed:
        logger.error('Could not find saddle point on 1D surface')

    logger.error('No possible TSs found on the 1D surface')
    return None


def poly2d_saddlepoints(coeff_mat, xs, ys):
    """Finds the saddle points of a 2d surface defined by a matrix of coefficients

    Arguments:
        coeff_mat (np.array): Matrix of coefficients of the n order polynomial
        xs (float) (np.ndarray): 1D
        ys (float) (np.ndarray): 1D

    Returns:
        list: list of saddle points
    """
    logger.info('Finding saddle points')
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    stationary_points = []

    # Optimise the derivatives over a uniform grid in x, y. 10x10 should find all the unique stationary points
    for x in np.linspace(min_x, max_x, num=10):
        for y in np.linspace(min_y, max_y, num=10):

            # Minimise (df/dx)^2 + (dy/dx)^2 with bounds ensuring the saddle points are within the surface
            opt = minimize(sum_squared_xy_derivative,
                           x0=np.array([x, y]), args=(coeff_mat,),
                           method='TNC',
                           bounds=Bounds(lb=np.array([min_x, min_y]),
                                         ub=np.array([max_x, max_y])))
            opt_x, opt_y = opt.x

            # Check that we're still inside the bounds and the optimisation has converged reasonably
            if min_x < opt_x < max_x and min_y < opt_y < max_y and opt.fun < 1E-1:
                stationary_points.append(opt.x)

    # Remove all repeated stationary points
    stationary_points = get_unique_stationary_points(stationary_points)

    # Return all stationary points that are first order saddle points (i.e. could be a TS)
    saddle_points = [point for point in stationary_points if is_saddle_point(point, coeff_mat)]
    logger.info(f'Found {len(saddle_points)} saddle points')

    saddle_points = get_sorted_saddlepoints(saddle_points=saddle_points, xs=xs, ys=ys)
    return saddle_points


def get_sorted_saddlepoints(saddle_points, xs, ys):
    """Get the list of saddle points ordered by their distance from the (x, y) mid-point"""

    mid_x, mid_y = np.average(xs), np.average(ys)

    return sorted(saddle_points, key=lambda point: np.abs(point[0] - mid_x) + np.abs(point[1] - mid_y))


def get_unique_stationary_points(stationary_points, dist_threshold=0.1):
    """Strip all points that are close to each other"""
    logger.info(f'Have {len(stationary_points)} stationary points')

    unique_stationary_points = stationary_points[:1]

    for stat_point in stationary_points[1:]:

        # Assume the point in unique and determine if it is close to any of the point already in the list
        unique = True

        for unique_stat_point in unique_stationary_points:
            distance = np.sqrt(np.sum(np.square(np.array(stat_point) - np.array(unique_stat_point))))
            if distance < dist_threshold:
                unique = False

        if unique:
            unique_stationary_points.append(stat_point)

    logger.info(f'Stripped {len(stationary_points) - len(unique_stationary_points)} stationary points')
    return unique_stationary_points


def sum_squared_xy_derivative(xy_point, coeff_mat):
    """For a coordinate, and function, finds df/dx and df/dy and returns the sum of the squares

    Arguments:
        xy_point (tuple): (x,y)
        coeff_mat (np.array): Matrix of coefficients of the n order polynomial

    Returns:
        (float): (df/dx + df/dy)^2 where at a stationary point ~ 0
    """
    order = coeff_mat.shape[0]
    x, y = xy_point
    dx, dy = 0, 0

    for i in range(order):  # x index
        for j in range(order):  # y index
            if i > 0:
                dx += coeff_mat[i, j] * i * x**(i-1) * y**j
            if j > 0:
                dy += coeff_mat[i, j] * x**i * j * y**(j-1)

    return dx**2 + dy**2


def is_saddle_point(xy_point, coeff_mat):
    """
    Calculates whether a point (x, y) is a saddle point by computing

    delta = ((d2f/dx2)*(d2f/dy2) - (d2f/dxdy)**2)

    Arguments:
        coeff_mat (np.array): Matrix of the coefficients of the n order polynomial (n x n)
        xy_point (tuple): the stationary point to be examined

    Returns:
         (bool):
    """
    dx2, dy2, dxdy = 0, 0, 0
    x, y = xy_point

    order = coeff_mat.shape[0]
    for i in range(order):  # x index
        for j in range(order):  # y index
            if i > 1:
                dx2 += coeff_mat[i, j] * i * (i - 1) * x**(i - 2) * y**j
            if j > 1:
                dy2 += coeff_mat[i, j] * x**i * j * (j - 1) * y**(j - 2)
            if i > 0 and j > 0:
                dxdy += coeff_mat[i, j] * i * x**(i - 1) * j * y**(j - 1)

    if dx2 * dy2 - dxdy**2 < 0:
        logger.info(f'Found saddle point at r1 = {x:.3f}, r2 = {y:.3f} Å')
        return True

    else:
        return False

def get_sum_energy_mep(saddle_point_r1r2, pes_2d):
    """
    Calculate the sum of the minimum energy path that traverses reactants (r)
    to products (p) via the saddle point (s)

            /          p
           /     s
      r2  /
         /r
         ------------
              r1

    Arguments:
        saddle_point_r1r2 (tuple(float)):

        pes_2d (autode.pes_2d.PES2d):
    """
    logger.info('Finding the total energy along the minimum energy pathway')

    reactant_point = (0, 0)
    product_point, product_energy = None, 9999

    # The saddle point indexes are those that are closest tp the saddle points
    # r1 and r2 distances
    saddle_point = (np.argmin(np.abs(pes_2d.r1s - saddle_point_r1r2[0])),
                    np.argmin(np.abs(pes_2d.r2s - saddle_point_r1r2[1])))

    # Generate a grid graph (i.e. all nodes are corrected
    energy_graph = nx.grid_2d_graph(pes_2d.n_points_r1, pes_2d.n_points_r2)

    min_energy = min([species.energy for species in pes_2d.species.flatten()])

    # For energy point on the 2D surface
    for i in range(pes_2d.n_points_r1):
        for j in range(pes_2d.n_points_r2):
            point_rel_energy = pes_2d.species[i, j].energy - min_energy

            # Populate the relative energy of each node in the graph
            energy_graph.nodes[i, j]['energy'] = point_rel_energy

            # Find the point where products are made
            if is_isomorphic(graph1=pes_2d.species[i, j].graph,
                             graph2=pes_2d.product_graph):

                # If products have not yet found, or they have and the energy
                # are lower but are still isomorphic
                if product_point is None or point_rel_energy < product_energy:
                    product_point = (i, j)
                    product_energy = point_rel_energy

    logger.info(f'Reactants at r1={pes_2d.r1s[0]:.4f} , '
                f'r2={pes_2d.r2s[0]:.4f} Å and '
                f'products r1={pes_2d.rs[product_point][0]:.4f}, '
                f'r2={pes_2d.rs[product_point][1]:.4f} Å')

    def energy_diff(curr_node, final_node, d):
        """Energy difference between the twp points on the graph. d is required
         to satisfy nx. Must only increase in energy to a saddle point so take
          the magnitude to prevent traversing s mistakenly"""
        return (np.abs(energy_graph.nodes[final_node]['energy']
                       - energy_graph.nodes[curr_node]['energy']))

    # Calculate the energy along the MEP up to the saddle point from reactants
    # and products
    path_energy = 0.0

    for point in (reactant_point, product_point):
        path_energy += nx.dijkstra_path_length(energy_graph,
                                               source=point,
                                               target=saddle_point,
                                               weight=energy_diff)

    logger.info(f'Path energy to {saddle_point} is {path_energy:.4f} Hd')
    return path_energy

class PES2d(PES):

    def get_species_saddle_point(self, name, method, keywords):
        """Get the species at the true saddle point on the surface"""
        saddle_points = poly2d_saddlepoints(coeff_mat=self.coeff_mat, xs=self.r1s, ys=self.r2s)

        logger.info('Sorting the saddle points by their minimum energy path to '
                    'reactants and products')
        saddle_points = sorted(saddle_points, key=lambda s: get_sum_energy_mep(s, self))

        for saddle_point in saddle_points:
            r1, r2 = saddle_point

            # Determine the indicies of the point closest to the analytic
            # saddle point to use as a guess
            close_point = (np.argmin(np.abs(self.r1s - r1)), np.argmin(np.abs(self.r2s - r2)))
            logger.info(f'Closest point is {close_point} with r1 = {self.rs[close_point][0]:.3f}, '
                        f'r2 = {self.rs[close_point][1]:.3f} Å')

            # Perform a constrained optimisation using the analytic saddle
            # point r1, r2 values
            species = deepcopy(self.species[close_point])
            const_opt = Calculation(name=f'{name}_const_opt', molecule=species,
                                    method=method,
                                    n_cores=Config.n_cores, keywords=keywords,
                                    distance_constraints={self.rs_idxs[0]: r1, self.rs_idxs[1]: r2})

            try:
                species.optimise(method=method, calc=const_opt)
            except AtomsNotFound:
                logger.error('Constrained optimisation at the saddle point '
                             'failed')

            return species

        return None

    def fit(self, polynomial_order):
        """Fit an analytic 2d surface"""

        energies = [species.energy for species in self.species.flatten()]
        if any(energy is None for energy in energies):
            raise FitFailed

        # Compute a flat list of relative energies to use to fit the polynomial
        min_energy = min(energies)
        rel_energies = [KcalMol.conversion * (species.energy - min_energy) for species in self.species.flatten()]

        # Compute a polynomial_order x polynomial_order matrix of coefficients
        self.coeff_mat = polyfit2d(x=[r[0] for r in self.rs.flatten()],
                                   y=[r[1] for r in self.rs.flatten()],
                                   z=rel_energies, order=polynomial_order)
        return None

    def products_made(self):
        """Check that somewhere on the surface the molecular graph is
        isomorphic to the product"""
        logger.info('Checking product(s) are made somewhere on the surface')

        for i in range(self.n_points_r1):
            for j in range(self.n_points_r2):
                make_graph(self.species[i, j])

                if is_isomorphic(graph1=self.species[i, j].graph, graph2=self.product_graph):
                    logger.info(f'Products made at ({i}, {j})')
                    return True

        return False

    @work_in('pes2d')
    def calculate(self, name, method, keywords):
        """Calculations on the surface with a method using the a decomposition similar to the following

        Calculation order            Indexes

           4  5  6  7          .        .        .

           3  4  5  6        (0, 2)     .        .

           2  3  4  5        (0, 1)  (1, 1)      .

           1  2  3  4        (0, 0)  (1, 0)    (2, 0)
                                ↖       ↖         ↖
                            sum = 0   sum = 1    sum = 2

        Arguments:
            name (str):
            method (autode.wrappers.ElectronicStructureMethod):
            keywords (list(str)):
        """
        logger.info(f'Running a 2D PES scan with {method.name}. {self.n_points_r1*self.n_points_r2} total points')

        for sum_indexes in range(self.n_points_r1 + self.n_points_r2 - 1):

            all_points = [(i, j) for i in range(self.n_points_r1) for j in range(self.n_points_r2)]

            # Strip those that are along the current diagonal – the sum of indices is constant
            diagonal_points = [point for point in all_points if sum(point) == sum_indexes]

            # Strip the indices that are not in the array. This applies if n_points_r1 != n_points_r2
            points = [(i, j) for (i, j) in diagonal_points if i < self.n_points_r1 and j < self.n_points_r2]

            # The cores for this diagonal are the floored number of total cores divided by the number of calculations
            cores_per_process = Config.n_cores // len(points) if Config.n_cores // len(points) > 1 else 1

            closest_species = [get_closest_species(p, self) for p in points]

            # Set up the dictionary of distance constraints keyed with bond indexes and values the current r1, r2.. value
            distance_constraints = [{self.rs_idxs[i]: self.rs[p][i] for i in range(2)} for p in points]

            # Use custom NoDaemonPool here, as there are several
            # multiprocessing events happening within the function
            with NoDaemonPool(processes=Config.n_cores) as pool:
                results = [pool.apply_async(func=get_point_species, args=(p, s, d, name, method, keywords, cores_per_process))
                           for p, s, d in zip(points, closest_species, distance_constraints)]

                for i, point in enumerate(points):
                    self.species[point] = results[i].get(timeout=None)

        logger.info('2D PES scan done')
        return None

    def _init_tensors(self, reactant, r1s, r2s):
        """Initialise the matrices of Species and distances"""
        logger.info(f'Initialising the {len(r1s)}x{len(r2s)} PES matrices')

        assert self.rs.shape == self.species.shape

        for i in range(len(self.rs)):
            for j in range(len(self.rs[i])):
                # Tuple of distances
                self.rs[i, j] = (r1s[i], r2s[j])

        # Copy of the reactant complex, whose atoms/energy will be set in the
        # scan
        self.species[0, 0] = deepcopy(reactant)

        return None

    def __init__(self, reactant, product, r1s, r1_idxs, r2s, r2_idxs):
        """
        A two dimensional potential energy surface

              /
          r2 /
            /
           /___________
                r1

        Arguments:
            reactant (autode.complex.ReactantComplex): Species at r1s[0] and r2s[0]
            r1s (np.ndarray): Bond length array in r1
            r1_idxs (tuple): Atom indexes that the PES will be calculated over in r1
            r2s (np.ndarray): Bond length array in r2
            r2_idxs (tuple): Atom indexes that the PES will be calculated over in r2
        """
        self.r1s, self.r2s = r1s, r2s
        self.n_points_r1, self.n_points_r2 = len(r1s), len(r2s)

        # Matrices to store the species and r1, r2 values at a point (i, j)
        self.species = np.empty(shape=(self.n_points_r1, self.n_points_r2), dtype=object)
        self.rs = np.empty(shape=(self.n_points_r1, self.n_points_r2), dtype=tuple)

        # List of tuples that contain atom indices of the coordinate r1
        self.rs_idxs = [r1_idxs, r2_idxs]

        # Populate the rs array and set species[0, 0] to the reactant
        self._init_tensors(reactant=reactant, r1s=r1s, r2s=r2s)

        # Coefficients of the fitted surface
        self.coeff_mat = None

        # Molecular graph of the product. Used to check that the products have
        # been made & find the MEP
        self.product_graph = product.graph


def get_ts_guess_2d(reactant, product, bond1, bond2, name, method, keywords,
                    polynomial_order=3, dr=0.1):
    """Scan the distance between two sets of two atoms and return a guess for
    the TS

    Arguments:
        reactant (autode.complex.ReactantComplex):
        product (autode.complex.ProductComplex):
        bond1 (autode.pes.ScannedBond):
        bond2 (autode.pes.ScannedBond):
        name (str): name of reaction
        method (autode.wrappers.base.ElectronicStructureMethod): electronic
        structure wrapper to use for the calcs
        keywords (autode.keywords.Keywords): keywords_list to use in the calcs

    Keyword Arguments:
        polynomial_order (int): order of polynomial to fit the data to
                                (default: {3})
        dr (float): Δr on the surface *absolute value*

    Returns:
        (autode.transition_states.ts_guess.TSguess)
    """
    logger.info(f'Getting TS guess from 2D relaxed potential energy scan,'
                f' using active bonds {bond1} and {bond2}')

    # Steps of +Δr if the final distance is greater than the current else -Δr.
    # Run at least a 3x3 PES
    n_steps1 = max(int(np.abs((bond1.final_dist - bond1.curr_dist) / dr)), 3)
    n_steps2 = max(int(np.abs((bond2.final_dist - bond2.curr_dist) / dr)), 3)

    if method.name in high_level_method_names:
        logger.warning('Limiting the number of steps to a maximum of 8 so <64 '
                       'high level optimisations have to be done')
        n_steps1 = min(n_steps1, 8)
        n_steps2 = min(n_steps2, 8)

    # Create a potential energy surface in the two active bonds and calculate
    pes = PES2d(reactant=reactant, product=product,
                r1s=np.linspace(bond1.curr_dist, bond1.final_dist, n_steps1),
                r1_idxs=bond1.atom_indexes,
                r2s=np.linspace(bond2.curr_dist, bond2.final_dist, n_steps2),
                r2_idxs=bond2.atom_indexes)

    pes.calculate(name=name, method=method, keywords=keywords)

    # Try to fit an analytic 2D PES to the surface and plot using matplotlib
    try:
        pes.fit(polynomial_order=polynomial_order)
    except FitFailed:
        logger.error('PES fit failed')
        return None

    if not pes.products_made():
        logger.error('Products were not made on the whole PES')
        return None

    # Get a TSGuess for the lowest energy MEP saddle point on the surface
    species = pes.get_species_saddle_point(name=name, method=method,
                                           keywords=keywords)

    if species is not None:
        return get_ts_guess(species, reactant, product, name=name)

    logger.error('No possible TSs found on the 2D surface')
    return None


def polyfit2d(x, y, z, order):
    """Takes x and y coordinates and their resultant z value, and creates a
    matrix where element i,j is the coefficient of the desired order polynomial
     x ** i * y ** j

    Arguments:
        x (np.array): flat array of x coordinates
        y (np.array): flat array of y coordinates
        z (np.array): flat array of z value at the corresponding x and y value
        order (int): max order of polynomial to work out

    Returns:
        np.array: matrix of polynomial coefficients
    """
    logger.info('Fitting 2D surface to polynomial in x and y')
    deg = np.array([int(order), int(order)])
    vander = polynomial.polyvander2d(x, y, deg)
    # vander matrix is matrix where each row i deals with x=x[i] and y=y[i],
    # and each item in the row has value x ** m * y ** n with (m,n) = (0,0),
    # (0,1), (0,2) ... (1,0), (1,1), (1,2) etc up to (order, order)
    coeff_mat, _, _, _ = np.linalg.lstsq(vander, z, rcond=None)
    return coeff_mat.reshape(deg + 1)




