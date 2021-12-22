import autode.exceptions as ex
from autode.config import Config
from autode.log import logger
from autode.calculation import Calculation
from autode.methods import get_lmethod
from autode.ts_guess import get_ts_guess
from autode.utils import work_in
from autode.mol_graphs import find_cycles
import numpy as np

"""
The theory behind this original NEB implementation is taken from
Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)
"""
from autode.log import logger
from autode.input_output import atoms_to_xyz_file
from autode.calculation import Calculation
from autode.utils import work_in
from scipy.optimize import minimize
from multiprocessing import Pool
from copy import deepcopy
import numpy as np


def energy_gradient(image, method, n_cores):
    """Calculate energies and gradients for an image using a EST method"""

    calc = Calculation(name=f'{image.name}_{image.iteration}',
                       molecule=image.species,
                       method=method,
                       keywords=method.keywords.grad,
                       n_cores=n_cores)

    @work_in(image.name)
    def run():
        calc.run()
        image.grad = calc.get_gradients().flatten()
        image.energy = calc.get_energy()
        return None

    run()
    return image


def total_energy(flat_coords, images, method, n_cores):
    """Compute the total energy across all images"""
    images.set_coords(flat_coords)

    # Number of cores per process is the floored total divided by n images
    n_cores_pp = max(int(n_cores//len(images)), 1)

    logger.info(f'Calculating energy and forces for all images with '
                f'{n_cores} total cores and {n_cores_pp} per process')

    # Run an energy + gradient evaluation in parallel across all images
    with Pool(processes=n_cores) as pool:
        results = [pool.apply_async(func=energy_gradient,
                                    args=(images[i], method, n_cores_pp))
                   for i in range(1, len(images) - 1)]

        images[1:-1] = [result.get(timeout=None) for result in results]

    # Advance all the iteration numbers on the images to name correctly
    for i in range(1, len(images) - 1):
        images[i].iteration += 1

    all_energies = [image.energy for image in images]
    rel_energies = [energy - min(all_energies) for energy in all_energies]

    logger.info(f'Path energy = {sum(rel_energies):.5f}')
    return sum(rel_energies)


def get_force(im_l, im, im_r, k=0.005):
    """
    Compute F_i. Notation from:
    Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)

    also a copy in autode/common

    Arguments:
        im_l (autode.neb.Image): Left image (i-1)
        im (autode.neb.Image): (i)
        im_r (autode.neb.Image): Right image (i+1)
        k (float): Force constant of the spring in Ha / Å^2
    """
    # ΔV_i^max
    dv_max = max(np.abs(im_r.energy - im.energy),
                 np.abs(im_l.energy - im.energy))

    # ΔV_i^min
    dv_min = min(np.abs(im_r.energy - im.energy),
                 np.abs(im_l.energy - im.energy))

    # x_i-1,   x_i,   x_i+1
    x_l, x, x_r = [image.species.get_coordinates().flatten()
                   for image in (im_l, im, im_r)]
    # τ_i+
    tau_plus = x_r - x
    # τ_i-
    tau_minus = x - x_l

    if im_l.energy < im.energy < im_r.energy:
        tau = tau_plus

    elif im_r.energy < im.energy < im_l.energy:
        tau = tau_minus

    elif im_l.energy < im_r.energy:
        tau = tau_plus * dv_max + tau_minus * dv_min

    elif im_r.energy < im_l.energy:
        tau = tau_plus * dv_min + tau_minus * dv_max

    else:
        raise RuntimeError

    # Normalised τ vector
    hat_tau = tau / np.linalg.norm(tau)

    # F_i^s||
    f_parallel = (np.linalg.norm(x_r - x) * k -
                  np.linalg.norm(x - x_l) * k) * hat_tau

    # ∇V(x)_i|_|_ = ∇V(x)_i - (∇V(x)_i•τ) τ
    grad_perp = im.grad - np.dot(im.grad, hat_tau) * hat_tau

    # F_i = F_i^s|| -  ∇V(x)_i|_|_
    return f_parallel - grad_perp


def derivative(flat_coords, images, method, n_cores):
    """Compute the derivative of the total energy with respect to all
    components"""

    # Forces for the first image are fixed at zero
    forces = np.array(images[0].grad)

    # No need to calculate gradient as should already be there from energy eval
    for i in range(1, len(images) - 1):
        force = get_force(im_l=images[i - 1],
                          im=images[i],
                          im_r=images[i + 1])

        forces = np.append(forces, force)

    # Final zero set of forces
    forces = np.append(forces, images[-1].grad)

    # dV/dx is negative of the force
    logger.info(f'|F| = {np.linalg.norm(forces):.4f} Ha Å-1')
    return -forces


class Image:

    def __init__(self, name):
        """
        Image in a NEB

        Arguments:
            name (str):
        """
        self.name = name

        # Current optimisation iteration of this image
        self.iteration = 0

        self.species = None         # autode.species.Species
        self.energy = None          # float
        self.grad = None            # np.ndarray shape (3xn_atoms,)


class Images:

    def __len__(self):
        return len(self._list)

    def __setitem__(self, key, value):
        self._list[key] = value

    def __getitem__(self, item):
        return self._list[item]

    def coords(self):
        """Get a flat array of all components of every atom"""
        coords = np.array([])
        for image in self._list:

            coords = np.append(coords,
                               image.species.get_coordinates().flatten())
        return coords

    def set_coords(self, coords):
        """
        Set the flat array of coordinates to the species in the images

        Arguments:
            coords (np.ndarray): shape (num x n x 3,)
        """

        n_atoms = self._list[0].species.n_atoms
        coords = coords.reshape((len(self), n_atoms, 3))

        for i, image in enumerate(self._list):
            image.species.set_coordinates(coords[i])

        return None

    def __init__(self, num):

        self._list = [Image(name=str(i)) for i in range(num)]


class NEB:

    def print_geometries(self, name='neb'):
        """Print an xyz trajectory of the geometries in the NEB"""

        # Empty the file
        open(f'{name}.xyz', 'w').close()

        for i, image in enumerate(self.images):
            assert image.species is not None
            energy = image.energy if image.energy is not None else 'none'

            atoms_to_xyz_file(image.species.atoms,
                              f'{name}.xyz',
                              title_line=f'autodE NEB point {i}. E = {energy}',
                              append=True)
        return None

    def interpolate_geometries(self):
        """Generate simple interpolated coordinates for these set of images"""
        n = len(self.images)

        # Interpolate images between the starting point i=0 and end point i=n-1
        for i in range(1, n - 1):

            # Use a copy of the starting point for atoms, charge etc.
            self.images[i].species = deepcopy(self.images[0].species)

            # For all the atoms in the species translate an amount so the
            # spacing is even between the initial and final points
            for j, atom in enumerate(self.images[i].species.atoms):

                # Shift vector is final minus current
                shift = self.images[-1].species.atoms[j].coord - atom.coord
                # then an equal spacing is the i-th point in the grid
                atom.translate(vec=shift * (i / n))

        self.print_geometries()
        return None

    @work_in('NEB')
    def calculate(self, method, n_cores):
        """
        Optimise the NEB using forces calculated from electronic structure

        Arguments:
            method (autode.wrappers.ElectronicStructureMethod)
            n_cores (int)
        """
        self.print_geometries(name='neb_init')

        # Calculate energy on the first and final points
        for idx in [0, -1]:
            energy_gradient(self.images[idx], method=method, n_cores=n_cores)
            # Zero the forces so the end points don't move
            self.images[idx].grad = np.zeros(shape=self.images[idx].grad.shape)

        # Minimise the total energy across the path initial -> final points
        # with respect to the coordinates of all the intermediate images
        init_coords = self.images.coords()

        # Energy tolerance is ~1 kcal mol-1 per image
        etol = 0.0015 * len(self.images)
        logger.info(f'Minimising to ∆E < {etol:.4f} Ha on all NEB coordinates')

        result = minimize(total_energy,
                          x0=init_coords,
                          method='L-BFGS-B',
                          jac=derivative,
                          args=(self.images, method, n_cores),
                          tol=etol,
                          options={'maxfun': 30})

        logger.info(f'NEB path energy = {result.fun:.5f} Ha, {result.message}')

        # Set the optimised coordinates for all the images
        self.images.set_coords(result.x)
        self.print_geometries(name='neb_optimised')
        return None

    def get_species_saddle_point(self):
        """Yield a TS guesses for this NEB from all saddle points"""
        if any(image.energy is None for image in self.images):
            logger.error('Optimisation of at least one image failed')
            return None

        def is_saddle(j):
            """Is an image j amn approximate saddle point in the surface?"""
            e = self.images[j].energy
            return self.images[j-1].energy < e and self.images[j+1].energy < e

        # A saddle point cannot be either the start or the end point..
        peaks = [i for i in range(1, len(self.images) - 1) if is_saddle(i)]

        for peak_idx in sorted(peaks, key=lambda p: -self.images[p].energy):
            yield self.images[peak_idx].species

        return None

    def _init_from_species_list(self, s_list):
        """Initialise from a list of species rather than just end points"""

        self.images = Images(num=len(s_list))

        for i, image in enumerate(self.images):
            image.species = s_list[i]

        return None

    def _init_from_end_points(self, initial, final):
        """Initialise from the start and finish points of the NEB"""

        self.images[0].species = initial
        self.images[-1].species = final

        return None

    def __init__(self, initial_species=None, final_species=None, num=8,
                 species_list=None):
        """
        Nudged elastic band

        Arguments:
            initial_species (autode.species.Species):
            final_species (autode.species.Species):
            num (int): Number of images in the NEB
            species_list (list(autode.species.Species)): Intermediate images
                         along the NEB
        """
        self.images = Images(num=num)

        if species_list is not None:
            self._init_from_species_list(species_list)

        else:
            self._init_from_end_points(initial_species, final_species)

        logger.info(f'Initialised a NEB with {num} images')



def get_ts_guess_neb(reactant, product, method, fbonds=None, bbonds=None,
                     name='neb',
                     n=None,
                     generate_final_species=True):
    """
    Get a transition state guess using a nudged elastic band calculation. The
    geometry of the reactant is used as the fixed initial point and the final
    product geometry generated by driving a linear path to products, which is
    used as the initial guess for the NEB images

    Arguments:
        reactant (autode.species.Species):
        product (autode.species.Species):
        method (autode.wrappers.base.ElectronicStructureMethod):

    Keyword Arguments:
        fbonds (list(autode.pes.pes.FormingBond)):
        bbonds (list(autode.pes.pes.BreakingBond)):
        name (str):
        n (int): Number of images to use in the NEB
        generate_final_species (bool):

    Returns:
        (autode.transition_states.ts_guess.TSguess) or None:
    """
    logger.info('Generating a TS guess using a nudged elastic band')

    if generate_final_species and fbonds is not None and bbonds is not None:

        try:
            species_list = get_interpolated(reactant, fbonds, bbonds,
                                            max_n=calc_n_images(fbonds, bbonds),
                                            method=method)
        except ex.AtomsNotFound:
            logger.error('Failed to locate linear path')
            return None

        neb = NEB(species_list=species_list)

    # Otherwise using the reactant and product geometries
    else:
        assert n is not None
        neb = NEB(initial_species=reactant.copy(),
                  final_species=product.copy(),
                  num=n)
        neb.interpolate_geometries()

    # Calculate and generate the TS guess
    try:
        neb.calculate(method=method, n_cores=Config.n_cores)

    except ex.CouldNotGetProperty:
        logger.error('NEB failed')
        return None

    # Yield all peaks?
    for peak_species in neb.get_species_saddle_point():
        return get_ts_guess(peak_species, reactant, product, name=name)

    logger.warning('NEB did not generate a saddle point')
    return None


@work_in('NEB_init_path')
def get_interpolated(initial_species, fbonds, bbonds, max_n, method=None,
                     stop_thresh=0.02):
    """
    Generate the end point on the NEB by running a 1D scan, using by default a
    low-level method. Supprorts using different methods for the starting and
    final (end) points to the method used for the interpolation.
    If method is set then this will be used for both the end and intermediate
     methods

    Arguments:
        initial_species (autode.species.Species):
        fbonds (list(autode.pes.pes.FormingBond)):
        bbonds (list(autode.pes.pes.BreakingBond)):
        max_n (int): Maximum number of intermediate species to generate between
              the initial and final species

    Keyword Arguments:
        method (autode.wrappers.base.ElectronicStructureMethod):
        stop_thresh (float): Energy threshold in Ha to terminate the
                    interpolation if ∆E between two adjacent points is > this
                    and there is a peak in the surface, return the points.
                    default is ~ 10 kcal mol-1

    Returns:
        (list(autode.species.Species)): Set of intermediate species between
    """
    assert fbonds is not None and bbonds is not None
    logger.info('Generating the interpolated species reactant -> product using'
                f' a maximum of {max_n} intermediate points')

    bonds = active_bonds_no_rings(initial_species, fbonds, bbonds)

    # Calculate the uniform change in each bond distance from initial -> final
    deltas = [(b.final_dist - b.curr_dist)/(max_n-1) for b in bonds]

    # Set a dictionary of bond length constraints
    consts = {b.atom_indexes: b.curr_dist for b in bonds}

    species_set = []

    # Generate a species with a constrained geometry for each point in the path
    for i in range(max_n):

        if i == 0:
            species = initial_species.copy()

        else:
            species = species_set[i-1].copy()

            # Add the required change in every bond length to get to the final
            # distances at step n-1
            for j, atom_indexes in enumerate(consts.keys()):
                consts[atom_indexes] += deltas[j]

        # Run the constrained optimisation
        if method is None:
            method = get_lmethod()

        opt = Calculation(name=f'{species.name}_constrained_opt{i}',
                          molecule=species,
                          method=method,
                          keywords=method.keywords.opt,
                          n_cores=Config.n_cores,
                          distance_constraints=consts)

        # Set the optimised atoms - can raise AtomsNotFound
        species.optimise(method=method, calc=opt)
        species_set.append(species)

        # Early stopping if a ~saddle point has already been traversed, must be
        # in the second half of the scan and above an energy threshold for ∆E
        if all((i > 1,
                i > max_n//2,
                contains_peak(species_set),
                species.energy - species_set[i-1].energy > stop_thresh)):

            logger.warning(f'Path contained an energy peak and the point '
                           f'before this one had a lower energy - stopping the'
                           f' interpolation on step {i}')

            return species_set

    logger.info('Generated initial NEB path')
    return species_set


def active_bonds_no_rings(initial_species, fbonds, bbonds):
    """
    From forming and breaking bonds determine which should be used as the
    set of active bonds that define bond constraints. Any breaking bonds that
    form rings with forming bonds in should be removed to try and avoid very
    high energy points in the potential. For example:

               H
    forming-> / \  <-breaking
             C--C
               ^
               |
             normal

    by default the breaking bonds final distance will be the current x1.5 or so
    which will push the H too far away (as it's migrating rather than leaving).
    A much better strategy is to only scan the forming C-H bond and leave the
    breaking bond to do it's own thing.

    Arguments:
        initial_species (autode.species.Species):
        fbonds (list(autode.pes.pes.FormingBond)):
        bbonds (list(autode.pes.pes.BreakingBond)):

    Returns:
        (list(autode.pes.pes.ScannedBond)):
    """
    logger.info('Removing breaking bonds that are also in the set of forming'
                'bonds')

    graph = initial_species.graph.copy()

    # Add all the active bonds if they don't already exist
    for bond in bbonds + fbonds:
        if bond.atom_indexes not in graph.edges:
            graph.add_edge(*bond.atom_indexes)

    # Find the rings in this molecular graph
    rings = find_cycles(graph)

    def in_ring(bond):
        (i, j) = bond.atom_indexes

        for fbond in fbonds:
            (m, n) = fbond.atom_indexes

            for ring in rings:
                if all(idx in ring for idx in (i, j, m, n)):
                    return True

        return False

    # Remove all the breaking bonds that form a ring that also include at
    # least one forming bond
    return fbonds + [bbond for bbond in bbonds if not in_ring(bbond)]


def contains_peak(species_list):
    """
    Does this list of species contain a peak in the energy?

    Arguments:
        species_list (list(autode.species.Species):

    Returns:
        (bool):
    """
    if any(species.energy is None for species in species_list):
        logger.warning('Cannot determine if path contains a peak, an E=None')
        return False

    for i, species in enumerate(species_list):

        # Cannot be a peak on the end points
        if i == 0 or i == len(species_list) - 1:
            continue

        # Points either side of this species must be lower in energy
        if all(species_list[k].energy < species.energy for k in (i-1, i+1)):
            return True

    return False


def calc_n_images(fbonds, bbonds, average_spacing=0.15):
    """
    Calculate the number of images to use in a NEB calculation based on the
    active bonds. Will use a number so the average ∆r between each step on
    each coordinate is ~average_spacing Å

    Arguments:
        fbonds (list(autode.pes.pes.FormingBond)):
        bbonds (list(autode.pes.pes.BreakingBond)):

    Keyword Arguments:
        average_spacing (float):

    Returns:
        (int): Number of images
    """
    differences = [(b.final_dist - b.curr_dist) for b in fbonds + bbonds]
    average_abs_difference = np.average(np.abs(np.array(differences)))

    return int(average_abs_difference / average_spacing)
