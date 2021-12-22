class KeywordsSet:

    def __init__(self, low_opt=None, grad=None, opt=None, opt_ts=None,
                 hess=None, optts_block='', sp=None):
        """
        Keywords used to specify the type and method used in electronic
        structure theory calculations. The input file for a single point
        calculation will look something like:

        ---------------------------------------------------------------------
        <keyword line directive> autode.Keywords.sp[0] autode.Keywords.sp[1]
        autode.Keywords.optts_block

        <coordinate directive> <charge> <multiplicity>
        .
        .
        coordinates
        .
        .
        <end of coordinate directive>
        ---------------------------------------------------------------------
        Keyword Arguments:

            low_opt (list(str)): List of keywords for a low level optimisation
            grad (list(str)): List of keywords for a gradient calculation
            opt (list(str)): List of keywords for a low level optimisation
            opt_ts (list(str)): List of keywords for a low level optimisation
            hess (list(str)): List of keywords for a low level optimisation
            optts_block (str): String as extra input for a TS optimisation
            sp  (list(str)): List of keywords for a single point calculation
        :return:
        """

        self.low_opt = OptKeywords(low_opt)
        self.opt = OptKeywords(opt)
        self.opt_ts = OptKeywords(opt_ts)

        self.grad = GradientKeywords(grad)
        self.hess = HessianKeywords(hess)

        self.sp = SinglePointKeywords(sp)

        self.optts_block = optts_block


class Keywords:

    def __str__(self):
        return '_'.join(self.keyword_list)

    def copy(self):
        return deepcopy(self.keyword_list)

    def append(self, item):
        assert type(item) is str

        # Don't re-add a keyword that is already there
        if any(kw.lower() == item.lower() for kw in self.keyword_list):
            return

        self.keyword_list.append(item)

    def remove(self, item):
        self.keyword_list.remove(item)

    def __getitem__(self, item):
        return self.keyword_list[item]

    def __init__(self, keyword_list):
        """
        Read only list of keywords

        Args:
            keyword_list (list(str)): List of keywords used in a QM calculation
        """
        self.keyword_list = keyword_list if keyword_list is not None else []

        # Input will break if all the keywords are not strings
        assert all(type(kw) is str for kw in self.keyword_list)


class OptKeywords(Keywords):
    pass


class HessianKeywords(Keywords):
    pass


class GradientKeywords(Keywords):
    pass


class SinglePointKeywords(Keywords):
    pass

class Config:
    # -------------------------------------------------------------------------
    # Total number of cores available
    #
    n_cores = 2
    #
    # -------------------------------------------------------------------------
    # Per core memory available in MB
    #
    max_core = 4000
    #
    # -------------------------------------------------------------------------
    # DFT code to use. If set to None then the highest priority available code
    # will be used:
    # 1. 'orca', 2. 'g09' 3. 'nwchem'
    #
    # hcode = 'nwchem'
    hcode = 'xtb'   
    #
    # -------------------------------------------------------------------------
    # Semi-empirical/tight binding method to use. If set to None then the
    # highest priority available will be used:   1. 'xtb', 2. 'mopac'
    #
    lcode = 'xtb'
    #
    # -------------------------------------------------------------------------
    # When using explicit solvent is stable this will be uncommented
    #
    # explicit_solvent = False
    #
    # -------------------------------------------------------------------------
    # Setting to keep input files, otherwise they will be removed
    #
    keep_input_files = True
    #
    # -------------------------------------------------------------------------
    # By default templates are saved to /path/to/autode/transition_states/lib/
    # unless ts_template_folder_path is set
    #
    ts_template_folder_path = None
    #
    # Whether or not to create and save transition state templates
    make_ts_template = True
    # -------------------------------------------------------------------------
    # Save plots with dpi = 400
    high_quality_plots = True
    #
    # -------------------------------------------------------------------------
    # RMSD in angstroms threshold for conformers. Larger values will remove
    # more conformers that need to be calculated but also reduces the chance
    # that the lowest energy conformer is found
    #
    rmsd_threshold = 0.3
    #
    # -------------------------------------------------------------------------
    # Total number of conformers generated in find_lowest_energy_conformer()
    # for single molecules/TSs
    #
    num_conformers = 300
    # -------------------------------------------------------------------------
    # Maximum random displacement in angstroms for conformational searching
    #
    max_atom_displacement = 4.0
    # -------------------------------------------------------------------------
    # Number of evenly spaced points on a sphere that will be used to generate
    # NCI and Reactant and Product complex conformers. Total number of
    # conformers will be:
    #   (num_complex_sphere_points ×
    #              num_complex_random_rotations) ^ (n molecules in complex - 1)
    #
    num_complex_sphere_points = 10
    # -------------------------------------------------------------------------
    # Number of random rotations of a molecule that is added to a NCI or
    # Reactant/Product complex
    #
    num_complex_random_rotations = 10
    # -------------------------------------------------------------------------
    # For more than 2 molecules in a complex the conformational space explodes,
    # so limit the maximum number to this value
    #
    max_num_complex_conformers = 300
    # -------------------------------------------------------------------------
    # Use the high + low level method to find the lowest energy
    # conformer, to use energies at the low_opt level of the low level code
    # set this to False
    #
    hmethod_conformers = True
    # -------------------------------------------------------------------------

    class G09:
        # ---------------------------------------------------------------------
        # Parameters for g09                 https://gaussian.com/glossary/g09/
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        disp = 'EmpiricalDispersion=GD3BJ'
        grid = 'integral=ultrafinegrid'
        ts_str = ('Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, '
                  'NoTrustUpdate)')

        keywords = KeywordsSet(low_opt=['PBEPBE/Def2SVP', 'Opt=Loose',
                                        disp, grid],
                               grad=['PBE1PBE/Def2SVP', 'Force(NoStep)',
                                     disp, grid],
                               opt=['PBE1PBE/Def2SVP', 'Opt',
                                    disp, grid],
                               opt_ts=['PBE1PBE/Def2SVP', 'Freq',
                                       disp, grid, ts_str],
                               hess=['PBE1PBE/Def2SVP', 'Freq', disp, grid],
                               sp=['PBE1PBE/Def2TZVP', disp, grid])

        # Only SMD implemented
        implicit_solvation_type = 'smd'

    class XTB:
        # ---------------------------------------------------------------------
        # Parameters for xtb                  https://github.com/grimme-lab/xtb
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        keywords = KeywordsSet()
        #
        # Only GBSA implemented
        implicit_solvation_type = 'gbsa'

from copy import deepcopy


