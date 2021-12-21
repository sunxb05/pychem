from copy import deepcopy


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
