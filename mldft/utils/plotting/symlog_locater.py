import numpy as np
from matplotlib.ticker import Locator


class MinorSymLogLocator(Locator):
    """Dynamically find minor tick positions based on the positions of major ticks for a symlog
    scaling.

    Based on [this stack exchange]
    (https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale)
    """

    def __init__(self, linthresh, nints=10):
        """Ticks will be placed between the major ticks.

        The placement is linear for x between -linthresh and linthresh, otherwise its
        logarithmically. nints gives the number of intervals that will be bounded by the minor
        ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]  # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0.0 and (
            (majorlocs[0] != self.linthresh and dmlower > self.linthresh)
            or (dmlower == self.linthresh and majorlocs[0] < 0)
        ):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * 10.0)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0.0 and (
            (np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh)
            or (dmupper == self.linthresh and majorlocs[-1] > 0)
        ):
            majorlocs = np.append(majorlocs, majorlocs[-1] * 10.0)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1] + self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.0

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError("Cannot get tick locations for a " "%s type." % type(self))
