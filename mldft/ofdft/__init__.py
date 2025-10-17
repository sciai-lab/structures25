r"""Subpackage containing OFDFT implementation.

The electronic energy is given by the following functional:

.. math::

    E[\rho] = E_{kin}[\rho] + E_{nuc}[\rho] + E_{H}[\rho] + E_{xc}[\rho]

where :math:`E_{kin}[\rho]` is the kinetic energy functional, :math:`E_{nuc}[\rho]` the nuclear
electron attraction energy, :math:`E_{H}[\rho]` the Hartree energy (electron-electron coulom
repulsion) and :math:`E_{xc}[\rho]` the exchange correlation energy.

The nuclear electron attraction energy is given by:

.. math::

    E_{nuc}[\rho] = \int v_{nuc}(r) \rho(r) \mathrm{d}r

where :math:`v_{nuc}(r)` is the nuclear electron attraction potential.

The Hartree energy is given by:

.. math::

        E_{H}[\rho] = \frac{1}{2} \int \int \frac{\rho(r) \rho(r')}{|r-r'|} \mathrm{d}r
        \mathrm{d}r'

To represent the electron density [M-OFDFT]_ we use the Linear Combination of Atomic Basis Functions
(LCAB) approach. The density is then given by:

.. math::

    \rho(r) = \sum_\mu p_\mu \omega_\mu(r)

where :math:`p_\mu` are the coefficients of the atomic basis and :math:`\omega_\mu(r)`
are the basis functions, for which we use the usual gaussian type 'atomic orbitals'.

To obtain the groundstate density and energy we need to minimize the energy functional w.r.t. the
electron density.
This is done by minimizing the energy functional w.r.t. the coefficients of the
density expansion.

.. math::

    \min_{\mathbf{p}} E[\rho_p]

which we solve by gradient descent.

.. math::

    \nabla E[\rho_p] = \nabla E_{kin}[\rho_p] + \nabla E_{xc}[\rho_p] + \nabla
    E_{H}[\rho_p] + \nabla E_{nuc}[\rho_p]

Additionally, the particle number needs to be preserved. To achieve this we project
the gradient onto the subspace of constant particle number. Which results in the
following update rule:

.. math::

    p^{(n+1)} = p^{(n)} - \epsilon \left( \mathbf{1} - \frac{ww^T}{w^Tw} \right)
    \nabla E[\rho_p]

with :math:`w` being the normalization vector of the basis functions
:math:`w_\mu = \int \omega_\mu (r) \mathrm{d}r` and :math:`\epsilon` being the
learning rate/step length of the gradient descent.
"""
