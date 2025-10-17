.. MLDFT documentation master file, created by
   sphinx-quickstart on Mon Nov 13 17:40:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of STRUCTURES25!
=============================================


.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   about.rst

.. autosummary::
   :toctree: reference
   :template: module_template.rst
   :recursive:

   mldft.datagen
   mldft.ml
   mldft.ofdft
   mldft.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

This package started as the reimplementation of [M-OFDFT]_.

References
==========

.. [M-OFDFT] Zhang, H.; Liu, S.; You, J.; Liu, C.; Zheng, S.; Lu, Z.; Wang, T.; Zheng, N.;
    Shao, B. M-OFDFT: "Overcoming the Barrier of Orbital-Free Density Functional
    Theory for Molecular Systems Using Deep Learning". arXiv September 28, 2023.
    http://arxiv.org/abs/2309.16578.

.. [Graphormer] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di
    He, Yanming Shen, Tie-Yan Liu: "Do Transformers really perform badly for graph
    representation?". Advances in Neural Information Processing Systems, 34:28877–28888, 2021.

..
    ADIIS is no longer supported which is why it is commented out.
    [ADIIS] Hu, Xiangqian, and Weitao Yang: "Accelerating self-consistent field convergence with
    the augmented Roothaan–Hall energy function". The Journal of chemical physics 132.5 (2010).
