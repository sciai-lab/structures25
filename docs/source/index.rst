STRUCTURES25 Documentation
==========================

Welcome to the documentation for STRUCTURES25! This package enables Orbital-Free Density Functional Theory (OF-DFT) calculations by learning the kinetic energy functional from data using equivariant graph neural networks.
For more information on the **installation** and **usage**, please refer to our `GitHub repository <https://github.com/sciai-lab/structures25>`_.

.. raw:: html

   <div style="text-align: center; margin-bottom: 30px;">
   <img src="https://github.com/user-attachments/assets/00abb696-95e3-4aaa-857b-2b7548d45646" alt="STRUCTURES25 Overview" style="max-width: 100%;">
   </div>

The code is based on our publication `Stable and Accurate Orbital-Free Density Functional Theory Powered by Machine Learning <https://pubs.acs.org/doi/10.1021/jacs.5c06219>`_.
To cite STRUCTURES25 in your work, please use the following BibTeX entry:

.. raw:: html

   <div style="position: relative;">
   <button id="copy-button" onclick="copyToClipboard()" title="Copy" style="position: absolute; right: 10px; top: 10px; padding: 5px 10px; cursor: pointer; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;">
   <div id="copy-icon" style="display: block; transform: scaleX(-1); margin-top: -25px">⧉</div>
   <div id="copy-text" style="display: none;">Copied!</div>
   </button>
   <pre id="citation-text" style="background: #f8f8f8; padding: 10px; border: 1px solid #ddd; border-radius: 5px; overflow-x: auto; font-size: 0.85em;">@article{Remme_Stable_and_Accurate_2025,
       author = {Remme, Roman and Kaczun, Tobias and Ebert, Tim and Gehrig, Christof A. and
                 Geng, Dominik and Gerhartz, Gerrit and Ickler, Marc K. and Klockow, Manuel V. and
                 Lippmann, Peter and Schmidt, Johannes S. and Wagner, Simon and Dreuw, Andreas and
                 Hamprecht, Fred A.},
       title = {Stable and Accurate Orbital-Free Density Functional Theory Powered by Machine Learning},
       journal = {Journal of the American Chemical Society},
       year = {2025},
       volume = {147},
       number = {32},
       pages = {28851--28859},
       doi = {10.1021/jacs.5c06219},
       url = {https://doi.org/10.1021/jacs.5c06219}
   }</pre>
   <script>
   function copyToClipboard() {
       const text = document.getElementById('citation-text').textContent;
       const button = document.getElementById('copy-button');
       const icon = document.getElementById('copy-icon');
       const copyText = document.getElementById('copy-text');
       navigator.clipboard.writeText(text).then(function() {
           icon.style.display = 'none';
           copyText.style.display = 'block';
           button.style.background = '#d4edda';
           setTimeout(function() {
               icon.style.display = 'block';
               copyText.style.display = 'none';
               button.style.background = '#f0f0f0';
           }, 2000);
       }, function(err) {
           alert('Could not copy text: ', err);
       });
   }
   </script>
   </div>


Submodules
==========

.. toctree::
   :maxdepth: 1
   :caption: Package Reference


.. autosummary::
   :toctree: reference
   :template: module_template.rst
   :recursive:

   mldft.api
   mldft.datagen
   mldft.ml
   mldft.ofdft
   mldft.utils




References
==========

.. [M-OFDFT] Zhang, H.; Liu, S.; You, J.; Liu, C.; Zheng, S.; Lu, Z.; Wang, T.; Zheng, N.; Shao,
    B.: "Overcoming the barrier of orbital-free density functional theory for molecular systems
    using deep learning." Nat Comput Sci 4, 210–223 (2024).
    https://doi.org/10.1038/s43588-024-00605-8

.. [Graphormer] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di
    He, Yanming Shen, Tie-Yan Liu: "Do Transformers really perform badly for graph
    representation?". Advances in Neural Information Processing Systems, 34:28877–28888, 2021.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
