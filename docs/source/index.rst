STRUCTURES25 Documentation
==========================



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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

This package started as the reimplementation of [M-OFDFT]_.


Citation
========

If you use this repository in your research, please cite the following paper:

.. raw:: html

   <div style="position: relative;">
   <button id="copy-button" onclick="copyToClipboard()" title="Copy" style="position: absolute; right: 10px; top: 10px; padding: 5px 10px; cursor: pointer; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;">
   <div id="copy-icon" style="display: block; transform: scaleX(-1); margin-top: -25px">⧉</div>
   <div id="copy-text" style="display: none;">Copied!</div>
   </button>
   <pre id="citation-text" style="background: #f8f8f8; padding: 10px; border: 1px solid #ddd; border-radius: 5px; overflow-x: auto;">@article{Remme_Stable_and_Accurate_2025,
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


References
==========

.. [M-OFDFT] Zhang, H.; Liu, S.; You, J.; Liu, C.; Zheng, S.; Lu, Z.; Wang, T.; Zheng, N.;
    Shao, B. M-OFDFT: "Overcoming the Barrier of Orbital-Free Density Functional
    Theory for Molecular Systems Using Deep Learning". arXiv September 28, 2023.
    http://arxiv.org/abs/2309.16578.

.. [Graphormer] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di
    He, Yanming Shen, Tie-Yan Liu: "Do Transformers really perform badly for graph
    representation?". Advances in Neural Information Processing Systems, 34:28877–28888, 2021.
