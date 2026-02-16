fitting (legacy wrapper)
========================

The :mod:`kmcpy.fitting` module is retained for backward compatibility.
Its ``Fitting`` class delegates the fitting implementation to
:mod:`kmcpy.models.fitting.fitter`.

Use :mod:`kmcpy.models.fitting.fitter` for new code.

.. autoclass:: kmcpy.fitting.Fitting
    :members:
