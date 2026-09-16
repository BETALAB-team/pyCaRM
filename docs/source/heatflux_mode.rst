Heat Flux Mode
==============

pyCaRM can drive the simulation from a building-side thermal load and supply
temperature instead of a prescribed inlet fluid temperature. In this mode,
the inlet fluid temperature ``Tf1`` is not an input: it is solved for at each
timestep so that the heat extracted from (or rejected to) the ground matches
the building load once heat pump performance is accounted for. This mode is
enabled by setting ``heat_flux=True`` on :class:`~carm.simulation.Simulation`.

Overview
--------

At each timestep, the building thermal load :math:`Q_{building}` and the
supply temperature :math:`T_{supply}` are given as input. Depending on the
sign of :math:`Q_{building}`, the corresponding heat pump performance
coefficient is evaluated as a function of the temperature lift between
:math:`T_{supply}` and the fluid outlet temperature :math:`T_{f,out}`:

.. math::

   \mathrm{COP} = f(T_{supply} - T_{f,out}), \quad Q_{building} > 0 \text{ (heating)}

.. math::

   \mathrm{EER} = f(T_{f,out} - T_{supply}), \quad Q_{building} < 0 \text{ (cooling)}

The heat exchanged with the ground follows from an energy balance across the
heat pump:

.. math::

   Q_{ground} = -Q_{building}\left(1 - \frac{1}{\mathrm{COP}}\right), \quad Q_{building} > 0

.. math::

   Q_{ground} = -Q_{building}\left(1 + \frac{1}{\mathrm{EER}}\right), \quad Q_{building} < 0

Performance function
---------------------

:math:`f` is currently **hardcoded** in the solver as the quadratic
ground-source heat pump (GSHP) regression from Ruhnau *et al.* (2019) [1]_:

.. math::

   f(\Delta T) = 10.29 - 0.21\,\Delta T + 0.0012\,\Delta T^2

where :math:`\Delta T` is the temperature lift defined above. The same
function is used for both COP and EER in the current implementation.

This is not exposed as a configurable parameter: to use a different
correlation (e.g. air-source or water-source coefficients from the same
reference, or a manufacturer-specific curve), fork the repository and edit
``f_COP`` directly in ``carm/simulation/solver.py``, in both
``_run_parallel`` and ``_run_series``.

Picard iteration
-----------------

Since :math:`Q_{ground}` depends on :math:`T_{f,out}`, which in turn depends
on the inlet temperature :math:`T_{f1}` computed from :math:`Q_{ground}`, the
two are solved iteratively at each timestep (Picard iteration):

.. math::

   T_{f1} = \bar{T}_{f,out} + \frac{Q_{ground}}{\dot{m}_{tot}\, c_{p,w}}

The system is re-solved with the updated :math:`T_{f1}`, and the resulting
:math:`T_{f,out}` is compared against the heat pump energy balance:

.. math::

   \left| Q_{ground} - \left(-\dot{m}_{tot}\, c_{p,w}\,(T_{f,out} - T_{f1})\right) \right| < \varepsilon

with :math:`\varepsilon = 10` W by default. The iteration stops on
convergence or after 200 iterations, whichever comes first.

Parallel vs. series
--------------------

In parallel configurations, :math:`\bar{T}_{f,out}` is the mass-flow-weighted
average outlet temperature across all boreholes, and the Picard loop solves
for a single field-level :math:`T_{f1}` applied to every borehole.

In series configurations, the same balance is evaluated at the **group**
level: :math:`\bar{T}_{f,out}` is the mass-flow-weighted average of the
outlet temperature of the **last** borehole in each series chain, and the
Picard loop solves for a per-group inlet temperature (``Tf1_groups``), which
is then propagated along each chain by the internal borehole-to-borehole
fluid coupling.

Usage
-----

When ``heat_flux=True``:

- ``Tf1`` must be ``None`` (it is computed internally).
- ``Q_buildings`` and ``T_supply`` must be provided as 1D time series of
  length ``n_steps``, shared across the whole field.
- ``mw_tot`` follows the usual shape convention: ``(n_bhes, n_steps)`` in
  parallel mode, ``(n_groups, n_steps)`` in series mode.

.. code-block:: python

   Q_buildings = np.zeros(n_steps, dtype=np.float64)
   Q_buildings[: n_steps // 2] = 5000.0  # constant extraction, first half of the year

   T_supply = np.full(n_steps, 45.0, dtype=np.float64)

   simulation = Simulation(
       model=model, envinput=env_input, timesteps=dt, n_steps=n_steps,
       envprops=env_props, mw_tot=mw_tot, Tf1=None,
       heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply,
   )
   T_history = simulation.run(parallel=True)  # or series=True with groups defined

.. note::

   To simulate the plant being fully off (as opposed to a zero load with the
   pump still running), set both ``Q_buildings`` and ``mw_tot`` to zero over
   the same period. A zero load with nonzero mass flow only skips the
   COP/EER/Picard block; it does not stop circulation.

Output
------

When ``heat_flux=True``, the following arrays are populated, and included in
the ``.npz`` archive if ``run(..., save_results=True)`` is used:

- ``COP``, ``EER``: heat pump performance coefficients per timestep (``NaN``
  where the corresponding mode is not active).
- ``Q_ground``: heat exchanged with the ground per timestep.
- ``abs_err``: Picard iteration residual at convergence (or at the last
  iteration if the maximum was reached).

References
----------

.. [1] Ruhnau, O., Hirth, L. & Praktiknjo, A. Time series of heat demand and
   heat pump efficiency for energy system modeling. *Scientific Data* **6**,
   189 (2019). https://doi.org/10.1038/s41597-019-0199-y