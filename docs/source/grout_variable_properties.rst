Variable Grout Properties
=========================

The :class:`~carm.properties.SoilMoisture` class updates grout thermal
conductivity, volumetric heat capacity, and density over time as a function
of volumetric water content, to account for irrigation-driven moisture
changes around the borehole.

At each timestep, a water volume balance is tracked around the borehole:
irrigation adds water, while gravity drainage and evaporation remove it.
Drainage loss follows the Brooks-Corey (1964) unsaturated hydraulic
conductivity model, so it depends on the current water content relative to
the soil's residual and saturated values (``theta_r``, ``theta_s``) and on
two soil-specific parameters, the saturated hydraulic conductivity ``Ks``
and the pore-size distribution index ``lambda``. Evaporation loss is driven
by the thermal power exchanged with the ground. The resulting water volume
is clipped between the residual and saturated volumes and converted back to
a volumetric water content, which feeds the Chung-Horton/de Vries
correlations for thermal conductivity, heat capacity, and density.

.. autoclass:: carm.properties.SoilMoisture
   :members:
   :undoc-members:


When to use it
---------------

``SoilMoisture`` is only instantiated when ``water_input`` is provided in
:class:`~carm.external_environment.EnvironmentalTimeSeries`. In that case,
``D_irrigation`` and ``perf_fraction`` must also be set on
:class:`~carm.borehole.BoreholeGeometry`, since they define the irrigation
pipe surface area used in the water balance:

.. code-block:: python

   A_irr = pi * borehole.D_irrigation * borehole.perf_fraction * borehole.Lbore


Field-level update
--------------------

Properties are recomputed once per timestep at the field level, not once
per borehole. All boreholes share the same grout composition and receive
the same irrigation input; the evaporation term uses the mean heat flux
across the field rather than a per-borehole value:

.. code-block:: python

   q = np.mean(self.q_nbhes[step - 1])

Sensitivity of the water content to the per-borehole heat flux (as opposed
to the field average) was verified to be negligible over the operating
range (0–10 kW).


Example
-------

.. code-block:: python

   from carm import BoreholeGeometry, BoreholeThermalProperties

   geom = BoreholeGeometry(
       Lbore=30, D0=0.5,
       D_irrigation=0.030, perf_fraction=0.5,
   )
   thermalprops = BoreholeThermalProperties(
       cp_0=796.14, rho_0=1587.09, k0=1.0, soil_type="sand",
   )

.. note::

   ``rho_0`` above is the dry soil density, required when ``water_input``
   is active. For simulations without irrigation, use the density at
   residual water content instead.