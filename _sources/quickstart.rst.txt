Quickstart
==========

This page shows a minimal end-to-end example: a single borehole with a
Single U-tube configuration, one soil layer, and no inter-borehole thermal
interference. It is the simplest possible pyCaRM simulation and a good
starting point before moving to multi-borehole fields.

Imports
-------

.. code-block:: python

    import numpy as np

    from carm import (
        BoreholeGeometry,
        BoreholeMesh,
        BoreholeThermalProperties,
        SingleUtube,
    )
    from carm import EnvironmentalProperties, EnvironmentalTimeSeries
    from carm import Fluid
    from carm import GroundGeometry, GroundMesh
    from carm import PhysicalModel
    from carm import Simulation

Fluid
-----

Define the thermal and physical properties of the heat carrier fluid:

.. code-block:: python

    fluid = Fluid(
        k_w=0.5687,       # thermal conductivity [W/(m·K)]
        rho_w=1000.14,    # density [kg/m³]
        cp_w=4207.4,      # specific heat [J/(kg·K)]
        ni_w=1.496e-6,    # kinematic viscosity [m²/s]
    )

Borehole
--------

Assemble the borehole geometry, mesh, thermal properties, and pipe
configuration into a ``SingleUtube`` object:

.. code-block:: python

    bore_geom = BoreholeGeometry(Lbore=100, D0=0.15)
    bore_mesh = BoreholeMesh(m_mesh=40)
    bore_th_props = BoreholeThermalProperties(cp_0=1460, rho_0=1655, k0=1.8)

    props_b = SingleUtube(
        geom=bore_geom,
        mesh=bore_mesh,
        thermalprops=bore_th_props,
        fluid=fluid,
        Rp0=0.25,
        RppB=0.72,
        pipe_spacing=0.0823,
        pipe_thick=0.003,
        Dpi=0.026,
        n_pipes=2,
    )

Ground
------

Define the ground geometry and radial/axial mesh. ``stratification`` is a
list of layers, each specified as ``(k, rho, cp, permeability)``:

.. code-block:: python

    ground_geom = GroundGeometry(D0=0.15, L=100, L_sup=1, L_inf=10, rn=10)
    ground_mesh = GroundMesh(n_mesh=20, m_mesh=40, m_mesh_sup=4, m_mesh_inf=40)

    stratification = [(1.8, 947.37, 1900, 111)]   # single homogeneous layer

Environmental boundary conditions
----------------------------------

Time-varying surface boundary conditions are built from plain NumPy arrays
via :meth:`~carm.external_environment.EnvironmentalTimeSeries.from_array`.
Here a synthetic seasonal + daily profile stands in for real measured data:

.. code-block:: python

    dt      = 3600          # timestep [s]
    n_steps = 276

    hours = np.arange(n_steps) * dt / 3600.0
    T_ext = 12.5 - 11.0 * np.cos(2 * np.pi * hours / 8760.0) \
                  + 3.0 * np.sin(2 * np.pi * hours / 24.0)
    daylight = np.clip(np.sin(2 * np.pi * (hours % 24.0 - 6.0) / 24.0), 0.0, None)
    seasonal = 0.5 - 0.5 * np.cos(2 * np.pi * hours / 8760.0)
    SolarRad = 900.0 * daylight * seasonal

    env_input = EnvironmentalTimeSeries.from_array(Tm=13, T_ext=T_ext, SolarRad=SolarRad)
    env_props = EnvironmentalProperties(
        R_ext=0.04,
        absorptance=0.7,
        eps=0.95,
        At=10,
        tau=0,
        tau_y=365 * 24 * 3600,
        tau_shift=210 * 24 * 3600,
    )

.. note::
   pyCaRM has no built-in Excel reader: ``from_array`` is the only way to
   build an :class:`~carm.external_environment.EnvironmentalTimeSeries` (or
   a :class:`~carm.field_layout.FieldInput`) from data. If your data lives
   in a spreadsheet, read it with ``pandas.read_excel`` and pass the
   resulting columns to ``from_array`` — see ``examples/Excel_input.py``
   for the full recipe.

Physical model
--------------

Combine all components into a ``PhysicalModel``. ``Tg`` is the undisturbed
ground temperature [°C]:

.. code-block:: python

    model = PhysicalModel(
        ground_geom=ground_geom,
        ground_mesh=ground_mesh,
        borehole=props_b,
        fluid=fluid,
        Tg=13,
        stratification=stratification,
    )

Simulation
----------

Define the inlet fluid temperature ``Tf1`` and mass flow rate ``mw_tot``
as arrays of shape ``(n_groups, n_steps)``. For a single borehole in
parallel (the default), ``n_groups = 1``:

.. code-block:: python

    Tf1    = np.full((1, n_steps), 2.0)      # inlet temperature [°C]
    mw_tot = np.full((1, n_steps), 0.1657)   # mass flow rate [kg/s]

    simulation = Simulation(
        model=model,
        envinput=env_input,
        envprops=env_props,
        timesteps=dt,
        n_steps=n_steps,
        mw_tot=mw_tot,
        Tf1=Tf1,
    )

    T_history = simulation.run()

``T_history`` has shape ``(n_steps + 1, n_bhes, n_dof)``, where index 0
along the first axis is the initial condition. See :doc:`output` for how
to extract temperatures of interest from this array.

.. note::
   ``run()`` does not write anything to disk unless asked: pass
   ``save_results=True`` to also dump a timestamped ``.npz`` archive to a
   ``results/`` directory. See ``examples/SingleUtube.py`` for a runnable
   use of this option.

Extracting the outlet temperature
----------------------------------

As a quick check, extract and plot the outlet fluid temperature over time:

.. code-block:: python

    import matplotlib.pyplot as plt

    m_mesh_sup = 4
    n_mesh     = 20
    m_mesh     = 40

    nsup    = m_mesh_sup + 1
    nground = n_mesh * m_mesh

    Tfout = T_history[1:, 0, nsup + nground + (props_b.n_equations - 1)]
    time  = np.arange(1, n_steps + 1) * dt / 3600   # [h]

    plt.plot(time, Tfout)
    plt.xlabel("Time [h]")
    plt.ylabel("Outlet temperature [°C]")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

.. note::
   For multi-borehole fields in parallel or series, see the examples in
   the ``examples/`` directory of the repository.