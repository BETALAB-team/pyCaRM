# pyCaRM

Simulation library for borehole heat exchanger (BHE) systems: models the transient
thermal response of the ground and borehole, and time-steps it under either a
prescribed inlet fluid temperature or a building-side thermal load.

## Language

**HeatFluxMode**:
Configuration passed to `Simulation` to drive it from a building-side thermal load
and supply temperature instead of a prescribed inlet fluid temperature. Bundles the
`Q_buildings`/`T_supply` time series with the (independently configurable) COP/EER
heat pump performance curves used to convert the building load into ground heat
exchange.
_Avoid_: heat flux mode (lowercase/informal), heat pump mode
