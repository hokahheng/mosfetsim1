"""Electrical equation compatibility API; implementations live in physics.py."""
from . import physics


def drain_current(m, vgs, vds):
    return physics.drain_current(m, vgs, vds)


def gm(m, vgs, vds):
    return physics.transconductance(m, vgs, vds)


def go(m, vgs, vds):
    return physics.output_conductance(m, vgs, vds)
