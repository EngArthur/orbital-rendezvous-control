"""
Orbital Rendezvous Control System

A complete implementation of autonomous orbital rendezvous guidance, navigation,
and control based on Okasha & Newman (2014).

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

__version__ = "1.0.0"
__author__ = "Arthur Allex Feliphe Barbosa Moreno"
__email__ = "arthur.moreno@ime.eb.br"

from .dynamics.orbital_elements import OrbitalElements
from .utils.constants import *

