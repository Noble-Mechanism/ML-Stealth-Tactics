"""Spec 5 network interface (docs/specs/05-network-interface.md).

- ``view``: ``BlueView``, the restricted per-jet picture (no truth) built by
  ``World.blue_view``.
- ``observation``: ``ObsSpec`` (231 inputs, named slices, fixed scales) and
  ``build_observation``.
- ``action``: ``ActionSpec`` (13 outputs), ``decode_action``, ``encode_commands``.
- ``controller``: ``NetworkBlueController`` (1 s decisions, held commands,
  radar dwell, pair bit).
- ``adapters``: ``ScriptedViaInterface``, ``HandBlue``, ``RandomMLPPolicy``.
"""

from .observation import ObsSpec, OBS_SPEC, build_observation, ObsInfo  # noqa: F401
from .action import ActionSpec, ACTION_SPEC, Action, decode_action, encode_commands  # noqa: F401
from .controller import NetworkBlueController  # noqa: F401
