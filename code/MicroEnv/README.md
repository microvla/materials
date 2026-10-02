# MicroEnv

Complete simulator source and 15 vessel layouts are in `MicroEnv/`.
Install `requirements.txt` in a compatible Python environment. From the parent
directory, import `MicroEnv.MicroEnv.torch_vessel_env`.

RL and OpenVLA each contain their own physical copy of this simulator, so they
do not depend on this folder. No evaluation entrypoints or weights are included.
