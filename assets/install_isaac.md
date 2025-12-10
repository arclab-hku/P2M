# Install Isaac Sim and Isaac Lab

## A. Install Isaac Sim

We use **[Isaac Sim 4.1.0](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.1.0-rc.7%2B4.1.14801.71533b68.gl.linux-x86_64.release.zip)** and recommand binaries installation.
```bash
# set the install location as ${HOME} for example
mkdir ~/isaacsim
unzip "isaac-sim-standalone@4.1.0-rc.7+4.1.14801.71533b68.gl.linux-x86_64.release.zip" -d ~/isaacsim
cd ~/isaacsim
```
```bash
# Set environment variables (you may also add these to ~/.bashrc or ~/.zshrc):
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

## B. Install Isaac Lab
We use the official GitHub release, ensure that the correct version is selected:

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
git checkout de76c2e9d193ec42b834b920f00ebc7c2bff622a
cd IsaacLab

# Create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

# Install Isaac Lab extensions (pip editable mode):
./isaaclab.sh --install # or "./isaaclab.sh -i"
```