# Installing the simulator on a new computer, from zero

Target: a Windows 11 laptop or PC with an NVIDIA RTX GPU. About 1.5 hours, most of it downloads.
At the end `run.bat` opens the launcher and every backend starts from it.

What runs where:

| Part | Runs on | Needed for |
|---|---|---|
| The project folder | Windows drive, `C:\formula_student\fsg-dv-sim` | everything |
| WSL 2 with Ubuntu 24.04, Python 3.12 | Windows 11 | launcher, tools, tests, offline demo |
| ROS 2 Jazzy + rmw_zenoh | Ubuntu in WSL | lite simulator, the Isaac relay, dashboard, RViz |
| Isaac Sim 6.1 | Windows | rendered camera and RTX LiDAR |
| Windows Python with PyTorch (CUDA), Ultralytics | Windows | optional: GPU training (RL, YOLO) |
| MATLAB | Windows | optional: the Performance department's LapSim in MATLAB |

You can stop after step 4 and already have the lite simulator, the dashboard and all tools. Isaac Sim (step 5)
needs the GPU: 8 GB of GPU memory is the tested minimum (RTX 5060 Laptop, about 0.8 x real time).

Check at any time what is still missing: `python3 tools/check_install.py` (or the launcher: **Maintenance →
Check the installation**). It lists every part with the command that fixes it.

---

## 1. Windows: WSL 2 and Ubuntu 24.04 (10 min)

PowerShell **as administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart when asked, open **Ubuntu 24.04** from the start menu and choose a user name and password. Then:

```powershell
wsl --update
wsl -l -v          # Ubuntu-24.04 must show VERSION 2
```

* The NVIDIA driver on Windows should be a current one (Studio or Game Ready). WSL uses it too; nothing to
  install inside Ubuntu.
* `run.bat` expects the distribution name `Ubuntu-24.04`. Another name: change `DISTRO` at the top of `run.bat`.

## 2. Get the project (5 min)

Copy the whole project folder to **`C:\formula_student\fsg-dv-sim`** (from the team drive, a USB stick or the
team's Git). WSL runs it from there as `/mnt/c/formula_student/fsg-dv-sim`; there is no second copy.

These are in the folder but are **not** in a public repository; make sure they came along:

| Path | What | Without it |
|---|---|---|
| `lapsim/LR26_LapSim/` | the Performance department's LapSim (MATLAB) with `config/lr26_params.m` | nothing runs with the LR26 model |
| `assets/lr26/*.usda` | the LR26 body and wheels for Isaac Sim (from the CAD, confidential) | Isaac draws the simple box car |
| `cad/lr26/*.stp` | CAD source (confidential), only needed to regenerate the USD | nothing else |
| `models/cones/<name>/model.onnx` | trained cone detector (YOLO) | camera cones need a new training |
| `rl/policies/` | trained RL policies | the RL driver falls back to the reference driver |

## 3. Ubuntu: ROS 2 Jazzy and the packages (25 min)

In the Ubuntu terminal. First the ROS 2 package source (the official way, see
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html if a command changed):

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update && sudo apt upgrade -y
```

Then everything the simulator uses:

```bash
sudo apt install -y ros-jazzy-desktop ros-jazzy-rmw-zenoh-cpp ros-jazzy-ackermann-msgs \
                    python3-colcon-common-extensions python3-numpy python3-scipy python3-yaml \
                    python3-matplotlib python3-pip unzip
```

Do **not** add `source /opt/ros/jazzy/setup.bash` to `~/.bashrc`. The project has its own aliases that load
ROS only when needed:

```bash
bash /mnt/c/formula_student/fsg-dv-sim/tools/wsl/setup_shell.sh    # backs up ~/.bashrc first; --dry-run shows the change
source ~/.bashrc
fsgbuild            # builds the ROS 2 workspace (about a minute); build files go to ~/.fsg-dv-sim/colcon
```

| Alias | Does |
|---|---|
| `fsgsim` | loads ROS 2 Jazzy and the workspace, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, `ROS_DOMAIN_ID=18` |
| `fsgrouter` | starts the Zenoh router (the launcher does this for you) |
| `fsgbuild` | rebuilds the workspace |

Optional Python packages in WSL (the launcher has buttons for both):

```bash
python3 -m pip install --user --break-system-packages onnxruntime      # camera cones (YOLO) in the relay
python3 -m pip install --user --break-system-packages torch            # RL training in WSL (CPU, or GPU with the Windows driver)
```

## 4. First check without Isaac Sim (10 min)

```bash
cd /mnt/c/formula_student/fsg-dv-sim
python3 tools/check_install.py                 # "ready: tools and offline demo yes | ROS 2 simulators yes"
python3 -m unittest discover -s tests          # 346 tests, OK (5-13 minutes; a few skipped without PyTorch)
python3 tools/regression.py --quick            # 12 closed-loop cases, all ok (under a minute)
```

Then on Windows double-click **`C:\formula_student\fsg-dv-sim\run.bat`**. The browser opens
http://localhost:8070 (the launcher). Under **System** ROS 2, Zenoh and the workspace are green.

* **Offline demo** → Start: the dashboard opens on http://localhost:8080, the car drives.
* **Lite**, mission Skidpad → Start: RViz opens (through WSLg, built into Windows 11), the dashboard shows the
  run, finished in about 5.4 s with 0 cones.

**Maintenance → Put run.bat on the desktop** makes a shortcut.

## 5. Windows: Isaac Sim 6.1 (30 min)

1. Download **Isaac Sim 6.1** (the standalone Windows package) from NVIDIA's Isaac Sim download page and unpack
   it to **`C:\isaacsim`**, so that `C:\isaacsim\python.bat` exists. Follow NVIDIA's installation page for that
   version for its own first-start steps (for the 5.x and 6.x packages: run `post_install.bat` once).
   *This step was not repeated when this guide was written; the team PC has it installed this way.*
2. Start `C:\isaacsim\isaac-sim.bat` once by hand. The first start compiles shaders and takes several minutes.
   Close it again.
3. Another folder, or the Python of an Isaac Lab 3.0 environment: launcher → **Maintenance → Settings**.
4. Nothing ROS-related is installed on Windows. Isaac Sim talks to the relay in WSL over one TCP connection
   (port 47800). `pyyaml` is installed into Isaac's Python on the first start if it is missing.

Test: launcher → **Isaac Sim**, mission **Skidpad**, reference driver → Start. The steps appear one by one
(router, relay, Isaac Sim window after 20-60 s), then the dashboard opens. If Isaac does not connect, see the
troubleshooting table in [WINDOWS_WSL.md](WINDOWS_WSL.md) (firewall and networking mode) and check the link
alone with `tools/link_check.py` (section 3 there).

## 6. Optional parts

**GPU training on Windows (RL universal policy, cone detector).** Install Python for Windows (python.org, tick
"Add to PATH"), then in PowerShell install PyTorch with CUDA using the command from https://pytorch.org/get-started
for your CUDA version. In the launcher: **Maintenance → Install packages for GPU jobs**, **Install YOLO
(Ultralytics) for the Windows Python**, then **Check the GPU code (PyTorch tests)**. Details:
[LAUNCHER.md](LAUNCHER.md) (GPU jobs), [PERCEPTION.md](PERCEPTION.md).

**MATLAB.** Any recent MATLAB on Windows; no toolbox is needed for `run_lapsim`. The launcher and the
dashboard find it under `C:\Program Files\MATLAB`; otherwise set `FSG_MATLAB` to `matlab.exe`. See
[DASHBOARD.md](DASHBOARD.md) (MATLAB from WSL).

**CAD to USD** (only when the CAD changes): [CAD.md](CAD.md).

## 7. If something fails

| Symptom | Fix |
|---|---|
| `run.bat`: "Could not reach WSL" | `wsl -l -v`; install `Ubuntu-24.04` or change `DISTRO` in `run.bat` |
| Launcher: ROS 2 / Zenoh / workspace yellow | step 3; then **Maintenance → Build workspace** |
| `fsgbuild` fails after copying a new project version | `rm -rf ~/.fsg-dv-sim/colcon && fsgbuild` |
| RViz window does not open | `wsl --update` (WSLg); the simulation runs without it (untick RViz) |
| Isaac Sim starts but "relay did not connect" | [WINDOWS_WSL.md](WINDOWS_WSL.md) section 5; try `run_isaac.ps1 -RelayHost <WSL IP>` (`hostname -I` in WSL) |
| Isaac Sim: car is a grey box | `assets/lr26/*.usda` missing (step 2) |
| Isaac Sim very slow, GPU memory full | close other GPU programs; start headless; do not train on the GPU at the same time |
| Camera cones: "no cone detector yet" | copy `models/cones/` or train one ([PERCEPTION.md](PERCEPTION.md)) |
| Something else | launcher → **Share with Claude → Save report to the project**; the report and logs go to `shared\` |

## What was checked for this guide

Checked on Linux (10 Nov 2026): the Ubuntu package list (through a ROS 2 Jazzy build of the workspace), the unit
tests, the regression run, the installation check, the launcher page and the lite simulator. **Not** checked
here: the Windows steps (WSL install, `run.bat`, Isaac Sim 6.1 download and first start, Windows Python / CUDA,
MATLAB). They follow the setup of the team PC as described in [WINDOWS_WSL.md](WINDOWS_WSL.md) and
[LAUNCHER.md](LAUNCHER.md); please report any step that differs on a fresh machine.
