<#
.SYNOPSIS
  Start the FSG driverless simulator in Isaac Sim on Windows and connect it to the
  ROS 2 relay running in WSL2.

.EXAMPLE
  # standalone Isaac Sim 6.1 in C:\isaacsim (default), relay IP detected from WSL
  .\isaac\run_isaac.ps1 -Mission skidpad

.EXAMPLE
  # Python from an Isaac Lab 3.0 environment (pip-installed isaacsim)
  .\isaac\run_isaac.ps1 -Python C:\IsaacLab\.venv\Scripts\python.exe -Mission trackdrive

.EXAMPLE
  # extra options are passed straight to run_isaac.py
  .\isaac\run_isaac.ps1 -Mission autocross -Headless -Extra "--camera-streams","left,right,depth","--seed","11"

.EXAMPLE
  # change vehicle parameters for this run (LapSim / Performance names)
  .\isaac\run_isaac.ps1 -Mission autocross -Set "m=300","c_z=4.5"

.NOTES
  In WSL first:  fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd       (Zenoh router, terminal 1)
                 fsgsim && ros2 launch fsd_sim_bringup isaac.launch.py  (relay, terminal 2)
  Details and troubleshooting:    docs/WINDOWS_WSL.md
#>
[CmdletBinding()]
param(
    [string]$Mission = "",
    [string]$IsaacPath = $(if ($env:ISAAC_SIM) { $env:ISAAC_SIM } else { "C:\isaacsim" }),
    [string]$Python = "",
    [string]$RelayHost = "",
    [string]$Distro = "",
    [int]$Port = 47800,
    [ValidateSet("tcp", "ros2")][string]$SensorTransport = "tcp",
    [int]$DomainId = $(if ($env:FSG_ROS_DOMAIN_ID) { [int]$env:FSG_ROS_DOMAIN_ID } else { 18 }),
    [string]$ZenohRouter = "127.0.0.1:7447",
    [string]$ZenohConfig = "C:\robots\zenoh_session_isaac.json5",
    [string[]]$Set = @(),
    [string]$Faults = "",
    [switch]$Headless,
    [string[]]$Extra = @()
)
# "Continue": native tools (wsl.exe, python) write to stderr, which Windows PowerShell 5.1
# would otherwise turn into terminating errors. Fatal conditions use explicit `throw`.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot "run_isaac.py"

# ---------------------------------------------------------------- Python ----
if (-not $Python) {
    $bat = Join-Path $IsaacPath "python.bat"
    if (-not (Test-Path $bat)) {
        throw "No python.bat in '$IsaacPath'. Pass -IsaacPath <Isaac Sim folder> or -Python <python.exe with isaacsim installed>."
    }
    $Python = $bat
}
Write-Host "[fsd] Isaac Python : $Python"

# ------------------------------------------------------------ relay host ----
function Test-Relay([string]$h, [int]$p) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $ok = $c.ConnectAsync($h, $p).Wait(1500)
        $c.Close()
        return $ok
    } catch { return $false }
}
if (-not $RelayHost) {
    # NAT mode: the WSL IP is the fast direct path; localhost also works (WSL forwarding).
    # Mirrored mode: localhost is the reliable choice. Probe both, prefer the WSL IP.
    $wslArgs = @()
    if ($Distro) { $wslArgs += @("-d", $Distro) }
    $wslIp = ""
    try {
        $ips = (& wsl.exe @wslArgs hostname -I 2>&1 | Out-String)
        $wslIp = ($ips -split "\s+" | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
    } catch { $wslIp = "" }
    # Mirrored networking: WSL shares the Windows IPs. Then the loopback path is the direct one; the LAN IP
    # carried only 35-80 MB/s with pauses of 0.3-0.6 s in the run on the team PC (11 Nov 2026).
    $order = @($wslIp, "127.0.0.1")
    try {
        $own = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | ForEach-Object { $_.IPAddress })
        if ($wslIp -and ($own -contains $wslIp)) { $order = @("127.0.0.1", $wslIp) }
    } catch { }
    foreach ($candidate in $order) {
        if ($candidate -and (Test-Relay $candidate $Port)) { $RelayHost = $candidate; break }
    }
    if (-not $RelayHost) {
        $RelayHost = "127.0.0.1"
        Write-Warning "Relay not reachable yet (WSL IP '$wslIp', localhost). Start 'ros2 launch fsd_sim_bringup isaac.launch.py' in WSL; Isaac keeps retrying on 127.0.0.1."
    }
}
Write-Host "[fsd] ROS 2 relay  : tcp://${RelayHost}:$Port"

# -------------------------------------------------- ROS 2 bridge (optional) ----
if ($SensorTransport -eq "ros2") {
    # Isaac publishes LiDAR / cameras / clock itself with its bundled ROS 2 Jazzy over Zenoh,
    # exactly like the Go2 setup (windows/run_go2_sim_ros.bat): the Zenoh router runs in WSL
    # (fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd) and Isaac connects to it on 127.0.0.1:7447.
    $rh, $rp = $ZenohRouter.Split(":")
    if (-not (Test-Relay $rh ([int]$rp))) {
        throw "No Zenoh router on tcp/$ZenohRouter. In WSL run:  fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd   then start this again."
    }
    Write-Host "[fsd] Zenoh router : tcp/$ZenohRouter"
    if (-not (Test-Path $ZenohConfig)) {
        throw "Zenoh session config '$ZenohConfig' not found. Use the one from the Go2 setup (tcp/127.0.0.1 for connect and listen) or pass -ZenohConfig <file>."
    }
    foreach ($v in @("ROS_DISTRO", "RMW_IMPLEMENTATION", "FASTRTPS_DEFAULT_PROFILES_FILE")) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    $env:ROS_DOMAIN_ID = "$DomainId"
    $env:ZENOH_SESSION_CONFIG_URI = $ZenohConfig
    $env:ZENOH_ROUTER_CHECK_ATTEMPTS = "10"
    $setupBat = Join-Path $IsaacPath "setup_ros_env.bat"
    if (Test-Path $setupBat) {
        # import the variables Isaac's own script sets (PATH to the bundled Jazzy libraries, ROS_DISTRO, ...)
        cmd /c "`"$setupBat`" >nul 2>&1 && set" | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
        }
    } else {
        Write-Warning "$setupBat not found; using the bundled libraries directly"
        $env:ROS_DISTRO = "jazzy"
        $lib = Get-ChildItem -Path (Join-Path $IsaacPath "exts") -Directory -Filter "isaacsim.ros2.*" -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName "jazzy\lib" } | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($lib) { $env:PATH = "$lib;$env:PATH" }
    }
    $env:RMW_IMPLEMENTATION = "rmw_zenoh_cpp"
    Write-Host "[fsd] ROS 2        : ROS_DISTRO=$env:ROS_DISTRO RMW=$env:RMW_IMPLEMENTATION ROS_DOMAIN_ID=$env:ROS_DOMAIN_ID"
}

# ------------------------------------------------------------- pyyaml ----
& $Python -c "import yaml, numpy" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[fsd] installing pyyaml into Isaac's Python"
    & $Python -m pip install pyyaml
}

# ---------------------------------------------------------------- run ----
$runArgs = @($Script, "--relay-host", $RelayHost, "--relay-port", $Port, "--sensor-transport", $SensorTransport)
if ($Mission) { $runArgs += @("--mission", $Mission) }
if ($Headless) { $runArgs += "--headless" }
foreach ($kv in $Set) { $runArgs += @("--set", $kv) }   # vehicle parameter overrides, e.g. -Set "m=300","mux=1.5"
if ($Faults) { $runArgs += @("--faults", $Faults) }     # sensor fault injection (fsdsim.faults)
$runArgs += $Extra
Write-Host "[fsd] $Python $($runArgs -join ' ')"
# cmd.exe (python.bat) cannot use a \\wsl.localhost UNC folder as working directory; all
# paths passed to run_isaac.py are absolute, so start from a local folder instead.
$workDir = if ($Root.StartsWith("\\")) { $env:TEMP } else { $Root }
Push-Location $workDir
try { & $Python @runArgs } finally { Pop-Location }
