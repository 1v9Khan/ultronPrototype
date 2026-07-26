<#
.SYNOPSIS
    Install the remote PTT agent as a logon-start Scheduled Task. RUN THIS ON
    THE GAME PC, once.

.DESCRIPTION
    Makes the agent come up by itself so Ultron never has to reach across the
    LAN to start it.

    WHY NOT HAVE ULTRON START IT REMOTELY (the obvious alternative):
    every mechanism for launching a process on another Windows box -- PsExec,
    WMI Win32_Process.Create, WinRM/Invoke-Command, schtasks /s, SSH -- is
    remote code execution against a machine running Vanguard's kernel-level
    anticheat, and several are textbook lateral-movement techniques that
    security tooling flags on sight. The agent is deliberately the most boring
    process on this machine (stdlib + hidapi, HID output reports only, no
    synthetic input, no game introspection); spawning it over the network
    would hand back exactly the risk that design avoids. A local logon task
    has ZERO remote-execution surface and is up BEFORE Ultron, so it also
    removes the start-ordering problem entirely.

    Runs UNELEVATED on purpose: writing HID output reports to a USB peripheral
    needs no admin rights, and an unprivileged process is the smaller
    footprint. The task is bound to your interactive logon because the HID
    device belongs to that session.

.PARAMETER UltronIp
    LAN address of the PC running Ultron. Restricts the agent to datagrams
    from that peer (on top of the HMAC token).

.PARAMETER RepoPath
    Path to the ultronPrototype checkout ON THIS (game) machine.

.PARAMETER Python
    Python to run the agent with. Defaults to pythonw.exe on PATH so no
    console window appears at logon.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\ptt_agent_install_task.ps1 -UltronIp 10.0.0.3 -RepoPath C:\ultronPrototype

.NOTES
    The shared token is read from the KENNING_PTT_NETWORK_TOKEN environment
    variable at RUN time -- it is never written into the task definition, so
    the secret does not land in the Task Scheduler XML on disk. Set it once,
    User scope, on this machine:
        [Environment]::SetEnvironmentVariable('KENNING_PTT_NETWORK_TOKEN','<token>','User')

    Verify afterwards:   Get-ScheduledTask -TaskName 'Kenning PTT Agent'
    Start it now:        Start-ScheduledTask -TaskName 'Kenning PTT Agent'
    Remove it:           Unregister-ScheduledTask -TaskName 'Kenning PTT Agent' -Confirm:$false
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$UltronIp,
    [Parameter(Mandatory = $true)][string]$RepoPath,
    [string]$Python = "pythonw.exe",
    [int]$Port = 8778,
    [string]$TaskName = "Kenning PTT Agent"
)

$ErrorActionPreference = "Stop"

# DOUBLE-REGISTRATION GUARD. The game PC was set up 2026-07-26 under the task
# name 'UltronPTTAgent' by a different route; registering a second task would
# start a SECOND agent, and the loser of the race for udp/8778 dies while the
# winner may hold the dongle -- an intermittent, miserable failure. Refuse if
# ANY existing task already launches ptt_agent.py, whatever it is called.
$existing = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
    $_.TaskName -ne $TaskName -and
    ($_.Actions | Where-Object { $_.Arguments -match 'ptt_agent\.py' })
}
if ($existing) {
    $names = ($existing | ForEach-Object { $_.TaskName }) -join ', '
    throw ("A scheduled task already starts the PTT agent: $names. " +
           "Only one agent may own udp/$Port. Update that task instead, or " +
           "remove it first: Unregister-ScheduledTask -TaskName '<name>'.")
}

$agent = Join-Path $RepoPath "scripts\ptt_agent.py"
if (-not (Test-Path $agent)) {
    throw "Agent script not found at $agent -- check -RepoPath."
}

# Resolve the interpreter to an absolute path: Task Scheduler does not inherit
# an interactive PATH, so a bare 'pythonw.exe' can silently fail to launch.
$pyResolved = (Get-Command $Python -ErrorAction SilentlyContinue)?.Source
if (-not $pyResolved) {
    throw "Could not resolve '$Python' on PATH. Pass -Python with a full path (e.g. C:\Python311\pythonw.exe)."
}

if (-not [Environment]::GetEnvironmentVariable('KENNING_PTT_NETWORK_TOKEN', 'User')) {
    Write-Warning ("KENNING_PTT_NETWORK_TOKEN is not set in the User environment on this machine. " +
                   "The agent will refuse unauthenticated datagrams and PTT will stay inert until you set it.")
}

$argline = "`"$agent`" --port $Port --allow-peer $UltronIp"

$action = New-ScheduledTaskAction -Execute $pyResolved -Argument $argline -WorkingDirectory $RepoPath
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# Unelevated, interactive session (the HID device lives in that session).
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited
# Survive a crash; never let Windows stop it for running "too long"; do not
# fight battery/idle policies -- this is a long-lived listener.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered '$TaskName'" -ForegroundColor Green
Write-Host "  interpreter : $pyResolved"
Write-Host "  agent       : $agent"
Write-Host "  listening   : udp/$Port, peer-restricted to $UltronIp"
Write-Host ""
Write-Host "Start it now without logging out:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Then confirm from the Ultron PC that the agent answers before starting Ultron."
