#requires -Version 5.1
<#
.SYNOPSIS
    Diagnostic for `smos service install` on Windows.

.DESCRIPTION
    Reproduces exactly what smos does when it shells out to sc.exe via
    Rust's std::process::Command, and reports what sc.exe returns
    (exit code, stdout, stderr) at every step.

    The core symptom under investigation:
        Error: sc create failed:
        Error: sc ["start", "smos"] failed:
    i.e. sc.exe exits non-zero but stdout AND stderr are EMPTY, so smos
    has nothing to put after the colon. This script checks whether that
    is sc.exe's own behaviour (a Windows/encoding/WOW64 issue) or
    something smos-specific.

    Safe to run: uses a unique test service name and removes it in a
    finally block. Does NOT touch the real 'smos' service.

.NOTES
    Run from an ELEVATED (Administrator) PowerShell so the create/start
    path can be exercised. If not elevated, Part B (create/start/delete)
    is skipped but Part A (environment + capture behaviour) still runs.

    Usage:
        pwsh -File .\diag-service-install.ps1
    (or right-click PowerShell 7 -> Run as Administrator, then run the
    file). A full report is written next to the script as
    diag-service-install.report.txt - please share that file back.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$testService = "smos_diag_test_$PID"
$report = [System.Collections.Generic.List[string]]::new()

function Write-Section($t) {
    $line = "`n========== $t =========="
    Write-Host $line -ForegroundColor Cyan
    $report.Add($line)
}

function Write-KV($k, $v) {
    $line = ("  {0,-34}: {1}" -f $k, $v)
    Write-Host $line
    $report.Add($line)
}

function Write-Diag($l) {
    Write-Host $l
    $report.Add($l)
}

# ---------------------------------------------------------------------------
# Mirrors of the Rust helpers in windows_helpers.rs, so the command line
# we hand to sc.exe is byte-identical to what smos produces.
# ---------------------------------------------------------------------------

# Inverse of CommandLineToArgvW: returns the literal string to splice into
# a command line so that CommandLineToArgvW parses it back as a single
# token equal to $s. Equivalent to smos::quote_for_argv + Command::raw_arg.
function ConvertTo-ArgvToken([string]$s) {
    $sb = [System.Text.StringBuilder]::new($s.Length + 2)
    [void]$sb.Append('"')
    $bs = 0
    foreach ($c in $s.ToCharArray()) {
        switch ($c) {
            '\' { $bs++ }
            '"' {
                for ($i = 0; $i -lt (2 * $bs + 1); $i++) { [void]$sb.Append('\') }
                [void]$sb.Append('"')
                $bs = 0
            }
            default {
                for ($i = 0; $i -lt $bs; $i++) { [void]$sb.Append('\') }
                $bs = 0
                [void]$sb.Append($c)
            }
        }
    }
    for ($i = 0; $i -lt (2 * $bs); $i++) { [void]$sb.Append('\') }
    [void]$sb.Append('"')
    return $sb.ToString()
}

# Approximation of Rust std's Command::arg quoting on Windows: wrap in
# double quotes iff the token contains a space or is empty. Tokens in
# this script never contain embedded quotes, so the simple rule matches
# Rust's output exactly for our inputs.
function ConvertTo-StdArg([string]$s) {
    if ($s.Length -eq 0 -or $s.Contains(' ')) { return '"' + $s + '"' }
    return $s
}

# Spawn $exe with command line $cmdLine via CreateProcessW (no shell),
# capturing stdout / stderr / exit code separately. This is exactly what
# Rust's Command::output() does.
function Invoke-NativeCapture([string]$exe, [string]$cmdLine) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $exe
    $psi.Arguments = $cmdLine
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::new()
    $p.StartInfo = $psi
    [void]$p.Start()
    # Read async on both pipes to avoid the classic deadlock when the
    # child fills one pipe's buffer while we block on the other.
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    return [pscustomobject]@{
        ExitCode = $p.ExitCode
        StdOut   = $outTask.Result
        StdErr   = $errTask.Result
    }
}

function Show-Capture([string]$label, $r) {
    Write-Diag "  -- $label --"
    Write-Diag ("    ExitCode: {0}" -f $r.ExitCode)
    Write-Diag ("    StdOut  : [{0}]" -f $r.StdOut)
    Write-Diag ("    StdErr  : [{0}]" -f $r.StdErr)
}

function Test-IsAdmin {
    $g = & whoami /groups 2>$null
    return [bool]($g -match 'S-1-16-12288')
}

# ---------------------------------------------------------------------------
# PART A - environment + sc.exe capture behaviour (no admin needed)
# ---------------------------------------------------------------------------

Write-Section 'PART A: environment'
Write-KV 'PSVersion'          $PSVersionTable.PSVersion.ToString()
Write-KV 'OS'                 [System.Environment]::OSVersion.VersionString
Write-KV 'PROCESSOR_ARCHITECTURE' $env:PROCESSOR_ARCHITECTURE
Write-KV 'Process is 64-bit'  ([System.IntPtr]::Size -eq 8)
Write-KV 'IsAdmin (S-1-16-12288)' (Test-IsAdmin)

$smos = Get-Command smos -ErrorAction SilentlyContinue | Select-Object -First 1
$smosExe = $smos.Source
Write-KV 'smos resolved to'   $smosExe
Write-KV 'where smos'         (((where.exe smos 2>$null)) -join ' ; ')

# Detect 32-bit smos on 64-bit Windows. A 32-bit process sees SysWOW64
# instead of System32, where sc.exe does NOT exist - that is a prime
# suspect for "sc.exe silently fails with no output".
if ($smosExe -and (Test-Path -LiteralPath $smosExe)) {
    try {
        $fs = [System.IO.File]::OpenRead($smosExe)
        $br = [System.IO.BinaryReader]::new($fs)
        $fs.Seek(0x3C, 'Begin') | Out-Null
        $peOff = $br.ReadInt32()
        $fs.Seek($peOff + 4, 'Begin') | Out-Null
        $mach = $br.ReadUInt16()
        $fs.Close()
        $arch = switch ($mach) {
            0x8664 { 'x64' }
            0x014C { 'x86 (32-BIT - likely root cause!)' }
            0xAA64 { 'arm64' }
            default { "0x{0:X4}" -f $mach }
        }
        Write-KV 'smos PE machine' $arch
    } catch {
        Write-KV 'smos PE machine' ("read failed: {0}" -f $_.Exception.Message)
    }
} else {
    Write-KV 'smos PE machine' 'n/a (smos not found)'
}

Write-KV 'where sc'           (((where.exe sc 2>$null)) -join ' ; ')
Write-KV 'where sc.exe'       (((where.exe sc.exe 2>$null)) -join ' ; ')
$scExe = 'C:\Windows\System32\sc.exe'
Write-KV 'sc.exe path used'   $scExe
Write-KV 'sc.exe exists (64)' (Test-Path -LiteralPath $scExe)
Write-KV 'sc.exe exists (32)' (Test-Path -LiteralPath 'C:\Windows\SysWOW64\sc.exe')

Write-Section 'PART A: does `sc` (no path, no .exe) resolve to the right exe?'
# smos calls Rust Command::new("sc"), which on Windows becomes
# CreateProcessW(lpApplicationName=NULL, lpCommandLine="sc ..."). Per
# MSDN the search order for that case is:
#   1. directory the calling process was loaded from (== smos.exe dir)
#   2. current directory of the calling process
#   3. System32 (or SysWOW64 for a 32-bit caller)
#   4. Windows dir
#   5. directories listed in PATH
# If ANY of those contains an sc.bat / sc.cmd / sc.ps1 / sc.com / sc.exe
# that is not the real System32\sc.exe, it will be launched instead and
# likely exit non-zero with no output - reproducing the bug. `where.exe`
# only checks PATH and is NOT sufficient to rule this out.
$exts = @('.exe','.com','.bat','.cmd','.ps1','.vbs')
$searchRoots = [System.Collections.Generic.List[string]]::new()
if ($smosExe) { $searchRoots.Add((Split-Path -Parent $smosExe)) }
$searchRoots.Add($PWD.Path)
$searchRoots.Add('C:\Users\redmi')
foreach ($p in ($env:PATH -split ';')) { if ($p) { $searchRoots.Add($p.Trim('"')) } }
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$conflicts = [System.Collections.Generic.List[string]]::new()
foreach ($root in $searchRoots) {
    if (-not $root) { continue }
    if (-not $seen.Add($root)) { continue }
    foreach ($e in $exts) {
        $cand = Join-Path $root ('sc' + $e)
        if (Test-Path -LiteralPath $cand) {
            $real = (Resolve-Path -LiteralPath $cand).Path
            $conflicts.Add($real)
        }
    }
}
if ($conflicts.Count -eq 0) {
    Write-Diag '  no sc.* found in any search root'
} else {
    Write-Diag '  sc.* candidates CreateProcessW may pick (topmost wins):'
    foreach ($c in $conflicts) {
        $tag = if ($c -eq 'C:\Windows\System32\sc.exe') { '  (the REAL sc.exe)' } else { '  <-- NOT System32 sc.exe!' }
        Write-Diag ("    {0}{1}" -f $c, $tag)
    }
}
# Reproduce EXACTLY what smos does: spawn by bare name "sc" (no .exe,
# no path) and see what happens. If this returns an empty stdout on a
# non-zero exit code, we have reproduced the bug.
$rBare = Invoke-NativeCapture 'sc' 'query __smos_diag_nonexistent__'
Show-Capture 'bare "sc" query nonexistent (== smos call)' $rBare

Write-Section 'PART A: sc.exe (full path) capture behaviour (nonexistent service)'
# Control: the same call but via the absolute path. This isolates
# "sc.exe misbehaves" from "`sc` resolves to something else".
$r = Invoke-NativeCapture $scExe 'query __smos_diag_nonexistent__'
Show-Capture 'sc.exe query nonexistent' $r

$r = Invoke-NativeCapture $scExe 'start __smos_diag_nonexistent__'
Show-Capture 'sc.exe start nonexistent' $r

# ---------------------------------------------------------------------------
# PART B - real create/start/delete cycle (admin required)
# ---------------------------------------------------------------------------

Write-Section 'PART B: real create / start / delete cycle'
if (-not (Test-IsAdmin)) {
    Write-Diag '  SKIPPED: not running elevated. Re-run from an Administrator PowerShell.'
} elseif (-not $smosExe) {
    Write-Diag '  SKIPPED: smos.exe not found on PATH.'
} else {
    # Build the logical binPath value the same way smos does:
    #   "<binary>" serve --config "<config>"
    $binDir  = Split-Path -Parent $smosExe
    $cfgPath = Join-Path $binDir 'smos.toml'
    $binValue = '"' + $smosExe + '" serve --config "' + $cfgPath + '"'
    $rawArg   = ConvertTo-ArgvToken $binValue

    # Reproduce the exact CreateProcessW command line Rust builds for
    # `sc create` (see create_service() in windows.rs):
    #   create <name> binPath= <raw_arg> DisplayName= <quoted> start= auto
    $cmdCreate = (@(
        (ConvertTo-StdArg 'create'),
        (ConvertTo-StdArg $testService),
        (ConvertTo-StdArg 'binPath='),
        $rawArg,
        (ConvertTo-StdArg 'DisplayName='),
        (ConvertTo-StdArg 'SMOS Semantic Memory OS'),
        (ConvertTo-StdArg 'start='),
        (ConvertTo-StdArg 'auto')
    ) -join ' ')

    Write-KV 'test service'        $testService
    Write-KV 'logical binPath'     $binValue
    Write-KV 'raw_arg (quoted)'    $rawArg
    Write-KV 'config exists'       (Test-Path -LiteralPath $cfgPath)
    Write-KV 'cmdline to sc.exe'   $cmdCreate

    try {
        $r = Invoke-NativeCapture $scExe $cmdCreate
        Show-Capture 'sc create' $r

        if ($r.ExitCode -eq 0) {
            $cmdStart = (ConvertTo-StdArg 'start') + ' ' + (ConvertTo-StdArg $testService)
            $r = Invoke-NativeCapture $scExe $cmdStart
            Show-Capture 'sc start' $r
        } else {
            Write-Diag '  (skipping start because create returned non-zero)'
        }
    } finally {
        $cmdDelete = (ConvertTo-StdArg 'delete') + ' ' + (ConvertTo-StdArg $testService)
        $rd = Invoke-NativeCapture $scExe $cmdDelete
        Show-Capture 'sc delete (cleanup)' $rd
    }
}

# ---------------------------------------------------------------------------
# Save full report next to the script.
# ---------------------------------------------------------------------------

Write-Section 'DONE'
$outPath = Join-Path $PSScriptRoot 'diag-service-install.report.txt'
($report -join "`r`n") | Set-Content -LiteralPath $outPath -Encoding UTF8
Write-Host "  Report saved to: $outPath" -ForegroundColor Green
Write-Host "  Please share diag-service-install.report.txt (or paste the full console output)."
