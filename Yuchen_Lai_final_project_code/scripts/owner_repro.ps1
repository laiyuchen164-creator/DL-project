param(
    [switch]$Run25Percent,
    [switch]$Run10Percent
)

# Personalized helper script for Yuchen Lai
# Packaging emphasis: experiment scripts, documentation, and figure regeneration.

$root = Split-Path -Parent $PSScriptRoot

if ($Run25Percent) {
    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_proposal_formal_f025.ps1')
}

if ($Run10Percent) {
    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_proposal_formal_f010.ps1')
}
