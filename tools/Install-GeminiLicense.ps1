[CmdletBinding()]
param(
    [string]$Source = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $InstallerDir
$LicenseDir = Join-Path $ProjectRoot "resources\license"
$Destination = Join-Path $LicenseDir "ChingmuGemini.license"

if ([string]::IsNullOrWhiteSpace($Source)) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Select a Chingmu Gemini license file"
    $dialog.Filter = "Chingmu Gemini license (*.license)|*.license|All files (*.*)|*.*"
    $dialog.Multiselect = $false
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        Write-Host "No license file selected."
        exit 2
    }
    $Source = $dialog.FileName
}

$sourceFull = [System.IO.Path]::GetFullPath($Source)
if (-not (Test-Path -LiteralPath $sourceFull -PathType Leaf)) {
    throw "License file not found: $sourceFull"
}

$document = Get-Content -LiteralPath $sourceFull -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace([string]$document.payload) -or
    [string]::IsNullOrWhiteSpace([string]$document.signature)) {
    throw "The selected file is not a Chingmu Gemini signed-license envelope."
}

New-Item -ItemType Directory -Path $LicenseDir -Force | Out-Null
$temporary = "$Destination.new"
Copy-Item -LiteralPath $sourceFull -Destination $temporary -Force
Move-Item -LiteralPath $temporary -Destination $Destination -Force

Write-Host "Installed: $Destination" -ForegroundColor Green
Write-Host "The license signature and online time will be verified before a new task starts."
