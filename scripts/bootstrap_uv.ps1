$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = Split-Path $PSScriptRoot -Parent
$uvExe = Join-Path $projectRoot 'tools\uv\uv.exe'
try {
    if (-not (Test-Path -LiteralPath $uvExe)) {
        Write-Host 'Preparing uv...'
        $downloadRoot = Join-Path $projectRoot 'tools\downloads'
        New-Item -ItemType Directory -Force $downloadRoot | Out-Null
        $archive = Join-Path $downloadRoot 'uv-windows.zip'
        Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.11.17/uv-x86_64-pc-windows-msvc.zip' -OutFile $archive -UseBasicParsing
        Expand-Archive -LiteralPath $archive -DestinationPath (Split-Path $uvExe -Parent) -Force
    }
    $env:PYTHONUTF8 = '1'
    & $uvExe run --no-project --python 3.11 (Join-Path $PSScriptRoot 'windows_setup.py') --uv $uvExe
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message
    exit 1
}
