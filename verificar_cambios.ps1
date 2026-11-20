# Verifica los cambios en C (plegado de constantes y correccion de NO_CACHE) y mide
# la diferencia contra la version anterior guardada en respaldo_20261119_2010.
#
# Uso, desde PowerShell en C:\proyecto:
#   powershell -ExecutionPolicy Bypass -File .\verificar_cambios.ps1
#   powershell -ExecutionPolicy Bypass -File .\verificar_cambios.ps1 -SinMedir   (solo pruebas)

param(
    [string]$Julia = "julia",
    [switch]$SinMedir
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$flags = @("--startup-file=no", "-O3")

Write-Host "`n[1] Pruebas de C"
& $Julia @flags "GP_JL_C\main.jl" --test
if ($LASTEXITCODE -ne 0) { throw "Fallan las pruebas de C" }

Write-Host "`n[2] Verificacion cruzada A <-> C"
& $Julia @flags "GP_JL_C\tools\cross_check.jl"
if ($LASTEXITCODE -ne 0) { throw "cross_check encontro diferencias entre A y C" }

Write-Host "`n[3] Verificacion A, C <-> B"
$gpb = "GP_CPP20_B\out\build\x64-release\gp_b.exe"
if (Test-Path $gpb) {
    & $Julia @flags "GP_JL_C\tools\verificar_con_b.jl" (Resolve-Path $gpb).Path
    if ($LASTEXITCODE -ne 0) { throw "verificar_con_b encontro diferencias" }
} else {
    Write-Host "  (no se encontro $gpb; compila B en x64-Release para esta prueba)"
}

if ($SinMedir) { Write-Host "`nPruebas correctas."; exit 0 }

Write-Host "`n[4] GP completo de C: version anterior (respaldo) contra la actual"
Write-Host "    El best_mse debe ser IGUAL en las dos; solo debe cambiar el tiempo."
$data = (Resolve-Path "GP_JL_C\data\ejemplo.csv").Path
$versions = @(
    @{ Name = "antes"; Main = "respaldo_20261119_2010\GP_JL_C\main.jl" },
    @{ Name = "ahora"; Main = "GP_JL_C\main.jl" }
)
foreach ($v in $versions) {
    Write-Host "== C $($v.Name)"
    & $Julia @flags $v.Main --bench-gp "--data=$data" --population=5000 --depths=7,9,12 --generations=10 |
        ForEach-Object { ($_ -split ',')[0..4] -join ',' }
    if ($LASTEXITCODE -ne 0) { throw "Fallo el benchmark de C $($v.Name)" }
}
Write-Host "`nListo."
