# Benchmark de la implementacion B (C++20, convencional; competencia 1), como lo pide el
# manual (Parte 4) y con lo acordado:
#   1. Pruebas internas (--test).
#   2. Crossover: 10,000 arboles grow_full, profundidades 7, 9, 10, 12 y 15.
#   3. GP completo: poblacion 5,000, 100% grow_full, 50% crossover / 50% mutacion,
#      profundidades 7, 9 y 12, 100 generaciones, dataset oficial (datos\dataset3a.csv).
# Cada profundidad corre en un proceso aparte. Del GP se guarda el tiempo, el mejor MSE
# final (gp_B.csv) y el mejor MSE de cada generacion (gp_historial_B.csv).
#
# Uso, desde PowerShell en C:\proyecto, con B compilado en x64-Release:
#   powershell -ExecutionPolicy Bypass -File .\benchmarks\benchmark_B.ps1
#
# Opciones (todas tienen valor por defecto):
#   -Seeds 5                 semillas (por defecto 5; por ejemplo -Seeds 1,5)
#   -Population 5000         poblacion del GP completo (multiplo de 4)
#   -Trees 10000             arboles del benchmark de crossover (par)
#   -GPDepths 7,9,12         profundidades del GP completo
#   -CrossDepths 7,9,10,12,15  profundidades del benchmark de crossover
#   -ExtraGP "--opcion"      opciones extra para el GP completo (por ejemplo "--no-dup")
#   -Generations 100         generaciones del GP completo
#   -SoloCrossover / -SoloGP correr solo una parte
#   -SinPendientes           GP sin plegado de constantes, corte temprano ni reutilizar el MSE
#                            de hijos identicos a su padre (mismo resultado)
#   -ParesEnMemoria -1       -1 = segun la RAM; 0 = 10,000 a la vez; 2500 = 5,000 a la vez
#   -B ruta\gp_b.exe         ejecutable de B
#   -Data ruta\dataset.csv   dataset del GP completo

param(
    [string]$B = "GP_CPP20_B\out\build\x64-Release\gp_b.exe",
    [string]$Data = "datos\dataset3a.csv",
    [int[]]$Seeds = @(5),
    [int]$Generations = 100,
    [int[]]$GPDepths = @(7, 9, 12),
    [int[]]$CrossDepths = @(7, 9, 10, 12, 15),
    [int]$Variables = 2,
    [int]$Population = 5000,
    [int]$Trees = 10000,
    [int]$ParesEnMemoria = -1,
    [string[]]$ExtraGP = @(),
    [switch]$SoloCrossover,
    [switch]$SoloGP,
    [switch]$SinPendientes
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. (Join-Path $PSScriptRoot "comun.ps1")

if (-not (Test-Path $B)) { throw "No se encontro el ejecutable de B: $B (compila en x64-Release o pasa -B)" }
$exe = (Resolve-Path $B).Path
$extra = @()
$pendientes = "activos"
if ($SinPendientes) {
    $extra += "--no-fold", "--no-cut", "--no-dup"
    $pendientes = "apagados (--no-fold --no-cut --no-dup)"
}
$info = @(
    "Implementacion: B (C++20, convencional, competencia 1: mas rapida convencional)",
    "Ejecutable: $exe (compilado: $((Get-Item $exe).LastWriteTime.ToString('yyyy-MM-dd HH:mm')))",
    "Plegado, corte temprano y MSE de hijos identicos al padre (GP completo): $pendientes"
)

Invoke-GPBenchmark -Impl "B" -Exe $exe -Pre @() -Data $Data -Seeds $Seeds `
    -Generations $Generations -GPDepths $GPDepths -CrossDepths $CrossDepths `
    -Variables $Variables -Population $Population -Trees $Trees -ParesEnMemoria $ParesEnMemoria `
    -GPExtra ($extra + $ExtraGP) -Info $info `
    -SoloCrossover:$SoloCrossover -SoloGP:$SoloGP
