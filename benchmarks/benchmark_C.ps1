# Benchmark de la implementacion C (Julia, no convencional; competencia 2), como lo pide el
# manual (Parte 4) y con lo acordado:
#   1. Pruebas internas (--test).
#   2. Crossover: 10,000 arboles grow_full, profundidades 7, 9, 10, 12 y 15.
#   3. GP completo: poblacion 5,000, 100% grow_full, 50% crossover / 50% mutacion,
#      profundidades 7, 9 y 12, 100 generaciones, dataset oficial (datos\dataset3a.csv).
# Cada profundidad corre en un proceso aparte. Del GP se guarda el tiempo, el mejor MSE
# final (gp_C.csv) y el mejor MSE de cada generacion (gp_historial_C.csv).
# Antes de cronometrar, cada proceso hace una corrida chica de calentamiento: la compilacion
# JIT de Julia no se cuenta.
#
# Uso, desde PowerShell en C:\proyecto:
#   powershell -ExecutionPolicy Bypass -File .\benchmarks\benchmark_C.ps1
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
#   -SinPendientes           GP sin plegado de constantes, corte temprano, MSE reutilizado de
#                            hijos identicos a su padre ni evaluacion por lotes (--no-batch,
#                            que el profesor dijo que no es valida en C). Mismo resultado.
#   -ParesEnMemoria -1       -1 = 10,000 a la vez (C ocupa ~1.3 GB); 2500 = 5,000 a la vez
#   -Julia julia             ejecutable de Julia
#   -Data ruta\dataset.csv   dataset del GP completo

param(
    [string]$Julia = "julia",
    [string]$C = "GP_JL_C",
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

if (-not (Get-Command $Julia -ErrorAction SilentlyContinue)) { throw "No se encontro Julia ('$Julia'); instalalo o pasa -Julia" }
$main = Join-Path $C "main.jl"
if (-not (Test-Path $main)) { throw "No se encontro $main (pasa -C)" }
$flags = @("--startup-file=no", "-O3")
$extra = @()
$pendientes = "activos"
if ($SinPendientes) {
    $extra += "--no-fold", "--no-cut", "--no-dup", "--no-batch"
    $pendientes = "apagados (--no-fold --no-cut --no-dup --no-batch)"
}
$info = @(
    "Implementacion: C (Julia, no convencional, competencia 2: mas rapida no convencional)"
) + (Get-JuliaInfo $Julia $flags) + @(
    "Plegado, corte temprano y MSE de hijos identicos al padre (GP completo): $pendientes",
    "Cache de resultados del GP: automatico (--cache-mb=1024)"
)

Invoke-GPBenchmark -Impl "C" -Exe $Julia -Pre ($flags + (Resolve-Path $main).Path) -Data $Data `
    -Seeds $Seeds -Generations $Generations -GPDepths $GPDepths -CrossDepths $CrossDepths `
    -Variables $Variables -Population $Population -Trees $Trees -ParesEnMemoria $ParesEnMemoria `
    -GPExtra ($extra + $ExtraGP) -Info $info `
    -SoloCrossover:$SoloCrossover -SoloGP:$SoloGP
