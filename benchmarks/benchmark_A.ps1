# Benchmark de la implementacion A (Julia, convencional de alto nivel; no compite), como lo
# pide el manual (Parte 4) y con lo acordado:
#   1. Pruebas internas (--test).
#   2. Crossover: 10,000 arboles grow_full, profundidades 7, 9, 10, 12 y 15.
#   3. GP completo: poblacion 5,000, 100% grow_full, 50% crossover / 50% mutacion,
#      profundidades 7, 9 y 12, 100 generaciones, dataset oficial (datos\dataset3a.csv).
# Cada profundidad corre en un proceso aparte. Del GP se guarda el tiempo, el mejor MSE
# final (gp_A.csv) y el mejor MSE de cada generacion (gp_historial_A.csv).
# Antes de cronometrar, cada proceso hace una corrida chica de calentamiento: la compilacion
# JIT de Julia no se cuenta.
#
# Uso, desde PowerShell en C:\proyecto:
#   powershell -ExecutionPolicy Bypass -File .\benchmarks\benchmark_A.ps1
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
#   -ParesEnMemoria -1       -1 = segun la RAM; 0 = 10,000 a la vez; 2500 = 5,000 a la vez
#   -Julia julia             ejecutable de Julia
#   -Data ruta\dataset.csv   dataset del GP completo

param(
    [string]$Julia = "julia",
    [string]$A = "GP_JL_A",
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
    [switch]$SoloGP
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. (Join-Path $PSScriptRoot "comun.ps1")

if (-not (Get-Command $Julia -ErrorAction SilentlyContinue)) { throw "No se encontro Julia ('$Julia'); instalalo o pasa -Julia" }
$main = Join-Path $A "main.jl"
if (-not (Test-Path $main)) { throw "No se encontro $main (pasa -A)" }
$flags = @("--startup-file=no", "-O3")
$info = @(
    "Implementacion: A (Julia, convencional de alto nivel; no compite)"
) + (Get-JuliaInfo $Julia $flags)

Invoke-GPBenchmark -Impl "A" -Exe $Julia -Pre ($flags + (Resolve-Path $main).Path) -Data $Data `
    -Seeds $Seeds -Generations $Generations -GPDepths $GPDepths -CrossDepths $CrossDepths `
    -Variables $Variables -Population $Population -Trees $Trees -ParesEnMemoria $ParesEnMemoria `
    -GPExtra $ExtraGP -Info $info `
    -SoloCrossover:$SoloCrossover -SoloGP:$SoloGP
