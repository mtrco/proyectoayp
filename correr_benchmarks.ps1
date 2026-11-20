# Benchmarks del proyecto GP: B (C++) y C (Julia) con el mismo protocolo; A (Julia) opcional.
#
# Coloca este archivo en la carpeta que contiene GP_CPP20_B, GP_JL_A y GP_JL_C,
# compila B en x64-Release, instala Julia y ejecuta desde PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File .\correr_benchmarks.ps1
#
# Opciones (todas tienen valor por defecto):
#   -Data "ruta\dataset.csv"   dataset del benchmark del GP completo
#   -Seeds 1,2,3,4,5           semillas (una corrida por semilla)
#   -Generations 10            el manual NO lo fija: acordarlo con el profesor
#   -Variables 2               variables para generar arboles en el benchmark de crossover
#   -SoloCrossover / -SoloGP   correr solo una parte
#   -IncluirA                  medir tambien A (no compite; tarda mas)
#   -ParesEnMemoria -1         parejas de padres en memoria a la vez en el benchmark de crossover.
#                              0 = los 10,000 arboles a la vez (protocolo acordado con el profesor);
#                              2500 = 5,000 a la vez (alternativa aceptada si no alcanza la RAM).
#                              -1 (por defecto) = elegir segun la RAM: B necesita ~7 GB con
#                              10,000 arboles de profundidad 15, asi que con menos de 12 GB usa 2500.
#   -Julia "julia"             ejecutable de Julia
#
# Regla confirmada: sin paralelismo. Todo corre en un solo hilo.

param(
    [string]$B = "GP_CPP20_B\out\build\x64-Release\gp_b.exe",
    [string]$A = "GP_JL_A",
    [string]$C = "GP_JL_C",
    [string]$Julia = "julia",
    [string]$Data = "GP_JL_C\data\ejemplo.csv",
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    [int]$Generations = 10,
    [int]$Variables = 2,
    [switch]$SoloCrossover,
    [switch]$SoloGP,
    [switch]$IncluirA,
    [int]$ParesEnMemoria = -1
)

$ErrorActionPreference = "Stop"
$CrossDepths = "7,9,10,12,15"
$GPDepths = "7,9,12"

if (-not (Test-Path $B)) { throw "No se encontro el ejecutable de B: $B (compila en x64-Release o pasa -B)" }
foreach ($dir in @($A, $C)) {
    if (-not (Test-Path (Join-Path $dir "main.jl"))) { throw "No se encontro $dir\main.jl (pasa -A / -C)" }
}
if (-not (Get-Command $Julia -ErrorAction SilentlyContinue)) { throw "No se encontro Julia ('$Julia'); instalalo o pasa -Julia" }
if (-not $SoloCrossover -and -not (Test-Path $Data)) { throw "No se encontro el dataset: $Data" }
$B = (Resolve-Path $B).Path
$AMain = (Resolve-Path (Join-Path $A "main.jl")).Path
$CMain = (Resolve-Path (Join-Path $C "main.jl")).Path
$CDir = (Resolve-Path $C).Path
$ADir = (Resolve-Path $A).Path
if (Test-Path $Data) { $Data = (Resolve-Path $Data).Path }
$JuliaFlags = @("--startup-file=no", "-O3")  

$ramGB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
if ($ParesEnMemoria -lt 0) {
    $ParesEnMemoria = if ($ramGB -ge 12) { 0 } else { 2500 }
}
$arbolesEnMemoria = if ($ParesEnMemoria -eq 0) { 10000 } else { 2 * $ParesEnMemoria }
Write-Host ("RAM: {0:N1} GB -> {1} padres en memoria a la vez en el benchmark de crossover" -f $ramGB, $arbolesEnMemoria)

$out = "resultados_" + (Get-Date -Format "yyyyMMdd_HHmm")
New-Item -ItemType Directory -Path $out | Out-Null
Write-Host "Resultados en: $out"

# Ejecuta un benchmark y agrega sus filas a un CSV con columnas extra impl,seed.
function Invoke-Bench([string]$Impl, [string]$Exe, [string[]]$Arguments, [int]$Seed, [string]$File) {
    $lines = @(& $Exe @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "$Impl fallo con argumentos: $($Arguments -join ' ')" }
    if (-not (Test-Path $File)) {
        "impl,seed," + $lines[0] | Out-File -FilePath $File -Encoding utf8
    }
    $lines | Select-Object -Skip 1 | ForEach-Object { "$Impl,$Seed,$_" } |
        Out-File -FilePath $File -Append -Encoding utf8
    $lines | Select-Object -Skip 1 | ForEach-Object { Write-Host "  $Impl seed=$Seed  $_" }
}

# 1. Correctitud antes de medir ------------------------------------------------
Write-Host "`n[1] Pruebas de correctitud"
& $B --test; if ($LASTEXITCODE -ne 0) { throw "B no pasa --test" }
& $Julia @JuliaFlags $CMain --test; if ($LASTEXITCODE -ne 0) { throw "C no pasa --test" }
& $Julia @JuliaFlags $AMain --test; if ($LASTEXITCODE -ne 0) { throw "A no pasa --test" }
& $Julia @JuliaFlags (Join-Path $CDir "tools\cross_check.jl") $ADir $Data | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { throw "cross_check encontro diferencias entre A y C" }
& $Julia @JuliaFlags (Join-Path $CDir "tools\verificar_con_b.jl") $B $ADir $Data | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { throw "verificar_con_b encontro diferencias entre B y A/C" }

# 2. Datos del equipo (el reporte debe incluirlos) -------------------------------
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ram = [math]::Round($ramGB, 1)
$juliaVersion = & $Julia --version
@(
    "Fecha: $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
    "CPU: $($cpu.Name)",
    "Nucleos/hilos: $($cpu.NumberOfCores)/$($cpu.NumberOfLogicalProcessors)",
    "RAM: $ram GB",
    "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption)",
    "Julia: $juliaVersion (opciones: $($JuliaFlags -join ' '))",
    "Semillas: $($Seeds -join ',')",
    "Generaciones: $Generations",
    "Variables (crossover): $Variables",
    "Padres en memoria a la vez (crossover): $arbolesEnMemoria",
    "Paralelismo: ninguno (un solo hilo)",
    "Dataset: $Data"
) | Out-File -FilePath "$out\equipo_y_parametros.txt" -Encoding utf8

# 3. Benchmark de crossover -----------------------------------------------------
if (-not $SoloGP) {
    Write-Host "`n[2] Benchmark de crossover (10,000 arboles, profundidades $CrossDepths)"
    foreach ($s in $Seeds) {
        $cross = @("--bench-crossover", "--trees=10000", "--depths=$CrossDepths",
                   "--variables=$Variables", "--seed=$s", "--batch-pairs=$ParesEnMemoria")
        Invoke-Bench "B" $B $cross $s "$out\crossover_B.csv"
        Invoke-Bench "C" $Julia ($JuliaFlags + $CMain + $cross) $s "$out\crossover_C.csv"
        if ($IncluirA) { Invoke-Bench "A" $Julia ($JuliaFlags + $AMain + $cross) $s "$out\crossover_A.csv" }
    }
}

# 4. Benchmark del GP completo --------------------------------------------------
if (-not $SoloCrossover) {
    Write-Host "`n[3] GP completo (poblacion 5000, profundidades $GPDepths, $Generations generaciones)"
    foreach ($s in $Seeds) {
        $common = @("--bench-gp", "--data=$Data", "--population=5000", "--depths=$GPDepths",
                    "--generations=$Generations", "--seed=$s")
        Invoke-Bench "B" $B $common $s "$out\gp_B.csv"
        Invoke-Bench "C" $Julia ($JuliaFlags + $CMain + $common) $s "$out\gp_C.csv"
        if ($IncluirA) { Invoke-Bench "A" $Julia ($JuliaFlags + $AMain + $common) $s "$out\gp_A.csv" }
    }
}

Write-Host "`nListo. Archivos en $out"
